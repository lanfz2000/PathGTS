"""
多教师知识蒸馏 + iBOT 自监督学习（精简版）
仅保留 DistillationiBOTModule 训练所需的组件
"""

import torch
torch.set_float32_matmul_precision('medium')
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List
import copy

import timm
from timm.layers import set_fused_attn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from teachers import load_teacher_models

# =============================================
set_fused_attn(True)

# ==========================================================================
# 日志记录工具
# ==========================================================================

class LossFileLogger(pl.Callback):
    """将训练/验证损失写入save_dir下的文本日志"""

    def __init__(self, save_dir: str, filename: str = "loss_log.txt"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.file_path = self.save_dir / self.filename

    def on_train_start(self, trainer, pl_module):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write("# Loss log start\n")

    def _write_metrics(self, trainer, prefix: str):
        metrics = {
            k: v for k, v in trainer.callback_metrics.items()
            if k.startswith(prefix)
        }
        if not metrics:
            return

        step = trainer.global_step
        parts = [f"step={step}"]
        for key in sorted(metrics.keys()):
            val = metrics[key]
            if hasattr(val, "detach"):
                val = val.detach().float().cpu().item()
            parts.append(f"{key}={val:.6f}")

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write("[" + prefix.rstrip("/") + "] " + " ".join(parts) + "\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % trainer.log_every_n_steps == 0:
            self._write_metrics(trainer, "train/")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._write_metrics(trainer, "val/")


class CompactStudentCheckpoint(pl.Callback):
    """Save inference-only momentum-student checkpoints for per-epoch probing."""

    def __init__(self, output_dir: str, every_n_epochs: int = 1):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        metrics = trainer.callback_metrics
        cosine = float(metrics.get("val/distill_cos_avg", float("nan")))
        ibot = float(metrics.get("val/ibot_loss", float("nan")))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"epoch-{epoch:02d}-cos-{cosine:.4f}-ibot-{ibot:.4f}.ckpt"
        state_dict = {
            f"student_momentum.{key}": value.detach().cpu()
            for key, value in pl_module.student_momentum.state_dict().items()
        }
        torch.save(
            {
                "state_dict": state_dict,
                "epoch": epoch,
                "val_distill_cos": cosine,
                "val_ibot_loss": ibot,
            },
            path,
        )
        print(f"[probe-checkpoint] saved {path}")

# ============================================================================
# 数据集定义
# ============================================================================

def is_synthetic_path(path: str) -> bool:
    return Path(path).name.startswith("cond_")


class BaseImageDataset(Dataset):
    """基础图像数据集，带错误处理"""

    def __init__(self, args, image_paths: List[str], mode: str = 'train', ratio: float = 1.0):
        """
        Args:
            image_paths: 图像路径列表
            mode: 'train' 或 'val'，决定数据增强策略
            ratio: 使用数据的比例 (0-1)
        """
        if ratio < 1.0:
            keep = max(1, int(len(image_paths) * ratio))
            rng = np.random.default_rng(getattr(args, "seed", 42))
            picked = rng.choice(len(image_paths), size=keep, replace=False)
            picked.sort()
            self.image_paths = [image_paths[i] for i in picked.tolist()]
        else:
            self.image_paths = image_paths
        self.mode = mode
        self.args = args
        self.transform = self._build_transform(args)
        self.is_synthetic = [is_synthetic_path(p) for p in self.image_paths]
        print(f"[{mode}] 加载了 {len(self.image_paths)} 张图片")

    def _build_transform(self, args):
        """构建数据增强pipeline"""
        return A.Compose([
            A.Resize(args.input_size, args.input_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """加载图像，带错误处理"""
        img_path = self.image_paths[idx]

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            return self.transform(image=image)["image"], self.is_synthetic[idx]
        except Exception as error:
            raise RuntimeError(f"Unable to load image: {img_path}") from error


class BalancedRealSyntheticBatchSampler:
    """兼容旧名称：使用比例与轮转采样"""

    def __init__(
        self,
        real_indices: List[int],
        synthetic_indices: List[int],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
        synthetic_ratio: float = 1.0,
    ):
        if batch_size < 2:
            raise ValueError("batch_size 必须>=2")
        if not real_indices or not synthetic_indices:
            raise ValueError("real_indices 和 synthetic_indices 不能为空")
        if synthetic_ratio <= 0:
            raise ValueError("synthetic_ratio 必须>0")

        self.real_indices = np.array(real_indices)
        self.synthetic_indices = np.array(synthetic_indices)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.synthetic_ratio = synthetic_ratio

        real_per = int(round(batch_size / (1.0 + synthetic_ratio)))
        real_per = max(1, min(real_per, batch_size - 1))
        synth_per = batch_size - real_per
        self.real_per = real_per
        self.synth_per = synth_per

        self.num_batches = len(self.real_indices) // self.real_per
        if not drop_last:
            self.num_batches = int(np.ceil(len(self.real_indices) / self.real_per))
        if not self.num_batches:
            raise ValueError('Each rank needs at least one complete training batch')

        self.epoch = 0

    def __iter__(self):
        real_perm = self.rng.permutation(self.real_indices)
        synth_perm = self.rng.permutation(self.synthetic_indices)

        real_offset = (self.epoch * self.real_per * self.num_batches) % len(real_perm)
        synth_offset = (self.epoch * self.synth_per * self.num_batches) % len(synth_perm)
        self.epoch += 1

        def take(indices, start, count):
            end = start + count
            if end <= len(indices):
                return indices[start:end]
            return indices[np.arange(start, end) % len(indices)]

        real_ptr = real_offset
        synth_ptr = synth_offset

        for _ in range(self.num_batches):
            real_batch = take(real_perm, real_ptr, self.real_per)
            synth_batch = take(synth_perm, synth_ptr, self.synth_per)
            real_ptr = (real_ptr + self.real_per) % len(real_perm)
            synth_ptr = (synth_ptr + self.synth_per) % len(synth_perm)

            batch = np.concatenate([real_batch, synth_batch])
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches


def get_distributed_env():
    """返回 torchrun/DDP 环境信息。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return world_size, rank, local_rank


def shard_indices_for_rank(indices: List[int], rank: int, world_size: int) -> List[int]:
    """按 rank 对索引做静态切分，避免 DDP 下每张卡重复扫全量数据。"""
    if world_size <= 1:
        return indices
    return indices[rank:(len(indices) // world_size) * world_size:world_size]


# ============================================================================
# 模型定义
# ============================================================================

class StudentModel(nn.Module):
    """学生模型: ViT (Tiny/Small/Base) + 多个Adapter (每个教师一个)"""

    # ViT模型配置
    VIT_CONFIGS = {
        'tiny': {
            'model_name': 'vit_tiny_patch16_224',
            'embed_dim': 192,
        },
        'small': {
            'model_name': 'vit_small_patch16_224',
            'embed_dim': 384,
        },
        'base': {
            'model_name': 'vit_base_patch16_224',
            'embed_dim': 768,
        }
    }

    def __init__(
        self,
        output_dims: List[int],  # 每个教师对应一个输出维度
        model_size: str = 'tiny',  # 'tiny', 'small', 'base'
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        adapter_hidden_dim: Optional[int] = None,  # Adapter中间层维度,默认为None时自动设置
    ):
        """
        Args:
            output_dims: 输出特征维度列表,对应每个教师模型
            model_size: ViT模型大小 ('tiny', 'small', 'base')
            pretrained: 是否使用预训练权重
            checkpoint_path: 预训练权重路径
            adapter_hidden_dim: Adapter中间层维度,默认为embed_dim*4
        """
        super().__init__()

        if model_size not in self.VIT_CONFIGS:
            raise ValueError(f"不支持的模型大小: {model_size}. 可选: {list(self.VIT_CONFIGS.keys())}")

        config = self.VIT_CONFIGS[model_size]
        self.model_size = model_size
        self.n_teachers = len(output_dims)
        self.embed_dim = config['embed_dim']
        self.output_dims = output_dims  # 保存output_dims供后续使用

        # 设置Adapter中间层维度
        if adapter_hidden_dim is None:
            adapter_hidden_dim = self.embed_dim * 4
        self.adapter_hidden_dim = adapter_hidden_dim

        # ViT backbone - 共享
        self.backbone = timm.create_model(
            config['model_name'],
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )

        # 加载预训练权重
        if pretrained and checkpoint_path:
            self._load_pretrained(checkpoint_path)

        # 为每个教师创建独立的Adapter
        self.adapters = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(self.embed_dim, self.adapter_hidden_dim)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(self.adapter_hidden_dim, dim))
            ]))
            for dim in output_dims
        ])

        # ========== iBOT Mask Token ==========
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        print(f"创建学生模型: model_size={model_size}, embed_dim={self.embed_dim}, "
              f"adapter_hidden_dim={self.adapter_hidden_dim}, "
              f"{self.n_teachers}个教师, 输出维度={output_dims}")

    def _load_pretrained(self, checkpoint_path: str):
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path, device='cpu')
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            for key in ('state_dict', 'model', 'model_state_dict'):
                if key in state_dict:
                    state_dict = state_dict[key]
                    break
        state_dict = {k: v for k, v in state_dict.items()
                      if k not in ('head.weight', 'head.bias')}
        self.backbone.load_state_dict(state_dict, strict=True)

    def forward(self, x: torch.Tensor, teacher_idx: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: 输入图像
            teacher_idx: 指定使用哪个adapter (None表示返回所有)

        Returns:
            如果teacher_idx指定: [B, output_dim]
            如果teacher_idx为None: List[[B, output_dim_i], ...]
        """
        features = self.backbone(x)

        if teacher_idx is not None:
            return self.adapters[teacher_idx](features)
        else:
            return [adapter(features) for adapter in self.adapters]

    # ========== 新增: iBOT辅助方法 ==========

    def get_feature_dim(self) -> int:
        """
        返回backbone输出的特征维度

        Returns:
            int: 特征维度 (例如ViT-Tiny: 192, ViT-Small: 384, ViT-Base: 768)
        """
        return self.embed_dim

    def get_patch_features(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取patch-level的特征（用于iBOT）

        Args:
            x: [B, C, H, W] 输入图像
            mask: [B, num_patches] 可选的mask，True表示被mask掉的patch

        Returns:
            patch_features: [B, num_patches, embed_dim] patch特征

        Note:
            这个方法提取ViT的patch embeddings（不包括CLS token）
            如果提供mask，会将对应的patch input替换为mask_token后再通过Transformer
        """
        # Patch embedding
        x = self.backbone.patch_embed(x)  # [B, num_patches, embed_dim]
        if x.dim() == 4:
            # 根据 embed_dim 的位置判断 shape 类型
            if x.shape[1] == self.embed_dim:
                # [B, D, H, W] -> [B, N, D]
                x = x.flatten(2).transpose(1, 2)
            elif x.shape[3] == self.embed_dim:
                # [B, H, W, D] -> [B, N, D]
                x = x.flatten(1, 2)
            else:
                # 其他情况尝试常用的 flatten(2).transpose(1, 2)
                x = x.flatten(2).transpose(1, 2)
        B, N, D = x.shape

        # 应用Masking (iBOT/MIM的核心)
        if mask is not None:
            # mask: [B, num_patches]
            # mask_token: [1, 1, D]
            m = mask.unsqueeze(-1).type_as(x)  # [B, num_patches, 1]
            x = x * (1 - m) + self.mask_token * m

        # 添加CLS token
        if hasattr(self.backbone, 'cls_token'):
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
        else:
            # 兼容没有cls_token的模型（虽然通常ViT都有）
            cls_token = torch.zeros(B, 1, D).to(x.device)

        x = torch.cat([cls_token, x], dim=1)  # [B, num_patches+1, embed_dim]

        # 添加position embedding
        if hasattr(self.backbone, 'pos_embed'):
            # 处理动态尺寸的情况
            if x.shape[1] != self.backbone.pos_embed.shape[1]:
                # 如果形状不匹配，通常意味着使用了dynamic_img_size
                # 这里简单处理，如果有更复杂的需求可以使用interpolate_pos_encoding
                pos_embed = F.interpolate(
                    self.backbone.pos_embed.permute(0, 2, 1),
                    size=x.shape[1],
                    mode='linear'
                ).permute(0, 2, 1)
            else:
                pos_embed = self.backbone.pos_embed
            x = x + pos_embed

        x = self.backbone.pos_drop(x)

        # 通过transformer blocks
        if hasattr(self.backbone, 'blocks'):
            for blk in self.backbone.blocks:
                x = blk(x)

        if hasattr(self.backbone, 'norm'):
            x = self.backbone.norm(x)

        # 去掉CLS token
        patch_features = x[:, 1:, :]  # [B, num_patches, embed_dim]

        return patch_features

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取CLS token的特征（用于全局特征）

        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            cls_features: [B, embed_dim] CLS token特征

        Note:
            这与forward方法返回的特征相同，但更明确
        """
        features = self.backbone(x)  # [B, embed_dim]
        return features

    def forward_backbone_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        只通过backbone，不经过adapters（用于iBOT）

        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            features: [B, embed_dim] backbone输出的CLS token特征
        """
        return self.backbone(x)

    def get_num_patches(self, img_size: int = 224, patch_size: int = 16) -> int:
        """
        计算给定图像大小下的patch数量

        Args:
            img_size: 图像尺寸
            patch_size: patch尺寸

        Returns:
            num_patches: patch数量
        """
        return (img_size // patch_size) ** 2

    # ========== 原有方法保持不变 ==========

    def get_num_params(self, trainable_only: bool = False) -> int:
        """获取参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def print_trainable_params(self):
        """打印可训练参数信息"""
        total_params = self.get_num_params(trainable_only=False)
        trainable_params = self.get_num_params(trainable_only=True)
        print(f"总参数: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"可训练比例: {100 * trainable_params / total_params:.2f}%")


class iBOTHead(nn.Module):
    """iBOT projection head"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 8192,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
        norm_last_layer: bool = True
    ):
        super().__init__()

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MaskGenerator:
    """生成随机mask用于masked image modeling"""

    def __init__(
        self,
        input_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.4,
        mask_prob_adjust: float = 0.0
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.mask_prob_adjust = mask_prob_adjust

    def __call__(self, batch_size: int) -> torch.Tensor:
        """
        Returns:
            mask: [B, num_patches] 布尔张量，True表示被mask的patch
        """
        num_masked = int(self.num_patches * self.mask_ratio)

        mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool)

        for i in range(batch_size):
            # 随机选择要mask的patch
            masked_indices = torch.randperm(self.num_patches)[:num_masked]
            mask[i, masked_indices] = True

        return mask

# ============================================================================
# 损失函数
# ============================================================================

class KLDivergenceLoss(nn.Module):
    """KL散度损失 (用于知识蒸馏)"""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        reduction: str = "batchmean",
    ) -> torch.Tensor:
        student_prob = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)

        if reduction == "none":
            kl = F.kl_div(student_prob.log(), teacher_prob, reduction="none")
            return kl.sum(dim=-1) * (self.temperature ** 2)

        kl_loss = F.kl_div(student_prob.log(), teacher_prob, reduction=reduction)
        return kl_loss * (self.temperature ** 2)


class FeatureCorrelationLoss(nn.Module):
    """
    特征相关性对齐损失 (借鉴 Barlow Twins)
    对齐学生和教师特征的自相关矩阵，确保特征维度的解耦倾向一致
    """
    def __init__(self, lambd=0.005):
        super().__init__()
        self.lambd = lambd

    def forward(self, z1, z2):
        # 标准化特征
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

        B = z1.size(0)
        c = torch.mm(z1.T, z2) / B # 互相关矩阵 [D, D]

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()

        return on_diag + self.lambd * off_diag

class ContrastiveDistillationLoss(nn.Module):
    """
    对比蒸馏损失 (InfoNCE)
    强迫学生特征与其对应的教师特征在空间中尽可能接近
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_feat, teacher_feat):
        # 归一化到单位球面
        student_feat = F.normalize(student_feat, dim=-1)
        teacher_feat = F.normalize(teacher_feat, dim=-1)

        # 计算相似度矩阵 [B, B]
        logits = torch.mm(student_feat, teacher_feat.T) / self.temperature

        # 目标是对角线元素（对应的教师特征）
        labels = torch.arange(student_feat.size(0)).to(student_feat.device)
        loss = F.cross_entropy(logits, labels)

        return loss

# ============================================================================
# DistillationiBOTModule
# ============================================================================

# ============================================================================
# DistillationiBOTModule
# ============================================================================
class DistillationiBOTModule(pl.LightningModule):
    """多教师知识蒸馏 + iBOT自监督学习模块"""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # ========== 知识蒸馏部分 ==========
        teachers_list, output_dims = load_teacher_models(args.teachers, args)
        self.teachers = nn.ModuleList(teachers_list)
        self.n_teachers = len(self.teachers)

        # 学生模型
        self.student = StudentModel(
            output_dims=output_dims,
            pretrained=True,
            checkpoint_path=args.student_ckpt,
            model_size=args.model_size
        )

        # 可选的高级损失函数
        self.use_correlation = getattr(args, 'use_correlation', False)
        self.use_contrastive = getattr(args, 'use_contrastive', False)

        if self.use_correlation:
            self.corr_loss = FeatureCorrelationLoss(lambd=getattr(args, 'corr_lambd', 0.005))
        if self.use_contrastive:
            self.cont_loss = ContrastiveDistillationLoss(temperature=getattr(args, 'cont_temp', 0.07))

        # 教师权重
        if args.teacher_weights is not None:
            self.teacher_weights = args.teacher_weights
        else:
            self.teacher_weights = [1.0 / self.n_teachers] * self.n_teachers

        # ========== iBOT自监督部分 ==========
        student_dim = self.student.get_feature_dim()

        # iBOT patch-level projection heads
        self.ibot_head = iBOTHead(
            in_dim=student_dim,
            out_dim=getattr(args, 'ibot_out_dim', 8192),
            hidden_dim=getattr(args, 'ibot_hidden_dim', 2048),
            bottleneck_dim=getattr(args, 'ibot_bottleneck_dim', 256),
            nlayers=getattr(args, 'ibot_nlayers', 3),
            norm_last_layer=True
        )

        # Momentum模型
        import copy
        self.student_momentum = copy.deepcopy(self.student)
        for param in self.student_momentum.parameters():
            param.requires_grad = False

        self.student_momentum.load_state_dict(self.student.state_dict())

        self.ibot_head_momentum = iBOTHead(
            in_dim=student_dim,
            out_dim=getattr(args, 'ibot_out_dim', 8192),
            hidden_dim=getattr(args, 'ibot_hidden_dim', 2048),
            bottleneck_dim=getattr(args, 'ibot_bottleneck_dim', 256),
            nlayers=getattr(args, 'ibot_nlayers', 3),
            norm_last_layer=True
        )
        self.ibot_head_momentum.load_state_dict(self.ibot_head.state_dict())

        # Momentum模型不需要梯度
        for param in self.student_momentum.parameters():
            param.requires_grad = False
        for param in self.ibot_head_momentum.parameters():
            param.requires_grad = False

        # Mask生成器
        self.mask_generator = MaskGenerator(
            input_size=getattr(args, 'input_size', 224),
            patch_size=getattr(args, 'patch_size', 16),
            mask_ratio=getattr(args, 'mask_ratio', 0.3)
        )

        # 超参数
        self.ibot_lambda_start = getattr(args, 'ibot_lambda_start', 0.1)
        self.ibot_lambda_end = getattr(args, 'ibot_lambda_end', 0.3)
        self.distill_lambda = getattr(args, 'distill_lambda', 1.0)
        self.momentum = getattr(args, 'momentum', 0.996)
        self.center_momentum = getattr(args, 'center_momentum', 0.9)
        self.ibot_student_temp = getattr(args, 'ibot_student_temp', 0.1)
        self.ibot_teacher_temp = getattr(args, 'ibot_teacher_temp', 0.04)
        self.synthetic_alpha_start = getattr(args, 'synthetic_alpha_start', 0.0)
        self.synthetic_alpha_end = getattr(args, 'synthetic_alpha_end', 0.5)
        self.schedule_warmup_ratio = getattr(args, 'schedule_warmup_ratio', 1.0)
        self.mgd_lambda = getattr(args, 'mgd_lambda', 0.0)
        self.mgd_loss_type = getattr(args, 'mgd_loss_type', 'l2')

        # iBOT center
        self.register_buffer(
            'center',
            torch.zeros(1, getattr(args, 'ibot_out_dim', 8192))
        )

        print(f"教师权重: {self.teacher_weights}")
        print(f"iBOT损失权重: {self.ibot_lambda_start} -> {self.ibot_lambda_end}, 蒸馏损失权重: {self.distill_lambda}")
        print(f"合成图像损失权重系数: {self.synthetic_alpha_start} -> {self.synthetic_alpha_end}")
        print(f"MGD损失权重: {self.mgd_lambda}, 类型: {self.mgd_loss_type}")

    def train(self, mode: bool = True):
        """重写train方法，确保教师和momentum模型始终处于eval模式"""
        super().train(mode)
        self.teachers.eval()
        self.student_momentum.eval()
        self.ibot_head_momentum.eval()
        return self

    @torch.no_grad()
    def _update_momentum_model(self):
        """使用固定momentum的EMA更新"""
        m = self.momentum
        for param, param_m in zip(self.student.parameters(), self.student_momentum.parameters()):
            param_m.data.mul_(m).add_((1 - m) * param.detach().data)
        for param, param_m in zip(self.ibot_head.parameters(), self.ibot_head_momentum.parameters()):
            param_m.data.mul_(m).add_((1 - m) * param.detach().data)

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """更新iBOT center"""
        if teacher_output.dim() == 2:
            batch_center = teacher_output.mean(dim=0, keepdim=True)
        else:
            batch_center = teacher_output.mean(dim=tuple(range(teacher_output.dim() - 1)), keepdim=True)
            batch_center = batch_center.reshape(1, -1)

        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()

        self.center.mul_(self.center_momentum)
        self.center.add_(batch_center, alpha=(1 - self.center_momentum))

    def _linear_schedule(self, start: float, end: float) -> float:
        if end == start:
            return float(end)
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None or total_steps <= 0:
            return float(end)
        warmup_steps = max(1, int(total_steps * self.schedule_warmup_ratio))
        progress = min(1.0, float(self.global_step) / float(warmup_steps))
        return float(start + (end - start) * progress)

    def forward_distillation(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """知识蒸馏的forward"""
        teacher_feats = []
        with torch.no_grad():
            for teacher in self.teachers:
                teacher_feats.append(teacher(x))
        student_feats = self.student(x)
        return student_feats, teacher_feats

    def forward_ibot(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        iBOT patch-level forward

        Returns:
            student_output: [N_masked, out_dim]
            teacher_output_centered: [N_masked, out_dim]
            teacher_output_raw: [N_masked, out_dim]
        """
        student_patch_feats = self.student.get_patch_features(x, mask=mask)
        B, N, D = student_patch_feats.shape

        if mask is not None:
            if mask.device != student_patch_feats.device:
                mask = mask.to(student_patch_feats.device)
            student_patch_feats_flat = student_patch_feats.reshape(-1, D)
            mask_flat = mask.reshape(-1)
            student_masked_feats = student_patch_feats_flat[mask_flat]
            student_output = self.ibot_head(student_masked_feats)
        else:
            student_output = self.ibot_head(student_patch_feats.reshape(B * N, D))

        with torch.no_grad():
            teacher_patch_feats = self.student_momentum.get_patch_features(x, mask=None)
            if mask is not None:
                teacher_patch_feats_flat = teacher_patch_feats.reshape(-1, D)
                teacher_masked_feats = teacher_patch_feats_flat[mask_flat]
                teacher_output_raw = self.ibot_head_momentum(teacher_masked_feats)
            else:
                teacher_output_raw = self.ibot_head_momentum(teacher_patch_feats.reshape(B * N, D))
            teacher_output_centered = teacher_output_raw - self.center

        return student_output, teacher_output_centered, teacher_output_raw

    def ibot_loss(
        self,
        student_output,
        teacher_output,
        mask: torch.Tensor,
        student_temp=0.1,
        teacher_temp=0.04,
    ) -> torch.Tensor:
        """iBOT cross-entropy损失（返回每个样本的loss）"""
        student_log_prob = F.log_softmax(student_output / student_temp, dim=-1)
        teacher_prob = F.softmax(teacher_output / teacher_temp, dim=-1)
        loss_per_patch = -torch.sum(teacher_prob * student_log_prob, dim=-1)

        B = mask.shape[0]
        patch_count = mask.shape[1]
        mask_flat = mask.reshape(-1)
        sample_ids = torch.arange(B, device=mask.device).repeat_interleave(patch_count)
        sample_ids = sample_ids[mask_flat]

        loss_per_sample = torch.zeros(B, device=loss_per_patch.device)
        counts = torch.zeros(B, device=loss_per_patch.device)
        loss_per_sample.scatter_add_(0, sample_ids, loss_per_patch)
        counts.scatter_add_(0, sample_ids, torch.ones_like(loss_per_patch))
        loss_per_sample = loss_per_sample / counts.clamp_min(1)
        return loss_per_sample

    def mgd_loss(
        self,
        student_patch_feats: torch.Tensor,
        teacher_patch_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked Generative Distillation loss (per-sample)."""
        B, N, D = student_patch_feats.shape
        mask_flat = mask.reshape(-1)
        student_flat = student_patch_feats.reshape(B * N, D)[mask_flat]
        teacher_flat = teacher_patch_feats.reshape(B * N, D)[mask_flat]

        if self.mgd_loss_type == 'cos':
            student_flat = F.normalize(student_flat, dim=-1)
            teacher_flat = F.normalize(teacher_flat, dim=-1)
            loss_per_patch = 1.0 - torch.sum(student_flat * teacher_flat, dim=-1)
        else:
            loss_per_patch = F.mse_loss(student_flat, teacher_flat, reduction='none').mean(dim=-1)

        sample_ids = torch.arange(B, device=mask.device).repeat_interleave(N)
        sample_ids = sample_ids[mask_flat]
        loss_per_sample = torch.zeros(B, device=loss_per_patch.device)
        counts = torch.zeros(B, device=loss_per_patch.device)
        loss_per_sample.scatter_add_(0, sample_ids, loss_per_patch)
        counts.scatter_add_(0, sample_ids, torch.ones_like(loss_per_patch))
        loss_per_sample = loss_per_sample / counts.clamp_min(1)
        return loss_per_sample

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """训练步骤：蒸馏 + iBOT"""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, is_synthetic = batch
        else:
            images = batch
            is_synthetic = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)

        is_synthetic = is_synthetic.to(images.device)
        current_synth_alpha = self._linear_schedule(self.synthetic_alpha_start, self.synthetic_alpha_end)
        current_ibot_lambda = self._linear_schedule(self.ibot_lambda_start, self.ibot_lambda_end)
        weights = torch.where(
            is_synthetic,
            torch.full_like(is_synthetic, current_synth_alpha, dtype=torch.float32),
            torch.ones_like(is_synthetic, dtype=torch.float32),
        )

        # 1. 知识蒸馏损失
        student_feats, teacher_feats = self.forward_distillation(images)

        distill_loss = 0
        loss_dict = {}

        for i in range(self.n_teachers):
            if student_feats[i].device != teacher_feats[i].device:
                teacher_feats[i] = teacher_feats[i].to(student_feats[i].device)

            student_norm = F.normalize(student_feats[i], dim=1)
            teacher_norm = F.normalize(teacher_feats[i], dim=1)
            cos_per_sample = F.cosine_similarity(student_norm, teacher_norm, dim=1)
            loss_cos_vec = 1.0 - cos_per_sample
            cos_sim = cos_per_sample.mean()

            extra_distill_loss = 0
            if self.use_correlation:
                extra_distill_loss += self.corr_loss(student_feats[i], teacher_feats[i])
            if self.use_contrastive:
                extra_distill_loss += self.cont_loss(student_feats[i], teacher_feats[i])

            weighted_loss = loss_cos_vec * weights
            teacher_loss = weighted_loss.mean() + extra_distill_loss * weights.mean()
            distill_loss += self.teacher_weights[i] * teacher_loss

            teacher_name = self.args.teachers[i]
            loss_dict[f'train/distill_cos_loss_{teacher_name}'] = loss_cos_vec.mean()
            loss_dict[f'train/distill_cos_{teacher_name}'] = cos_sim

        # 2. iBOT patch-level MIM损失
        mask = self.mask_generator(images.size(0)).to(images.device)
        student_ibot, teacher_ibot_centered, teacher_ibot_raw = self.forward_ibot(images, mask)

        ibot_loss_vec = self.ibot_loss(
            student_ibot,
            teacher_ibot_centered,
            mask=mask,
            student_temp=self.ibot_student_temp,
            teacher_temp=self.ibot_teacher_temp,
        )
        ibot_loss = (ibot_loss_vec * weights).mean()
        self._update_center(teacher_ibot_raw)

        # 2.5 MGD: mask 后重构 teacher patch 特征
        mgd_loss = torch.tensor(0.0, device=images.device)
        if self.mgd_lambda > 0:
            student_patch_feats = self.student.get_patch_features(images, mask=mask)
            with torch.no_grad():
                teacher_patch_feats = self.student_momentum.get_patch_features(images, mask=None)
            mgd_loss_vec = self.mgd_loss(student_patch_feats, teacher_patch_feats, mask)
            mgd_loss = (mgd_loss_vec * weights).mean()

        # 3. 总损失
        total_loss = (
            self.distill_lambda * distill_loss
            + current_ibot_lambda * ibot_loss
            + self.mgd_lambda * mgd_loss
        )

        # 更新momentum模型
        self._update_momentum_model()

        # 记录损失
        loss_dict.update({
            'train/loss': total_loss,
            'train/loss_ibot': ibot_loss,
            'train/loss_mgd': mgd_loss,
            'train/loss_distill': distill_loss,
            'train/ibot_lambda': current_ibot_lambda,
            'train/synthetic_alpha': current_synth_alpha,
            'train/momentum': self.momentum,
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        })

        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        with torch.no_grad():
            # 蒸馏验证
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, _ = batch
            else:
                images = batch

            student_feats, teacher_feats = self.forward_distillation(images)

            loss_dict = {}

            for i in range(self.n_teachers):
                if student_feats[i].device != teacher_feats[i].device:
                    teacher_feats[i] = teacher_feats[i].to(student_feats[i].device)

                loss_l1 = F.l1_loss(student_feats[i], teacher_feats[i])
                cos_sim = F.cosine_similarity(student_feats[i], teacher_feats[i], dim=1).mean()

                teacher_name = self.args.teachers[i]
                loss_dict[f'val/distill_l1_{teacher_name}'] = loss_l1
                loss_dict[f'val/distill_cos_{teacher_name}'] = cos_sim

            avg_cos_sim = sum([loss_dict[f'val/distill_cos_{name}'] for name in self.args.teachers]) / self.n_teachers
            loss_dict['val/distill_cos_avg'] = avg_cos_sim

            # iBOT验证
            mask = self.mask_generator(images.size(0)).to(images.device)
            student_ibot, teacher_ibot_centered, _ = self.forward_ibot(images, mask)
            ibot_loss_vec = self.ibot_loss(
                student_ibot,
                teacher_ibot_centered,
                mask=mask,
                student_temp=self.ibot_student_temp,
                teacher_temp=self.ibot_teacher_temp,
            )
            loss_dict['val/ibot_loss'] = ibot_loss_vec.mean()

            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        param_groups = [
            {
                'params': self.student.parameters(),
                'lr': self.args.lr,
                'name': 'student'
            },
            {
                'params': self.ibot_head.parameters(),
                'lr': self.args.lr * getattr(self.args, 'ibot_lr_scale', 1.0),
                'name': 'ibot_head'
            }
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=getattr(self.args, 'weight_decay', 0.04)
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6
        )

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

# ============================================================================
# 数据处理工具
# ============================================================================

def extract_slide_id(patch_path: str) -> str:
    """从patch路径中提取slide_id"""
    filename = Path(patch_path).name
    if '_patch_' in filename:
        return filename.split('_patch_')[0]
    return Path(filename).stem


def split_by_slide(
    image_paths: List[str],
    split_csv_path: str,
    slide_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据slide级别的split信息划分patches

    Returns:
        (train_images, val_images, test_images)
    """
    print(f"\n正在读取split CSV: {split_csv_path}")
    split_df = pd.read_csv(split_csv_path, dtype={'slide_id': str})

    # 创建slide_id到split的映射
    slide_to_split = dict(zip(split_df['slide_id'], split_df['split']))
    print(f"加载了 {len(slide_to_split)} 个slide的split信息")

    # 统计
    split_counts = split_df['split'].value_counts()
    print("\nSlide划分统计:")
    for split_name in ['train', 'val', 'test']:
        if split_name in split_counts:
            print(f"  {split_name}: {split_counts[split_name]} slides")

    # 划分patches
    train_imgs, val_imgs, test_imgs, unmatched = [], [], [], []
    if slide_ids is None:
        slide_ids = [extract_slide_id(path) for path in image_paths]
    for patch_path, slide_id in zip(image_paths, slide_ids):
        split = slide_to_split.get(slide_id)

        if split == 'train':
            train_imgs.append(patch_path)
        elif split == 'val':
            val_imgs.append(patch_path)
        elif split == 'test':
            test_imgs.append(patch_path)
        else:
            unmatched.append(patch_path)

    # 转换为numpy array
    train_imgs = np.array(train_imgs)
    val_imgs = np.array(val_imgs)
    test_imgs = np.array(test_imgs)

    # 打印统计
    print(f"\nPatch划分结果:")
    print(f"  Train: {len(train_imgs)} patches")
    print(f"  Val:   {len(val_imgs)} patches")
    print(f"  Test:  {len(test_imgs)} patches")
    print(f"  Total: {len(train_imgs) + len(val_imgs) + len(test_imgs)} patches")

    if unmatched:
        print(f"\n警告: {len(unmatched)} 个patches未能匹配")
        print("未匹配示例 (前5个):")
        for patch in unmatched[:5]:
            print(f"  - {Path(patch).name} (slide_id: {extract_slide_id(patch)})")

    return train_imgs, val_imgs, test_imgs


# ============================================================================
# 主训练函数
# ============================================================================

def train_distillation_ibot(args):
    """知识蒸馏 + iBOT 联合训练"""
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger

    print("\n" + "="*80)
    print("开始知识蒸馏 + iBOT 联合训练")
    print("="*80)

    # 设置随机种子
    pl.seed_everything(args.seed)

    # ========== 数据加载 ==========
    print("\n加载数据...")
    manifest = pd.read_csv(args.data_csv, dtype={'slide_id': str})
    image_paths = manifest['image_paths'].tolist()
    slide_ids = manifest['slide_id'].tolist() if 'slide_id' in manifest else None

    train_images, val_images, _ = split_by_slide(image_paths, args.split_csv, slide_ids)
    if not len(train_images) or not len(val_images):
        raise ValueError('The slide split must provide both training and validation patches.')

    world_size, rank, local_rank = get_distributed_env()
    use_external_ddp = world_size > 1
    if use_external_ddp:
        print(f"[DDP] WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}")

    train_dataset = BaseImageDataset(
        args,
        image_paths=train_images,
        mode='train',
        ratio=args.ratio
    )
    val_dataset = BaseImageDataset(
        args,
        image_paths=val_images[:10000],
        mode='val',
        ratio=1.0
    )

    real_indices = [i for i, is_syn in enumerate(train_dataset.is_synthetic) if not is_syn]
    synth_indices = [i for i, is_syn in enumerate(train_dataset.is_synthetic) if is_syn]
    real_indices = shard_indices_for_rank(real_indices, rank, world_size)
    synth_indices = shard_indices_for_rank(synth_indices, rank, world_size)
    if use_external_ddp:
        print(f"[DDP] rank={rank} shard sizes: real={len(real_indices)}, synthetic={len(synth_indices)}")
    batch_sampler = BalancedRealSyntheticBatchSampler(
        real_indices=real_indices,
        synthetic_indices=synth_indices,
        batch_size=args.bs,
        drop_last=True,
        seed=args.seed,
        synthetic_ratio=args.synthetic_ratio,
    )
    val_sampler = None
    if use_external_ddp:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 16,
        pin_memory=True,
        persistent_workers=(args.num_workers if hasattr(args, 'num_workers') else 16) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 16,
        pin_memory=True
    )

    args.data_loader_len = len(train_loader)

    print("\n创建模型...")
    model = DistillationiBOTModule(args)

    callbacks = [
        CompactStudentCheckpoint(
            Path(args.save_dir) / "probe_checkpoints",
            every_n_epochs=getattr(args, "compact_save_every_n_epochs", 1),
        )
    ]
    checkpoint_callback = ModelCheckpoint(
        monitor='val/distill_cos_avg',
        mode='max',
        save_top_k=3,
        filename='best-{epoch:02d}-{val/distill_cos_avg:.4f}',
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)

    checkpoint_ibot = ModelCheckpoint(
        monitor='val/ibot_loss',
        mode='min',
        save_top_k=1,
        filename='best_ibot-{epoch:02d}-{val/ibot_loss:.4f}',
        verbose=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_ibot)

    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(LossFileLogger(args.save_dir, filename="distill_ibot_loss.txt"))

    experiment_name = f"distill_ibot_{'_'.join(args.teachers)}"
    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name=experiment_name,
        version=f"seed_{args.seed}"
    )

    ddp_strategy = DDPStrategy(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=False,
    )

    use_ddp = args.devices > 1 or getattr(args, 'num_nodes', 1) > 1 or use_external_ddp
    trainer_devices = int(os.environ.get('LOCAL_WORLD_SIZE', args.devices)) if use_external_ddp else args.devices
    if use_ddp:
        print(
            f"[DDP] trainer setup: devices={trainer_devices}, num_nodes={getattr(args, 'num_nodes', 1)}, "
            f"strategy=ddp, launched_via_torchrun={use_external_ddp}"
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision='bf16-mixed',
        accelerator='auto',
        devices=trainer_devices,
        num_nodes=getattr(args, 'num_nodes', 1),
        strategy=ddp_strategy if use_ddp else 'auto',
        use_distributed_sampler=False,
        check_val_every_n_epoch=getattr(args, 'val_check_interval', 1),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=getattr(args, 'gradient_clip_val', 1.0),
        log_every_n_steps=getattr(args, 'log_every_n_steps', 50),
        sync_batchnorm=use_ddp,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        accumulate_grad_batches=getattr(args, 'accumulate_grad_batches', 1),
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer, model

# ============================================================================
# 命令行入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
       description="多教师知识蒸馏 + 自监督学习训练框架",
       formatter_class=argparse.ArgumentDefaultsHelpFormatter
   )

    # 数据相关
    parser.add_argument('--data_csv', type=str, required=True,
                       help='训练数据CSV路径')
    parser.add_argument('--split_csv', type=str, required=True,
                       help='Slide级别split信息CSV路径')
    parser.add_argument('--ratio', type=float, default=1.0,
                       help='使用数据的比例')

    # 训练相关
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--bs', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--devices', type=int, default=1,
                       help='使用的GPU数量')
    parser.add_argument('--num_nodes', type=int, default=1,
                       help='使用的GPU nodes')
    parser.add_argument('--num_workers', type=int, default=16,
                              help='数据加载线程数')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                              help='日志记录步长')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                              help='梯度裁剪阈值')
    parser.add_argument('--val_check_interval', type=int, default=1,
                              help='验证间隔(epoch)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                              help='梯度累积步数')

    # 模型相关 - 改为支持多个教师
    parser.add_argument('--model_size', choices=StudentModel.VIT_CONFIGS, default='base')
    parser.add_argument('--teachers', type=str, nargs='+',
                       default=['virchow2'],
                       choices=['virchow2', 'conchv1.5', 'uni'],
                       help='教师模型列表,如: --teachers conchv1.5 uni virchow2')
    parser.add_argument('--teacher_weights', type=float, nargs='+',
                       default=None,
                       help='教师权重列表,如: --teacher_weights 0.5 0.3 0.2 (需与教师数量一致)')
    parser.add_argument('--student_ckpt', required=True, help='AugReg student weights')
    parser.add_argument('--virchow2_dir', help='Local Virchow2 Hugging Face model directory')
    parser.add_argument('--titan_dir', help='Local TITAN model directory for CONCH v1.5')
    parser.add_argument('--uni_ckpt', help='Local UNI pytorch_model.bin')

    # 损失相关
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='保留参数占位；当前简化主蒸馏默认不使用KL')
    parser.add_argument('--use_correlation', action='store_true',
                       help='是否使用特征相关性损失(Barlow Twins)')
    parser.add_argument('--use_contrastive', action='store_true',
                       help='是否使用对比蒸馏损失(InfoNCE)')
    parser.add_argument('--corr_lambd', type=float, default=0.005,
                       help='相关性损失权重因子')
    parser.add_argument('--cont_temp', type=float, default=0.07,
                       help='对比蒸馏温度')

    # iBOT相关
    parser.add_argument('--ibot_lambda_start', type=float, default=0.1,
                              help='iBOT损失权重起始值')
    parser.add_argument('--ibot_lambda_end', type=float, default=0.3,
                              help='iBOT损失权重结束值')
    parser.add_argument('--distill_lambda', type=float, default=1.0,
                              help='蒸馏损失权重')
    parser.add_argument('--ibot_out_dim', type=int, default=8192,
                              help='iBOT输出维度')
    parser.add_argument('--ibot_hidden_dim', type=int, default=2048,
                              help='iBOT隐藏层维度')
    parser.add_argument('--ibot_bottleneck_dim', type=int, default=256,
                              help='iBOT瓶颈层维度')
    parser.add_argument('--ibot_nlayers', type=int, default=2,
                              help='iBOT头部层数')
    parser.add_argument('--ibot_lr_scale', type=float, default=1.0,
                              help='iBOT头部学习率缩放')
    parser.add_argument('--ibot_student_temp', type=float, default=0.1,
                              help='iBOT学生温度')
    parser.add_argument('--ibot_teacher_temp', type=float, default=0.04,
                              help='iBOT教师温度 (初始)')
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                              help='iBOT Mask比例')
    parser.add_argument('--momentum', type=float, default=0.996,
                              help='EMA动量因子 (初始)')
    parser.add_argument('--center_momentum', type=float, default=0.9,
                              help='iBOT中心更新动量')
    parser.add_argument('--input_size', type=int, default=224,
                              help='输入图像尺寸')
    parser.add_argument('--patch_size', type=int, default=16,
                              help='Patch尺寸')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                              help='权重衰减 (初始)')
    parser.add_argument('--synthetic_ratio', type=float, default=2.0,
                              help='合成:真实 比例 (n 倍, 每batch合成=真实*n)')
    parser.add_argument('--synthetic_alpha_start', type=float, default=0.0,
                              help='合成图像损失权重起始值 (真实=1.0)')
    parser.add_argument('--synthetic_alpha_end', type=float, default=0.5,
                              help='合成图像损失权重结束值 (真实=1.0)')
    parser.add_argument('--schedule_warmup_ratio', type=float, default=1.0,
                              help='iBOT/synthetic 权重线性拉升所覆盖的训练比例')
    parser.add_argument('--mgd_lambda', type=float, default=0.0,
                              help='MGD损失权重 (0关闭)')
    parser.add_argument('--mgd_loss_type', type=str, default='cos', choices=['l2', 'cos'],
                              help='MGD损失类型 (l2/cos)')

    # 保存相关
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--compact_save_every_n_epochs', type=int, default=1,
                       help='每隔多少个epoch保存一个用于probe的compact momentum student checkpoint')

    args = parser.parse_args()

   # 验证teacher_weights
    if args.teacher_weights is not None:
        if len(args.teacher_weights) != len(args.teachers):
           raise ValueError(f"teacher_weights长度({len(args.teacher_weights)}) 必须等于teachers数量({len(args.teachers)})")
        if abs(sum(args.teacher_weights) - 1.0) > 1e-5:
           print(f"警告: teacher_weights总和不为1.0 ({sum(args.teacher_weights)}), 将自动归一化")
           total = sum(args.teacher_weights)
           args.teacher_weights = [w / total for w in args.teacher_weights]

    # 打印配置
    print("\n" + "="*80)
    print("训练配置:")
    print("="*80)
    for arg, value in sorted(vars(args).items()):
       print(f"  {arg:20s}: {value}")
    print("="*80 + "\n")

    # 只保留 distill_ibot
    train_distillation_ibot(args)
