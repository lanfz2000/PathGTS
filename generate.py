"""
CytoSyn 生成脚本（H200 算力拉满版：分布式 + 多进程I/O + 异步落盘 + 极致编译）
- 针对 Hopper (H100/H200) 架构进行极限优化
- 开启 TF32、cuDNN Benchmark 硬件加速
- 开启 xFormers / Flash Attention 2
- 开启全链路 torch.compile (max-autotune)
- 开启全链路 BF16 (BFloat16) 半精度极限显存优化
- 引入 PyTorch DataLoader 实现 CPU/GPU 异步预读取
- 引入 ThreadPoolExecutor 实现全异步无阻塞落盘
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import concurrent.futures
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import DiffusionPipeline
from PIL import Image
import timm
from timm.data.transforms_factory import create_transform
from accelerate import Accelerator
from tqdm import tqdm

# ================= 硬件极限加速设置 (专为 H200/A100 打造) =================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# ========================================================================


class ReferenceImageDataset(Dataset):
    """
    专为多进程预读取设计的数据集类
    把硬盘读图和 Resize 的脏活累活全扔给 CPU 后台多线程
    """
    def __init__(self, paths: list[Path], prefixes: list[str], transform):
        self.paths = paths
        self.prefixes = prefixes
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB").resize((224, 224))
            tensor = self.transform(img)
            return tensor, self.prefixes[idx]
        except Exception as e:
            raise RuntimeError(f"Unable to load reference image: {self.paths[idx]}") from e


def save_image_async(img: Image.Image, out_path: Path, save_format: str, jpeg_quality: int):
    """供后台线程池调用的存图函数，彻底释放主进程阻塞"""
    if save_format in {"jpg", "jpeg"}:
        img.save(out_path, format="JPEG", quality=jpeg_quality)
    else:
        img.save(out_path, format=save_format.upper())


def split_h0_tokens(model, tokens: torch.Tensor):
    num_prefix_tokens = getattr(model, "num_prefix_tokens", None)
    if num_prefix_tokens is None:
        num_reg_tokens = getattr(model, "num_reg_tokens", 0)
        num_prefix_tokens = 1 + int(num_reg_tokens)
    return tokens[:, 0], tokens[:, num_prefix_tokens:]


def load_cytosyn(local_path: str, device: torch.device) -> DiffusionPipeline:
    # 🔥 极限优化 1：强制使用 BF16 半精度加载庞大的扩散模型，显存直接砍半！
    pipe = DiffusionPipeline.from_pretrained(
        local_path,
        custom_pipeline=local_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )

    pipe.set_progress_bar_config(disable=True)
    print(f"[*] 论文指定的 Scheduler 已就绪: {pipe.scheduler.__class__.__name__}")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[*] xFormers 显存优化已开启！")
    except Exception:
        print("[*] H200 已启用 PyTorch 原生 Flash Attention 2。")

    try:
        print("[*] 正在为 SiT-XL/2 (Transformer) 注入 torch.compile 编译外挂...")
        if hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune")
            print("[*] SiT Transformer 极致编译外挂注入成功！准备起飞 🚀")
        else:
            print("[!] 警告：未找到 pipe.transformer 属性，请检查 custom_pipeline 的内部实现。")
    except Exception as e:
        print(f"[!] torch.compile 注入失败，将使用原生模式运行。原因: {e}")

    return pipe.to(device)


def load_h0_mini(local_path: str, device: torch.device):
    path = Path(local_path)
    ckpt_file = path / "model.safetensors"
    if not ckpt_file.exists():
        ckpt_file = path / "pytorch_model.bin"

    if not ckpt_file.exists():
        raise FileNotFoundError(f"未找到 H0-mini 权重: {local_path}")

    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2",
        pretrained=False,
        checkpoint_path=str(ckpt_file),
        img_size=224,
        init_values=1e-05,
        num_classes=0,
        reg_tokens=4,
        mlp_ratio=5.33334,
        global_pool="",
        dynamic_img_size=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    # 🔥 极限优化 1.5：H0-mini 也使用 BF16 精确加载
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    # 🔥 极限优化 4：给 H0-mini 也打上编译补丁！
    try:
        model = torch.compile(model, mode="max-autotune")
        print("[*] H0-mini (ViT) 极致编译外挂注入成功！")
    except Exception as e:
        print(f"[!] H0-mini torch.compile 注入失败: {e}")

    transform = create_transform(
        input_size=(3, 224, 224),
        is_training=False,
        interpolation="bicubic",
        mean=[0.707223, 0.578729, 0.703617],
        std=[0.211883, 0.230117, 0.177517],
        crop_pct=1.0,
        crop_mode="center"
    )
    return model, transform


def run_conditional_batched(
    pipe,
    h0_model,
    batch_tensor: torch.Tensor,
    prefixes: list[str],
    num_images: int,
    save_top_k: int,
    steps: int,
    guidance_scale: float,
    guidance_low: float,
    guidance_high: float,
    out_dir: Path,
    device: torch.device,
    save_format: str,
    jpeg_quality: int,
    executor: concurrent.futures.ThreadPoolExecutor
):
    """支持大批量并发的推理逻辑 (已剥离 PIL 读图逻辑，纯享算力)"""

    # 确保输入张量也是 BF16 格式，防止类型不匹配报错
    batch_tensor = batch_tensor.to(device, dtype=torch.bfloat16)

    with torch.inference_mode():
        ref_tokens = h0_model(batch_tensor)
        h0_mini_embeds, _ = split_h0_tokens(h0_model, ref_tokens)
        h0_mini_embeds = h0_mini_embeds.float().clone()

    output = pipe(
        h0_mini_embeds=h0_mini_embeds,
        num_images_per_prompt=num_images,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        guidance_low=guidance_low,
        guidance_high=guidance_high,
    )

    images = output["images"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_format = save_format.lower().lstrip(".")

    rerank_enabled = save_top_k > 0 and save_top_k < num_images
    keep_k = max(1, min(save_top_k, num_images)) if rerank_enabled else num_images
    if rerank_enabled:
        rerank_tensors = []
        for img in images:
            rerank_tensors.append(
                torch.from_numpy(np.array(img.convert("RGB"), copy=True)).permute(2, 0, 1).float().div(255.0)
            )
        rerank_batch = torch.stack(rerank_tensors, dim=0).to(device=device, dtype=torch.float32)
        mean = rerank_batch.new_tensor([0.707223, 0.578729, 0.703617]).view(1, 3, 1, 1)
        std = rerank_batch.new_tensor([0.211883, 0.230117, 0.177517]).view(1, 3, 1, 1)
        rerank_batch = (rerank_batch - mean) / std

        with torch.inference_mode():
            gen_tokens = h0_model(rerank_batch.to(dtype=torch.bfloat16))
            gen_cls, _ = split_h0_tokens(h0_model, gen_tokens)

        keep_indices_per_ref = []
        for batch_idx in range(len(prefixes)):
            start = batch_idx * num_images
            end = start + num_images
            ref_embed = h0_mini_embeds[batch_idx:batch_idx + 1]
            gen_embed = gen_cls[start:end].float()
            scores = F.cosine_similarity(
                F.normalize(gen_embed, dim=-1),
                F.normalize(ref_embed.expand_as(gen_embed), dim=-1),
                dim=-1,
            )
            topk = torch.topk(scores, k=keep_k, largest=True).indices.tolist()
            keep_indices_per_ref.append(set(topk))
    else:
        keep_indices_per_ref = [set(range(num_images)) for _ in prefixes]

    save_jobs = []
    for batch_idx, prefix in enumerate(prefixes):
        for var_idx in range(num_images):
            if var_idx not in keep_indices_per_ref[batch_idx]:
                continue
            flat_idx = batch_idx * num_images + var_idx
            img = images[flat_idx].convert("RGB")
            out_path = out_dir / f"{prefix}_{var_idx:04d}.{save_format}"

            # 🔥 极限优化 3：把存盘动作一脚踢给后台线程池，主进程立刻返回去算下一个 Batch！
            save_jobs.append(executor.submit(save_image_async, img, out_path, save_format, jpeg_quality))
    return save_jobs


def get_all_indexed_reference_images(
    reference_image: str | None,
    reference_dir: str | None,
    reference_csv: str | None,
) -> list[tuple[int, Path]]:
    all_indexed_paths = []
    if reference_image:
        all_indexed_paths.append((0, Path(reference_image)))
        return all_indexed_paths
    if reference_dir:
        ref_dir = Path(reference_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for idx, path in enumerate(sorted(ref_dir.rglob("*"))):
            if path.suffix.lower() in exts:
                all_indexed_paths.append((idx, path))
        return all_indexed_paths
    if reference_csv:
        with open(reference_csv, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "image_paths" not in reader.fieldnames:
                raise ValueError("The reference CSV requires an image_paths column")
            for idx, row in enumerate(reader):
                all_indexed_paths.append((idx, Path(row["image_paths"])))
        return all_indexed_paths
    return all_indexed_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cytosyn_path", required=True)
    parser.add_argument("--finetuned_ckpt_path", default=None, help="可选：全量微调后保存的 pipeline 目录。若提供，则优先加载该目录。")
    parser.add_argument("--h0_mini_path", required=True)
    parser.add_argument("--mode", choices=["cond"], default="cond")
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference_image")
    reference.add_argument("--reference_dir")
    reference.add_argument("--reference_csv")
    parser.add_argument("--max_refs", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--save_top_k", type=int, default=0, help="保留每个 reference 的 top-k 生成结果（按 H0 cosine rerank）；<=0 表示关闭 rerank")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guidance_low", type=float, default=0.0)
    parser.add_argument("--guidance_high", type=float, default=0.75)
    parser.add_argument("--save_format", default="jpg")
    parser.add_argument("--jpeg_quality", type=int, default=95)

    # 因为有 BF16 加持，你在外层 slurm 脚本里可以大胆尝试把 Batch Size 往上拉（比如 64 甚至 128）
    parser.add_argument("--batch_size", type=int, default=32, help="单卡推理批次大小")
    # 🔥 新增参数：DataLoader 的后台进程数
    parser.add_argument("--num_workers", type=int, default=8, help="后台预读取图片的进程数")

    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"[*] H200 分布式环境就绪！共探测到 {accelerator.num_processes} 个 GPU。")
        print(f"[*] 当前单卡极限 Batch Size 设置为: {args.batch_size}")
        print(f"[*] 异步数据加载进程数 (Num Workers): {args.num_workers}")
        print(f"[*] 当前采样参数: steps={args.steps}, guidance_scale={args.guidance_scale}, guidance_high={args.guidance_high}")
        rerank_enabled = args.save_top_k > 0 and args.save_top_k < args.num_images
        print(f"[*] Best-of-N 重排: {'开启' if rerank_enabled else '关闭'} (num_images={args.num_images}, save_top_k={args.save_top_k})")
    model_path = args.finetuned_ckpt_path if args.finetuned_ckpt_path else args.cytosyn_path
    if accelerator.is_main_process:
        if args.finetuned_ckpt_path:
            print(f"[*] 使用微调后 checkpoint 进行推理: {args.finetuned_ckpt_path}")
        else:
            print(f"[*] 使用原始 CytoSyn 进行推理: {args.cytosyn_path}")

    pipe = load_cytosyn(model_path, device)
    h0_model, transform = load_h0_mini(args.h0_mini_path, device)
    out_dir = Path(args.out_dir)

    if args.mode in ["cond", "both"]:
        full_ref_list = get_all_indexed_reference_images(
            args.reference_image, args.reference_dir, args.reference_csv
        )

        if not full_ref_list:
            parser.error("No reference images found.")
        if args.max_refs > 0:
            full_ref_list = full_ref_list[:args.max_refs]

        if accelerator.is_main_process:
            print(f"[*] 准备向集群下发 {len(full_ref_list)} 条生成任务...")

        # 利用 Accelerator 完美拆分任务给各个 GPU
        with accelerator.split_between_processes(full_ref_list) as my_split_list:

            # 分离出路径和对应的前缀名
            my_paths = [Path(p) for idx, p in my_split_list]
            my_prefixes = [f"cond_{orig_idx:07d}" for orig_idx, p in my_split_list]

            # 🔥 极限优化 2：装载进 Dataset，开启并发预读 + 锁页内存 (pin_memory=True 极速搬运进 GPU)
            dataset = ReferenceImageDataset(my_paths, my_prefixes, transform)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,  # 加速 CPU 张量向 GPU 显存的拷贝速度
                shuffle=False,
                drop_last=False
            )

            disable_tqdm = not accelerator.is_local_main_process
            pbar = tqdm(dataloader, desc="GPU 0 生成进度 (按Batch计算)", disable=disable_tqdm)

            # 🔥 启动后台写盘线程池 (max_workers=8 足以应付 GPFS/Ceph 的高并发写入)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                pending_writes = []
                for batch_tensor, batch_prefixes in pbar:

                    current_writes = run_conditional_batched(
                        pipe=pipe,
                        h0_model=h0_model,
                        batch_tensor=batch_tensor,
                        prefixes=batch_prefixes,
                        num_images=args.num_images,
                        save_top_k=args.save_top_k,
                        steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        guidance_low=args.guidance_low,
                        guidance_high=args.guidance_high,
                        out_dir=out_dir,
                        device=device,
                        save_format=args.save_format,
                        jpeg_quality=args.jpeg_quality,
                        executor=executor # 传给生成函数去异步存盘
                    )

                    for write in pending_writes:
                        write.result()
                    pending_writes = current_writes
                for write in pending_writes:
                    write.result()
            # 离开 with 块时，主进程会自动阻塞等待，确保最后一张图安全写入硬盘，绝不丢数据！

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("[*] 🎉 恭喜！所有生成任务已在集群上极速完成，且文件已全部安全落盘！")


if __name__ == "__main__":
    main()
