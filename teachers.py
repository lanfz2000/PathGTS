"""Frozen teachers used for specialist distillation."""

import timm
import torch
from torch import nn
from timm.layers import SwiGLUPacked


class Virchow2(nn.Module):
    def __init__(self, model_dir=None):
        super().__init__()
        source = f"local-dir:{model_dir}" if model_dir else "hf-hub:paige-ai/Virchow2"
        self.model = timm.create_model(
            source,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
        )

    def forward(self, images):
        tokens = self.model(images)
        return torch.cat([tokens[:, 0], tokens[:, 5:].mean(dim=1)], dim=-1)


def load_teacher_models(names, args):
    teachers, dimensions = [], []
    for name in names:
        if name == "virchow2":
            model, dim = Virchow2(args.virchow2_dir), 2560
        elif name == "conchv1.5":
            from transformers import AutoModel

            if not args.titan_dir:
                raise ValueError("CONCH v1.5 requires --titan_dir pointing to a TITAN snapshot")
            titan = AutoModel.from_pretrained(
                args.titan_dir, trust_remote_code=True, local_files_only=True
            )
            model, _ = titan.return_conch()
            dim = 768
        elif name == "uni":
            if not args.uni_ckpt:
                raise ValueError("UNI requires --uni_ckpt pointing to pytorch_model.bin")
            model = timm.create_model(
                "vit_large_patch16_224", init_values=1e-5,
                num_classes=0, dynamic_img_size=True,
            )
            model.load_state_dict(torch.load(args.uni_ckpt, map_location="cpu", weights_only=True))
            dim = 1024
        else:
            raise ValueError(f"Unknown teacher: {name}")
        teachers.append(model.float().eval().requires_grad_(False))
        dimensions.append(dim)
    return teachers, dimensions
