"""Extract frozen specialist embeddings from a CSV of H&E patches."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from train import BaseImageDataset, StudentModel


def load_backbone(checkpoint_path, model_size="base"):
    model = timm.create_model(
        StudentModel.VIT_CONFIGS[model_size]["model_name"],
        pretrained=False, num_classes=0, dynamic_img_size=True,
    )
    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "student_momentum.backbone."
    backbone = {key[len(prefix):]: value for key, value in state_dict.items()
                if key.startswith(prefix)}
    if not backbone:
        raise ValueError(f"Checkpoint lacks {prefix} weights")
    model.load_state_dict(backbone)
    return model.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="PathGTS iBOT checkpoint")
    parser.add_argument("--data_csv", required=True, help="CSV with an image_paths column")
    parser.add_argument("--output", required=True, type=Path, help="Output .npy file")
    parser.add_argument("--model_size", choices=StudentModel.VIT_CONFIGS, default="base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=224)
    args = parser.parse_args()

    paths = pd.read_csv(args.data_csv)["image_paths"].tolist()
    if not paths:
        parser.error("The input CSV contains no patches.")
    model = load_backbone(args.checkpoint, args.model_size).to(args.device)
    loader = DataLoader(
        BaseImageDataset(args, paths, mode="val"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
    )

    # Stream embeddings to disk so memory usage stays bounded for large cohorts.
    # The features follow the order of the input CSV.
    feature_dim = StudentModel.VIT_CONFIGS[args.model_size]["embed_dim"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npy_path = args.output.with_suffix(".npy")
    if args.output != npy_path:
        parser.error("--output must have a .npy suffix.")
    features = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(len(paths), feature_dim)
    )
    row = 0
    with torch.inference_mode():
        for images, _ in loader:
            batch_features = model(images.to(args.device, non_blocking=True)).float().cpu().numpy()
            features[row:row + len(images)] = batch_features
            row += len(images)
    features.flush()


if __name__ == "__main__":
    main()
