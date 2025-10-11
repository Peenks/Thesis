#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def device_autoselect():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class MobileNetV2Backbone(nn.Module):
    """Return 1280-D embeddings from MobileNetV2 pretrained on ImageNet."""
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


def load_split(split_dir, img_size=224):
    tfm = build_transform(img_size)
    return datasets.ImageFolder(split_dir, transform=tfm)


@torch.no_grad()
def extract_split(ds, model, device, batch_size=64, num_workers=2):
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu"),
    )
    all_embeds, all_labels, all_paths = [], [], []
    t0 = time.time()
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        emb = model(xb)
        all_embeds.append(emb.cpu())
        all_labels.append(yb.clone().cpu())
    for i in range(len(ds)):
        all_paths.append(ds.samples[i][0])

    embeds = torch.cat(all_embeds, dim=0) if all_embeds else torch.empty((0, 1280))
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long)
    dt = time.time() - t0
    return embeds, labels, all_paths, dt


def save_tensors(out_dir, split_name, embeds, labels, paths):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeds, out_dir / f"{split_name}_embeddings.pt")
    torch.save(labels, out_dir / f"{split_name}_labels.pt")
    with open(out_dir / f"{split_name}_paths.json", "w") as f:
        json.dump(paths, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Extract embeddings with MobileNetV2 backbone.")
    ap.add_argument("--data_root", default="data/processed", help="Root folder containing train/val/test")
    ap.add_argument("--out_dir", default="data/embeddings", help="Where to save .pt files")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    device = device_autoselect()
    print("Device:", device)

    model = MobileNetV2Backbone().to(device)
    model.eval()

    splits = ["train", "val", "test"]
    meta = {
        "backbone": "MobileNetV2",
        "weights": "ImageNet1K",
        "embedding_dim": 1280,
        "img_size": args.img_size,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "batch_size": args.batch_size,
        "device": str(device),
        "class_to_idx": None,
        "split_counts": {},
        "paths_files": {s: f"{s}_paths.json" for s in splits},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    train_dir = data_root / "train"
    if not train_dir.exists():
        raise SystemExit("Expected data/processed/train, val, test. Run prepare_dataset.py first.")
    ds_train = load_split(train_dir, img_size=args.img_size)
    meta["class_to_idx"] = ds_train.class_to_idx

    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"Skipping {split} (folder not found).")
            continue
        print(f"Extracting {split} from {split_dir} ...")
        ds = datasets.ImageFolder(split_dir, transform=build_transform(args.img_size))
        embeds, labels, paths, dt = extract_split(
            ds, model, device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        print(f"{split}: {embeds.shape[0]} images -> embeddings shape {tuple(embeds.shape)} in {dt:.1f}s")
        save_tensors(out_dir, split, embeds, labels, paths)
        meta["split_counts"][split] = int(embeds.shape[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("All done. Saved to", out_dir.resolve())


if __name__ == "__main__":
    main()