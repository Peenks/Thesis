import os
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==== CONFIG ====
DATA_SPLIT_DIR = "data/split"          # where train/val/test image folders are
EMBEDDINGS_DIR = "data/embeddings"     # where we'll save .pt files
BATCH_SIZE = 64
NUM_WORKERS = 4
IMAGE_SIZE = 224

# 🔴 IMPORTANT: change this if your checkpoint is in a different path or name
CHECKPOINT_PATH = "models/simclr_encoder.pth"


# ==== TRANSFORMS (match your SimCLR encoder input size/normalization) ====
def get_base_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # ImageNet mean/std (commonly used with MobileNetV2)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ==== LOAD ENCODER ====
def load_encoder(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Loads a trained SimCLR encoder from a .pth/.pt file.
    Assumes the checkpoint contains a torch.nn.Module that returns embeddings.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Please check the path/filename."
        )

    model = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

    # If your checkpoint is a dict like {"model": encoder, ...}, adjust here:
    # model = torch.load(checkpoint_path, map_location=device)["encoder"]

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model.to(device)


# ==== DATA LOADER FOR A SPLIT (train/val/test) ====
def get_dataloader(split_name: str, transform, batch_size: int, num_workers: int):
    split_dir = os.path.join(DATA_SPLIT_DIR, split_name)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    dataset = datasets.ImageFolder(root=split_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,   # important: order is fixed so labels match embeddings
        num_workers=num_workers,
    )

    return dataloader, dataset.classes, dataset.class_to_idx


# ==== MAIN EMBEDDING FUNCTION ====
def generate_and_save_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Make sure embeddings folder exists
    Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

    # Load encoder
    print(f"Loading encoder from: {CHECKPOINT_PATH}")
    encoder = load_encoder(CHECKPOINT_PATH, device)

    transform = get_base_transform()

    all_splits = ["train", "val", "test"]
    class_to_idx_global = None

    for split in all_splits:
        print(f"\n=== Processing split: {split} ===")
        dataloader, classes, class_to_idx = get_dataloader(
            split_name=split,
            transform=transform,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        # Save class mapping from the first split (should be same for all)
        if class_to_idx_global is None:
            class_to_idx_global = class_to_idx
            mapping_path = os.path.join(EMBEDDINGS_DIR, "class_to_idx.json")
            with open(mapping_path, "w") as f:
                json.dump(class_to_idx_global, f, indent=4)
            print(f"Saved class_to_idx mapping to {mapping_path}")
        else:
            # sanity check: ensure class mapping matches
            if class_to_idx != class_to_idx_global:
                print("WARNING: class_to_idx mismatch between splits!")

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                # Forward pass through encoder
                outputs = encoder(images)

                # If your SimCLR model returns (h, z), pick one:
                # outputs = encoder(images)[0]  # adjust if needed

                # Move to CPU and store
                all_embeddings.append(outputs.cpu())
                all_labels.append(labels.clone())

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed { (batch_idx + 1) * BATCH_SIZE } images...")

        # Concatenate all batches
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        print(f"{split}: embeddings shape = {embeddings_tensor.shape}")
        print(f"{split}: labels shape = {labels_tensor.shape}")

        # Save to .pt file
        save_path = os.path.join(EMBEDDINGS_DIR, f"{split}_embeddings.pt")
        torch.save(
            {
                "embeddings": embeddings_tensor,   # [N, D]
                "labels": labels_tensor,           # [N]
            },
            save_path,
        )
        print(f"Saved {split} embeddings to: {save_path}")

    print("\n✅ All splits processed. Embeddings saved in:", EMBEDDINGS_DIR)


if __name__ == "__main__":
    generate_and_save_embeddings()