import os
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==== CONFIG ====
DATA_SPLIT_DIR = "data/split"
EMBEDDINGS_DIR = "data/embeddings"

BATCH_SIZE = 64
NUM_WORKERS = 0   # ✅ FIXED FOR MAC
IMAGE_SIZE = 224

CHECKPOINT_PATH = "models/simclr_encoder.pth"


# ==== TRANSFORMS ====
def get_base_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ==== LOAD ENCODER ====
def load_encoder(checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model.to(device)


# ==== DATA LOADER ====
def get_dataloader(split_name, transform):
    split_dir = os.path.join(DATA_SPLIT_DIR, split_name)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing folder: {split_dir}")

    dataset = datasets.ImageFolder(root=split_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return dataloader, dataset.class_to_idx


# ==== MAIN ====
def generate_and_save_embeddings():

    # ✅ FIXED DEVICE FOR MAC
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loading encoder from: {CHECKPOINT_PATH}")
    encoder = load_encoder(CHECKPOINT_PATH, device)

    transform = get_base_transform()

    class_to_idx_global = None

    for split in ["train", "val", "test"]:
        print(f"\n=== Processing: {split} ===")

        dataloader, class_to_idx = get_dataloader(split, transform)

        # Save mapping once
        if class_to_idx_global is None:
            class_to_idx_global = class_to_idx
            with open(os.path.join(EMBEDDINGS_DIR, "class_to_idx.json"), "w") as f:
                json.dump(class_to_idx, f, indent=4)

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):

                images = images.to(device)
                outputs = encoder(images)

                all_embeddings.append(outputs.cpu())
                all_labels.append(labels.clone())

                if (batch_idx + 1) % 20 == 0:
                    print(f"{split}: processed {(batch_idx+1)*BATCH_SIZE} images")

        embeddings = torch.cat(all_embeddings)
        labels = torch.cat(all_labels)

        print(f"{split} embeddings shape: {embeddings.shape}")

        save_path = os.path.join(EMBEDDINGS_DIR, f"{split}_embeddings.pt")

        torch.save({
            "embeddings": embeddings,
            "labels": labels
        }, save_path)

        print(f"Saved → {save_path}")

    print("\n✅ ALL EMBEDDINGS DONE")


if __name__ == "__main__":
    generate_and_save_embeddings()