import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report


# ===== CONFIG =====
DATA_SPLIT_DIR = "data/split"
CHECKPOINT_PATH = "models/simclr_encoder.pth"
MODEL_SAVE_PATH = "models/finetuned_model.pth"
CLASS_MAP_PATH = "data/embeddings/class_to_idx.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 35
NUM_CLASSES = 10
PATIENCE = 5

ENCODER_LR = 3e-5
HEAD_LR = 1e-3


# ===== DEVICE =====
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ===== LESS AGGRESSIVE TRANSFORMS =====
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


# ===== DATA =====
def get_dataloaders():
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_SPLIT_DIR, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_SPLIT_DIR, "val"),
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_dataset, val_dataset, train_loader, val_loader


# ===== CLASS WEIGHTS =====
def compute_class_weights(dataset, num_classes):
    targets = torch.tensor(dataset.targets)
    counts = torch.bincount(targets, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts)
    return weights


# ===== MODEL =====
class FineTunedTrashClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = torch.nn.functional.normalize(features, dim=1)
        logits = self.classifier(features)
        return logits


# ===== FREEZE / UNFREEZE =====
def freeze_encoder(encoder):
    for param in encoder.parameters():
        param.requires_grad = False


def unfreeze_last_layers(encoder, num_layers=150):
    params = list(encoder.parameters())
    for param in params[-num_layers:]:
        param.requires_grad = True


def print_trainable_parameters(model):
    total = 0
    trainable = 0

    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count

    print(f"Trainable parameters: {trainable:,} / {total:,}")


# ===== EVALUATION =====
def evaluate(model, loader, device, target_names):
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = correct / total
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    return acc, macro_f1, report


# ===== MAIN =====
def train():
    device = get_device()
    print(f"Using device: {device}")

    Path("models").mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = get_dataloaders()

    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    os.makedirs("data/embeddings", exist_ok=True)
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(class_to_idx, f, indent=4)

    print("Loading encoder...")
    encoder = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    encoder = encoder.to(device)

    freeze_encoder(encoder)
    unfreeze_last_layers(encoder, num_layers=150)

    model = FineTunedTrashClassifier(encoder, NUM_CLASSES).to(device)

    print_trainable_parameters(model)

    class_weights = compute_class_weights(train_dataset, NUM_CLASSES).to(device)
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": ENCODER_LR},
        {"params": head_params, "lr": HEAD_LR}
    ])

    best_macro_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        val_acc, val_macro_f1, val_report = evaluate(
            model,
            val_loader,
            device,
            target_names
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            print("\n✅ Best model updated and saved.")
            print("=== Validation Classification Report ===")
            print(val_report)
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= PATIENCE:
            print("\n⏹ Early stopping triggered.")
            break

    print(f"\n✅ Fine-tuning finished. Best model saved to {MODEL_SAVE_PATH}")
    print(f"Best Validation Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    train()