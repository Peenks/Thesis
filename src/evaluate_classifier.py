import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# ===== CONFIG =====
DATA_SPLIT_DIR = "data/split"
CHECKPOINT_PATH = "models/simclr_encoder.pth"
MODEL_PATH = "models/finetuned_model.pth"
CLASS_MAP_PATH = "data/embeddings/class_to_idx.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_CLASSES = 10


# ===== DEVICE =====
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ===== TRANSFORMS =====
def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


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
        logits = self.classifier(features)
        return logits


def main():
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_PATH}")

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_SPLIT_DIR, "test"),
        transform=get_test_transform()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    if os.path.exists(CLASS_MAP_PATH):
        with open(CLASS_MAP_PATH, "r") as f:
            class_to_idx = json.load(f)
    else:
        class_to_idx = test_dataset.class_to_idx

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("Loading encoder...")
    encoder = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    encoder = encoder.to(device)

    model = FineTunedTrashClassifier(encoder, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Test Macro-F1: {macro_f1:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()