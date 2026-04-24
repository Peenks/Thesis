import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# =====================
# CONFIG
# =====================
DATA_SPLIT_DIR = "data/split"
ENCODER_PATH = "models/simclr_encoder.pth"
CLASSIFIER_PATH = "models/mlp_classifier.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32


# =====================
# DEVICE
# =====================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =====================
# TRANSFORM
# =====================
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


# =====================
# ENCODER BUILDER (IMPORTANT FIX)
# =====================
def build_encoder():
    from torchvision import models

    model = models.mobilenet_v2(weights=None)

    encoder = nn.Sequential(
        model.features,
        nn.Conv2d(1280, 1280, 1),
        nn.BatchNorm2d(1280),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )

    return encoder


# =====================
# MODEL
# =====================
class Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder

        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


# =====================
# MAIN
# =====================
def main():
    device = get_device()
    print("Device:", device)

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_SPLIT_DIR, "test"),
        transform=get_transform()
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(test_dataset.classes)

    encoder = build_encoder()
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device), strict=True)
    encoder.to(device)

    model = Classifier(encoder, num_classes).to(device)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    model.eval()

    preds_all, labels_all = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            preds_all.append(preds.cpu())
            labels_all.append(yb.cpu())

    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(labels_all).numpy()

    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()