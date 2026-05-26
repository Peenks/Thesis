import torch
from torchvision import models
import torch.nn as nn
import os

CHECKPOINT_PATH = "models/checkpoints/last.pth"
SAVE_PATH = "models/simclr_encoder.pth"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_encoder():
    model = models.mobilenet_v2(weights=None)
    encoder = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    return encoder

encoder = get_encoder().to(device)

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
encoder.load_state_dict(ckpt["encoder"])

os.makedirs("models", exist_ok=True)
torch.save(encoder, SAVE_PATH)

print(f"Encoder exported to {SAVE_PATH}")