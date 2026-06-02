import os
import time
import torch
import torchvision.models as models
import torch.nn as nn
from thop import profile

from train_finetune import (
    FineTunedTrashClassifier,
    freeze_encoder,
    unfreeze_last_percent
)

MODEL_PATH = "models/finetuned_model.pth"
NUM_CLASSES = 10
IMAGE_SIZE = 224

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Recreate MobileNetV2 encoder
mobilenet = models.mobilenet_v2(weights=None)

encoder = nn.Sequential(
    mobilenet.features,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

# Apply the same fine-tuning strategy used during training
freeze_encoder(encoder)
unfreeze_last_percent(encoder, percent=0.3)

# Recreate full fine-tuned model
model = FineTunedTrashClassifier(encoder, NUM_CLASSES)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# Parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

frozen_params = total_params - trainable_params
trainable_percent = (trainable_params / total_params) * 100

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Frozen Parameters: {frozen_params:,}")
print(f"Trainable Percentage: {trainable_percent:.2f}%")

# Model size
size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Model Size: {size_mb:.2f} MB")

# Inference time
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)

start = time.time()

with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)

end = time.time()

avg_time = (end - start) / 100
print(f"Average Inference Time: {avg_time * 1000:.2f} ms/image")

# FLOPs on CPU
cpu_model = model.to("cpu")
cpu_model.eval()
cpu_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

flops, params = profile(cpu_model, inputs=(cpu_input,), verbose=False)

print(f"FLOPs: {flops:,.0f}")
print(f"GFLOPs: {flops / 1e9:.3f}")