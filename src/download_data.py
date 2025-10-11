import os
import zipfile

# Dataset: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
dataset = "asdasdasasdas/garbage-classification"
raw_dir = "data/raw"

# Create folder if not exists
os.makedirs(raw_dir, exist_ok=True)

# Download dataset (with --unzip to avoid manual extraction)
cmd = f"kaggle datasets download -d {dataset} -p {raw_dir} --unzip"
os.system(cmd)

print("✅ Dataset downloaded and extracted to", raw_dir)