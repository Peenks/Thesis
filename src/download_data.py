import os
import zipfile
import shutil
from pathlib import Path

#Kaggle 
dataset = "sumn2u/garbage-classification-v2"
raw_dir = Path("data/raw")

raw_dir.mkdir(parents=True, exist_ok=True)

#Download dataset
cmd = f"kaggle datasets download -d {dataset} -p {raw_dir} --unzip"
os.system(cmd)

#Flattens 2 versions of kaggle dataset v1 and v2 incase u made a mistake (Like me :p)
possible_folders = [
    raw_dir / "Garbage classification",
    raw_dir / "Garbage classification v2",
]

for folder in possible_folders:
    if folder.exists():
        print(f"Found {folder.name}")
        for item in folder.iterdir():
            dest = raw_dir / item.name
            if dest.exists():
                print(f"Skipping {item.name} (already exists)")
                continue
            shutil.move(str(item), raw_dir)
        shutil.rmtree(folder, ignore_errors=True)

# remove leftover zips
for z in raw_dir.glob("*.zip"):
    z.unlink()

# organize folders
expected_classes = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "paper", "plastic", "shoes", "trash"
]

# Move loose images into correct class folders
for img in raw_dir.glob("*.jpg"):
    label = img.stem.split("_")[0].lower()
    if label in expected_classes:
        dest = raw_dir / label
        dest.mkdir(exist_ok=True)
        shutil.move(str(img), dest / img.name)

found_classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
print("\nDataset organized under data/raw/")
print("Found categories:", found_classes)