import os
import shutil
import random

random.seed(42)

def split_class_folder(class_path, output_base, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    class_name = os.path.basename(class_path)

    for split_name, file_list in splits.items():
        split_folder = os.path.join(output_base, split_name, class_name)
        os.makedirs(split_folder, exist_ok=True)

        for filename in file_list:
            src = os.path.join(class_path, filename)
            dst = os.path.join(split_folder, filename)
            shutil.copy(src, dst)

    print(f"{class_name}: {total} images → "
          f"{len(splits['train'])} train, "
          f"{len(splits['val'])} val, "
          f"{len(splits['test'])} test")


def split_dataset(raw_folder, output_folder):

    for class_name in os.listdir(raw_folder):
        class_path = os.path.join(raw_folder, class_name)

        if os.path.isdir(class_path):
            split_class_folder(class_path, output_folder)


if __name__ == "__main__":

    RAW_PATH = "data/raw/original"   # 🔥 FIXED HERE
    OUTPUT_PATH = "data/split"

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    split_dataset(RAW_PATH, OUTPUT_PATH)

    print("Dataset split complete!")