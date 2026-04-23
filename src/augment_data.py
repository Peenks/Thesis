import os
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# ========== PATH SETUP ==========
ROOT_DIR = Path(__file__).resolve().parents[1]  # go up one level (src → root)
RAW_DIR = ROOT_DIR / "data" / "raw"
AUG_DIR = ROOT_DIR / "data" / "augmented"

# Create output directories if not existing
AUG_DIR.mkdir(parents=True, exist_ok=True)

# ========== CONFIG ==========
IMAGE_SIZE = (224, 224)
AUG_PER_IMAGE = 3  # how many new images per original

# ========== AUGMENTATION FUNCTIONS ==========
def random_crop(img, scale=(0.6, 1.0)):
    """Randomly crop the image while keeping the aspect ratio."""
    w, h = img.size
    crop_scale = random.uniform(*scale)
    new_w, new_h = int(w * crop_scale), int(h * crop_scale)
    left = random.randint(0, max(0, w - new_w))
    top = random.randint(0, max(0, h - new_h))
    return img.crop((left, top, left + new_w, top + new_h)).resize(IMAGE_SIZE)

def random_color_jitter(img):
    """SimCLR-style color jitter."""
    enhancers = [
        (ImageEnhance.Brightness, 0.6, 1.4),
        (ImageEnhance.Contrast, 0.6, 1.4),
        (ImageEnhance.Color, 0.6, 1.4),
    ]
    for enhancer, low, high in enhancers:
        factor = random.uniform(low, high)
        img = enhancer(img).enhance(factor)
    return img

def random_blur(img):
    """Apply Gaussian blur with 50% chance."""
    if random.random() < 0.5:
        radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius))
    return img

def random_grayscale(img):
    """Randomly convert to grayscale (30% chance)."""
    if random.random() < 0.3:
        img = ImageOps.grayscale(img).convert("RGB")
    return img

def random_flip(img):
    """Random horizontal flip (50% chance)."""
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    return img

def simclr_augment(img):
    """Apply all SimCLR-style augmentations."""
    img = random_crop(img)
    img = random_flip(img)
    img = random_color_jitter(img)
    img = random_blur(img)
    img = random_grayscale(img)
    return img

# ========== MAIN FUNCTION ==========
def augment_dataset(input_dir=RAW_DIR, output_dir=AUG_DIR, n_aug=AUG_PER_IMAGE):
    """Augment all images in the dataset folder."""
    os.makedirs(output_dir, exist_ok=True)

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for cls in classes:
        input_cls = os.path.join(input_dir, cls)
        output_cls = os.path.join(output_dir, cls)
        os.makedirs(output_cls, exist_ok=True)

        print(f"Augmenting class: {cls}")

        for img_name in os.listdir(input_cls):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(input_cls, img_name)
            try:
                img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
            except Exception as e:
                print(f"Skipping {img_name}: {e}")
                continue

            # Save the resized original image
            img.save(os.path.join(output_cls, img_name))

            # Generate augmentations
            base_name, ext = os.path.splitext(img_name)
            for i in range(n_aug):
                aug_img = simclr_augment(img)
                aug_name = f"{base_name}_aug{i+1}{ext}"
                aug_img.save(os.path.join(output_cls, aug_name))

        print(f"Finished augmenting {cls}")

    print("\nAugmentation complete! All results saved to:", output_dir)

# ========== RUN ==========
if __name__ == "__main__":
    augment_dataset()