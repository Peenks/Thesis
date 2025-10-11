#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, shutil, random, argparse
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageOps
try:
    from PIL import UnidentifiedImageError
except ImportError:
    class UnidentifiedImageError(Exception):
        pass
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def is_image(p):
    return Path(p).suffix.lower() in IMG_EXTS

def safe_open(path):
    img = Image.open(path)
    img.load()              # catch truncated/corrupt files
    img = ImageOps.exif_transpose(img)  # normalize EXIF orientation
    return img

def find_src_dir(raw_dir, expected_classes):
    candidates = [
        raw_dir / "Garbage classification",
        raw_dir / "garbage-classification",
        raw_dir,
    ]
    for c in candidates:
        if not c.exists():
            continue
        try:
            classes = {d.name for d in c.iterdir() if d.is_dir()}
        except PermissionError:
            continue
        if expected_classes.issubset(classes):
            return c
    raise SystemExit(
        "Could not locate class folders. Use --src to point at the folder that directly "
        "contains the class subfolders."
    )

def collect_samples(src_dir):
    classes = [d for d in src_dir.iterdir() if d.is_dir()]
    if not classes:
        raise SystemExit("No class folders found inside: {}".format(src_dir))
    samples = []
    for cdir in classes:
        for p in cdir.rglob("*"):
            if p.is_file() and is_image(p):
                samples.append((p, cdir.name))
    return samples, sorted([c.name for c in classes])

def validate_and_prune(samples):
    ok, bad = [], []
    for p, y in tqdm(samples, desc="Validating images"):
        try:
            _ = safe_open(p)
            ok.append((p, y))
        except (UnidentifiedImageError, OSError):
            bad.append((p, y))
    return ok, bad

def stratified_split(samples, split, seed):
    by_class = defaultdict(list)
    for p, y in samples:
        by_class[y].append(p)
    rng = random.Random(seed)
    for y in by_class:
        rng.shuffle(by_class[y])

    out = {"train": [], "val": [], "test": []}
    for y, paths in by_class.items():
        n = len(paths)
        n_train = int(n * split["train"])
        n_val = int(n * split["val"])
        splits = {
            "train": paths[:n_train],
            "val": paths[n_train:n_train+n_val],
            "test": paths[n_train+n_val:],
        }
        for k, plist in splits.items():
            out[k].extend([(str(p), y) for p in plist])
    return out

def copy_split(split_map, out_dir):
    for part, items in split_map.items():
        for src, y in tqdm(items, desc="Copying {}".format(part), leave=False):
            dst = out_dir / part / y / Path(src).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                img = safe_open(Path(src))
                img.save(dst)
            except Exception:
                shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="Folder where Kaggle unzipped")
    ap.add_argument("--src", default="", help="Exact path to folder containing class subfolders")
    ap.add_argument("--out", default="data/processed", help="Output folder for splits")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    split = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(split.values()) - 1.0) > 1e-6:
        raise SystemExit("Splits must sum to 1.0")

    expected = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
    src_dir = Path(args.src) if args.src else find_src_dir(raw_dir, expected)

    print("Source:", src_dir)
    print("Output:", out_dir)

    samples, class_names = collect_samples(src_dir)
    print("Found {} images across {} classes.".format(len(samples), len(class_names)))

    good, bad = validate_and_prune(samples)
    if bad:
        print("Corrupt/unreadable: {} (excluded)".format(len(bad)))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "corrupt_list.txt", "w") as f:
            for p, y in bad:
                f.write("{}\t{}\n".format(p, y))

    counts = Counter(y for _, y in good)
    print("Per-class counts (clean):")
    for c in sorted(counts):
        print("  {}: {}".format(c, counts[c]))

    split_map = stratified_split(good, split, args.seed)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_split(split_map, out_dir)

    manifest = {
        "source_dir": str(src_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "classes": class_names,
        "split": split,
        "seed": args.seed,
        "totals": {k: len(v) for k, v in split_map.items()},
        "per_class_clean_counts": dict(counts),
    }
    with open(out_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "splits.json", "w") as f:
        json.dump(split_map, f, indent=2)

    print("Done. Folders at data/processed/{train,val,test} and splits.json written.")

if __name__ == "__main__":
    main()