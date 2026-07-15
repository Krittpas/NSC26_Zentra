#!/usr/bin/env python3
# training/balance_subset.py — Class-balanced training subset for MPS-efficient PPE training
# ================================================================
# WHY: a random `--fraction 0.3` subset mirrors the training set's heavy class
# imbalance — helmet+person ≈ 50% of instances, while the weak classes appear in
# ~1-2% of images (no_glasses 1.1%, boots 1.1%, glasses 2.0%). So a random subset
# starves exactly the classes we need to improve, wasting MPS training-hours.
#
# WHAT: this builds a subset that INCLUDES EVERY image containing a rare/weak
# class, and CAPS the images that contain only the dominant classes. Result: far
# more weak-class signal per training-hour, at a fraction of the image count.
#
# OUTPUT: a YOLO `data.yaml` whose `train:` points to a generated .txt list of
# absolute image paths (ultralytics reads either a dir or a path-list file). The
# 11-class `names`/order are copied VERBATIM from the source data.yaml — taxonomy
# index stability is sacred (must match the deployed ppe_finetuned.pt).
#
# USAGE:
#   python -m training.balance_subset \
#       --src data/train_dataset/roboflow_dl/data.yaml \
#       --out data/train_dataset/balanced \
#       --cap 2500            # max common-only images per dominant class
#
# Then train on the emitted yaml:
#   python -m training.trainer --task ppe --data data/train_dataset/balanced/data.yaml --epochs 40
# ================================================================
from __future__ import annotations
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# Dominant classes to CAP (everything else is a rare/weak class → include ALL of
# its images). By names so it stays correct regardless of index order.
DOMINANT_DEFAULT = {"helmet", "person"}


def _labels_dir_for(images_dir: Path) -> Path:
    # roboflow layout: .../train/images  ->  .../train/labels
    return images_dir.parent / "labels"


def _iter_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in exts:
            yield p


def build(src_yaml: Path, out_dir: Path, cap: int, dominant: set[str], seed: int = 0) -> Path:
    random.seed(seed)
    src = yaml.safe_load(src_yaml.read_text())
    names: list[str] = src["names"]
    if isinstance(names, dict):  # ultralytics sometimes stores {0:name,...}
        names = [names[i] for i in range(len(names))]
    base = src_yaml.parent
    train_rel = src.get("train", "train/images")
    images_dir = (base / train_rel).resolve()
    labels_dir = _labels_dir_for(images_dir)
    if not images_dir.is_dir():
        raise SystemExit(f"train images dir not found: {images_dir}")

    dom_ids = {i for i, n in enumerate(names) if n in dominant}
    rare_ids = set(range(len(names))) - dom_ids
    print(f"names          : {names}")
    print(f"dominant (cap) : {sorted(dominant)}  ids={sorted(dom_ids)}")
    print(f"rare (all in)  : {[names[i] for i in sorted(rare_ids)]}")

    # Pass 1: classify every image by the class-ids it contains
    img_classes: dict[Path, set[int]] = {}
    for img in _iter_images(images_dir):
        lbl = labels_dir / f"{img.stem}.txt"
        cids: set[int] = set()
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if line:
                    cids.add(int(line.split()[0]))
        img_classes[img] = cids

    selected: set[Path] = set()
    # Pass 2: include EVERY image containing any rare class
    for img, cids in img_classes.items():
        if cids & rare_ids:
            selected.add(img)

    # Pass 3: cap the dominant-only images (contain only dominant classes, or
    # background). Add up to `cap` fresh images per dominant class for context.
    per_dom_added: Counter = Counter()
    dom_only = [img for img, cids in img_classes.items()
                if img not in selected and (not cids or cids <= dom_ids)]
    random.shuffle(dom_only)
    for img in dom_only:
        cids = img_classes[img] or {-1}
        # add if any dominant class this image has is still under its cap
        if any(per_dom_added[c] < cap for c in cids):
            selected.add(img)
            for c in cids:
                per_dom_added[c] += 1

    # Report resulting per-class coverage (number of SELECTED images that
    # contain each class — this is what drives how much the class is seen).
    imgs_with = Counter()
    for img in selected:
        for c in img_classes[img]:
            imgs_with[c] += 1
    print(f"\nselected images: {len(selected)} / {len(img_classes)} "
          f"({100*len(selected)/max(1,len(img_classes)):.1f}%)")
    print(f"{'class':<12}{'imgs_with_class':>16}")
    for i, n in enumerate(names):
        print(f"{n:<12}{imgs_with[i]:>16}")

    # Write outputs: path-list txt + data.yaml (names/order copied verbatim)
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "balanced_train.txt"
    list_path.write_text("\n".join(str(p) for p in sorted(selected)) + "\n")

    val_rel = src.get("val", "valid/images")
    val_dir = (base / val_rel).resolve()
    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": str(list_path.resolve()),
        "val": str(val_dir),
        "nc": len(names),
        "names": names,
    }, sort_keys=False, allow_unicode=True))
    print(f"\n✅ wrote {list_path}")
    print(f"✅ wrote {out_yaml}")
    return out_yaml


def main():
    ap = argparse.ArgumentParser(description="Build a class-balanced PPE training subset")
    ap.add_argument("--src", required=True, help="source data.yaml (e.g. roboflow_dl/data.yaml)")
    ap.add_argument("--out", required=True, help="output dir for balanced data.yaml + list")
    ap.add_argument("--cap", type=int, default=2500,
                    help="max dominant-only images to add per dominant class")
    ap.add_argument("--dominant", nargs="*", default=sorted(DOMINANT_DEFAULT),
                    help="class names to cap (default: helmet person)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(Path(args.src), Path(args.out), args.cap, set(args.dominant), args.seed)


if __name__ == "__main__":
    main()
