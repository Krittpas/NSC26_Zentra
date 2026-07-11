#!/usr/bin/env python3
"""Stage 1 — build a leak-free, person-only YOLO dataset.

Two problems with training person detection on `roboflow_dl` directly:

  * LEAKAGE. Roboflow splits on augmented files, so 24.5% of valid and 21.6% of
    test share a source image with train. Any score measured there is inflated.
    We re-split by SOURCE image (`<base>.rf.<hash>.jpg` -> `<base>`), so every
    augmentation of one photo lands in exactly one split.

  * The other 10 classes are noise for this stage. Person detection is the
    foundation every other module stands on; train it alone, score it alone.

Images are SYMLINKED (the source dataset is ~8 GB and the disk has ~16 GB free);
only the rewritten single-class label files are real.

Only images containing at least one `person` box are kept: an image with no
person box is not evidence that no person is there — the other classes were
annotated from product photos where people were simply never labelled. Feeding
those in as "background" would actively teach the model to miss people.

Usage:
    python3 scripts/build_person_dataset.py \
        --src backend/data/train_dataset/roboflow_dl \
        --dst backend/data/train_dataset/person_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def base_name(name: str) -> str:
    return name.split(".rf.")[0]


def split_of(base: str, val_pct: float, test_pct: float) -> str:
    """Deterministic, group-stable assignment: a source image always lands in the
    same split regardless of how many augmentations it has, or what order we walk
    the directory in. Hash of the base name, not a shuffled index."""
    h = int(hashlib.sha256(base.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--val-pct", type=float, default=0.15)
    ap.add_argument("--test-pct", type=float, default=0.15)
    ap.add_argument("--one-per-source", action="store_true",
                    help="Keep a single augmentation per source photo. Roboflow baked "
                         "~3.2 copies of each image into the dataset, but ultralytics "
                         "augments on the fly during training — so the extra copies cost "
                         "3.2x the epoch time and add almost no information. The unique "
                         "photo count is the real size of this dataset.")
    args = ap.parse_args()

    cfg = yaml.safe_load((args.src / "data.yaml").read_text())
    person_idx = cfg["names"].index("person")
    print(f"source person class index: {person_idx}")

    if args.dst.exists():
        shutil.rmtree(args.dst)
    for s in ("train", "val", "test"):
        (args.dst / s / "images").mkdir(parents=True)
        (args.dst / s / "labels").mkdir(parents=True)

    # Gather every (image, label) pair across ALL source splits, then re-split.
    pairs: list[tuple[Path, Path]] = []
    for s in ("train", "valid", "test"):
        idir, ldir = args.src / s / "images", args.src / s / "labels"
        if not idir.is_dir():
            continue
        for img in idir.iterdir():
            if img.suffix.lower() in IMG_EXT:
                lab = ldir / (img.stem + ".txt")
                if lab.exists():
                    pairs.append((img, lab))
    print(f"found {len(pairs)} labelled images across source splits")

    if args.one_per_source:
        # Deterministic pick: lexicographically first augmentation of each photo.
        chosen: dict[str, tuple[Path, Path]] = {}
        for img, lab in pairs:
            b = base_name(img.name)
            if b not in chosen or img.name < chosen[b][0].name:
                chosen[b] = (img, lab)
        print(f"one-per-source: {len(pairs)} augmented → {len(chosen)} unique photos")
        pairs = list(chosen.values())

    stats = defaultdict(lambda: {"images": 0, "persons": 0, "bases": set()})
    skipped_no_person = 0

    for img, lab in pairs:
        rows = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()]
        persons = [r for r in rows if len(r) >= 5 and int(r[0]) == person_idx]
        if not persons:
            skipped_no_person += 1
            continue

        base = base_name(img.name)
        s = split_of(base, args.val_pct, args.test_pct)

        # single-class dataset: person becomes class 0
        out_lab = args.dst / s / "labels" / (img.stem + ".txt")
        out_lab.write_text("\n".join(" ".join(["0", *r[1:5]]) for r in persons) + "\n")

        out_img = args.dst / s / "images" / img.name
        if not out_img.exists():
            out_img.symlink_to(img.resolve())

        st = stats[s]
        st["images"] += 1
        st["persons"] += len(persons)
        st["bases"].add(base)

    (args.dst / "data.yaml").write_text(yaml.safe_dump({
        "path": str(args.dst.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["person"],
    }, sort_keys=False))

    print(f"\nskipped {skipped_no_person} images with no person box")
    print(f"\n{'split':<8}{'images':>10}{'persons':>10}{'src images':>13}{'persons/img':>13}")
    for s in ("train", "val", "test"):
        st = stats[s]
        ppi = st["persons"] / st["images"] if st["images"] else 0
        print(f"{s:<8}{st['images']:>10}{st['persons']:>10}{len(st['bases']):>13}{ppi:>13.2f}")

    # prove the re-split is leak-free
    bases = {s: stats[s]["bases"] for s in ("train", "val", "test")}
    print("\nleakage check (shared SOURCE images):")
    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(bases[a] & bases[b])
        ok &= n == 0
        print(f"  {'✓' if n == 0 else '✗'} {a}↔{b}: {n}")
    print(("\n✓ leak-free" if ok else "\n✗ LEAKAGE REMAINS"))

    (args.dst / "build_report.json").write_text(json.dumps(
        {s: {"images": stats[s]["images"], "persons": stats[s]["persons"],
             "source_images": len(stats[s]["bases"])} for s in ("train", "val", "test")}, indent=2))


if __name__ == "__main__":
    main()
