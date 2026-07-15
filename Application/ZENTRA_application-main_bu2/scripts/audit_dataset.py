#!/usr/bin/env python3
"""Audit a YOLO dataset before trusting any number that comes out of it.

Answers three questions the training loop cannot:

  1. LEAKAGE  — Roboflow emits N augmented copies per source image
     (`<base>.rf.<hash>.jpg`) and splits on the *augmented* files. If a base
     image lands in both train and valid, the val score is inflated: the model
     has already seen that scene.

  2. CO-OCCURRENCE — a PPE class is only learnable *on a person* if its boxes
     appear in frames that also contain a person box. `helmet` boxes cropped
     from a product photo teach the model what a helmet looks like on a table.

  3. MEASURABILITY — a class with zero instances in val cannot be scored. Any
     mAP that includes it is an average over a missing number.

Usage:
    python3 scripts/audit_dataset.py --root backend/data/train_dataset/roboflow_dl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def base_name(p: Path) -> str:
    """Source image behind a Roboflow augmentation: `foo.rf.<hash>.jpg` -> `foo`."""
    return p.name.split(".rf.")[0]


def read_labels(label_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    out: dict[str, list] = {}
    for f in label_dir.glob("*.txt"):
        rows = []
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append((int(parts[0]), *(float(v) for v in parts[1:5])))
        out[f.stem] = rows
    return out


def audit_split(root: Path, split: str, names: list[str], person_idx: int) -> dict:
    lab = root / split / "labels"
    img = root / split / "images"
    if not lab.is_dir():
        return {}
    labels = read_labels(lab)

    instances = Counter()
    imgs_with_class = defaultdict(set)
    imgs_with_person = set()
    box_px = defaultdict(list)          # class -> [(w,h) normalized]

    for stem, rows in labels.items():
        classes = {r[0] for r in rows}
        if person_idx in classes:
            imgs_with_person.add(stem)
        for c, _cx, _cy, w, h in rows:
            instances[c] += 1
            imgs_with_class[c].add(stem)
            box_px[c].append((w, h))

    per_class = {}
    for ci, cname in enumerate(names):
        imgs = imgs_with_class.get(ci, set())
        with_person = len(imgs & imgs_with_person)
        whs = box_px.get(ci, [])
        med_w = med_h = 0.0
        if whs:
            ws = sorted(w for w, _ in whs); hs = sorted(h for _, h in whs)
            med_w, med_h = ws[len(ws) // 2], hs[len(hs) // 2]
        per_class[cname] = {
            "instances": instances.get(ci, 0),
            "images": len(imgs),
            "images_with_person": with_person,
            "pct_with_person": round(100.0 * with_person / len(imgs), 1) if imgs else 0.0,
            # median box size at imgsz 640, purely to show how big the object is
            "median_box_px_at_640": [round(med_w * 640), round(med_h * 640)],
        }

    return {
        "n_images": len(list(img.glob("*"))) if img.is_dir() else 0,
        "n_labels": len(labels),
        "n_base_images": len({base_name(Path(s + ".jpg")) for s in labels}),
        "images_with_person": len(imgs_with_person),
        "per_class": per_class,
    }


def leakage(root: Path, a: str, b: str) -> dict:
    def bases(split: str) -> set[str]:
        d = root / split / "images"
        return {base_name(p) for p in d.iterdir()} if d.is_dir() else set()

    ba, bb = bases(a), bases(b)
    overlap = ba & bb
    return {
        "split_a": a, "split_b": b,
        "base_images_a": len(ba), "base_images_b": len(bb),
        "overlapping_base_images": len(overlap),
        "pct_of_b_leaked": round(100.0 * len(overlap) / len(bb), 1) if bb else 0.0,
        "examples": sorted(overlap)[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--json", type=Path, help="also write the full report here")
    args = ap.parse_args()

    cfg = yaml.safe_load((args.root / "data.yaml").read_text())
    names = cfg["names"]
    person_idx = names.index("person") if "person" in names else -1

    report = {"root": str(args.root), "names": names, "splits": {}, "leakage": []}
    for split in ("train", "valid", "test"):
        r = audit_split(args.root, split, names, person_idx)
        if r:
            report["splits"][split] = r
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if a in report["splits"] and b in report["splits"]:
            report["leakage"].append(leakage(args.root, a, b))

    # ── print ────────────────────────────────────────────────────────────
    print(f"\ndataset: {args.root}")
    for split, r in report["splits"].items():
        print(f"\n  {split}: {r['n_images']} images "
              f"({r['n_base_images']} unique source images) · "
              f"{r['images_with_person']} contain a person")

    print("\n── co-occurrence with `person` (train) " + "─" * 32)
    print(f"  {'class':<12} {'instances':>10} {'images':>8} {'w/ person':>10} {'median box @640':>18}")
    tr = report["splits"].get("train", {}).get("per_class", {})
    for cname, c in sorted(tr.items(), key=lambda kv: -kv[1]["instances"]):
        w, h = c["median_box_px_at_640"]
        flag = "" if c["pct_with_person"] >= 40 or cname == "person" else "  ⚠"
        print(f"  {cname:<12} {c['instances']:>10} {c['images']:>8} "
              f"{c['pct_with_person']:>9.1f}% {f'{w}x{h}':>18}{flag}")

    print("\n── measurability (valid) " + "─" * 45)
    va = report["splits"].get("valid", {}).get("per_class", {})
    for cname, c in va.items():
        if c["instances"] == 0:
            print(f"  ✗ {cname}: 0 instances in valid → cannot be scored")
    zero = [c for c, v in va.items() if v["instances"] == 0]
    if not zero:
        print("  ✓ every class has val instances")

    print("\n── augmentation leakage " + "─" * 46)
    for lk in report["leakage"]:
        mark = "✗" if lk["overlapping_base_images"] else "✓"
        print(f"  {mark} {lk['split_a']}↔{lk['split_b']}: "
              f"{lk['overlapping_base_images']} shared source images "
              f"({lk['pct_of_b_leaked']}% of {lk['split_b']})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nfull report → {args.json}")


if __name__ == "__main__":
    main()
