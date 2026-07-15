"""
training/dedupe.py — thin out near-duplicate collected frames

The collector used to save ~20-30 near-identical frames per incident. For
good training you want DIVERSE frames. This walks a category in time order
and keeps an image only if it looks different enough from the last kept one;
near-duplicates (and their .txt labels) are MOVED to <category>_dupes/
(reversible — nothing is deleted).

Usage:
  python -m training.dedupe --category ppe_violations            # dry-run (report only)
  python -m training.dedupe --category ppe_violations --apply    # actually move dupes
  python -m training.dedupe --category ppe_violations --diff 10 --apply
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

import config as cfg


def _thumb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (32, 32)).astype(np.float32)


def dedupe(category: str, diff: float = 8.0, apply: bool = False) -> dict:
    src = Path(cfg.COLLECTED_DIR) / category
    imgs = sorted(src.glob("*.jpg"))
    if not imgs:
        print(f"[Dedupe] no images in {src}")
        return {"total": 0, "kept": 0, "moved": 0}

    dup_dir = Path(cfg.COLLECTED_DIR) / f"{category}_dupes"
    kept = moved = 0
    last = None

    for ip in imgs:
        th = _thumb(ip)
        if th is None:
            continue
        if last is not None and float(np.mean(np.abs(th - last))) < diff:
            # near-duplicate of the last kept frame
            moved += 1
            if apply:
                dup_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ip), str(dup_dir / ip.name))
                lbl = ip.with_suffix(".txt")
                if lbl.exists():
                    shutil.move(str(lbl), str(dup_dir / lbl.name))
        else:
            kept += 1
            last = th

    verb = "moved" if apply else "would move"
    print(f"[Dedupe] {category}: {len(imgs)} → keep {kept}, {verb} {moved} "
          f"(diff<{diff}){'' if apply else '  [DRY-RUN — add --apply]'}")
    if apply and moved:
        print(f"[Dedupe] duplicates moved to: {dup_dir}")
    return {"total": len(imgs), "kept": kept, "moved": moved}


def main():
    ap = argparse.ArgumentParser(description="Remove near-duplicate collected frames")
    ap.add_argument("--category", default="ppe_violations",
                    choices=["ppe_violations", "zone_intrusions", "fall_events", "normal"])
    ap.add_argument("--diff", type=float, default=8.0, help="similarity threshold (higher = more aggressive)")
    ap.add_argument("--apply", action="store_true", help="actually move dupes (default: dry-run)")
    args = ap.parse_args()
    dedupe(args.category, diff=args.diff, apply=args.apply)


if __name__ == "__main__":
    main()
