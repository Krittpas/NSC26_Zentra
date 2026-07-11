#!/usr/bin/env python3
# training/autolabel.py — Pre-label real frames with a model, in the DEPLOYED
# 11-class taxonomy, so a human only has to CORRECT (not draw from scratch).
# ================================================================
# WHY: closing the live/domain gap needs real-camera frames labelled in our
# taxonomy. Hand-drawing every box is too much work. This runs a model over a
# folder of images and writes YOLO .txt pre-labels (+ a data.yaml) that you then
# open in a labeling tool (labelImg / Roboflow / CVAT) and just FIX — mainly the
# weak classes (no_glasses, boots, real eyeglasses) the model gets wrong.
#
# The model's own class NAMES are remapped to the deployed taxonomy via
# cfg.ppe_taxo_index, so this works with BOTH our ppe_finetuned.pt (names already
# = taxonomy) AND an external "teacher" PPE model (different names → remapped).
# Unmapped detections are dropped (reported), never guessed.
#
# USAGE:
#   # pre-label with our current best model (good for the easy/bulk classes):
#   python -m training.autolabel --model models/ppe_finetuned.pt \
#       --images data/collected/ppe_violations --out data/realframes --conf 0.25
#
#   # or with an external open-source teacher (helps classes ours misses):
#   python -m training.autolabel --model <teacher.pt> --images <dir> --out <dir>
#
# THEN: open <out> in a labeling tool, CORRECT the boxes (focus on weak classes),
# split into realval/ (frozen, never trained) + train/, and feed train/ to
# merge_external.py / trainer.py.
# ================================================================
from __future__ import annotations
import argparse
import shutil
from collections import Counter
from pathlib import Path

import config as cfg

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Pre-label frames in the deployed PPE taxonomy")
    ap.add_argument("--model", required=True, help="model .pt (ours or a teacher) or roboflow id")
    ap.add_argument("--images", required=True, help="folder of real frames to pre-label")
    ap.add_argument("--out", required=True, help="output dir (images/ + labels/ + data.yaml)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence floor")
    ap.add_argument("--imgsz", type=int, default=960, help="inference imgsz (match live PPE_IMGSZ)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics")

    src = Path(args.images)
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise SystemExit(f"no images found under {src}")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    names = model.names  # model's own class-id -> name

    mapped, dropped = Counter(), Counter()
    labelled = 0
    print(f"pre-labelling {len(imgs)} images with {args.model} …")
    for img in imgs:
        r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz,
                          device=args.device, verbose=False)[0]
        lines = []
        for b in r.boxes:
            raw = names[int(b.cls)]
            idx = cfg.ppe_taxo_index(raw)
            if idx is None:
                dropped[raw] += 1
                continue
            xc, yc, w, h = b.xywhn[0].tolist()  # normalized
            lines.append(f"{idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            mapped[cfg.PPE_TAXONOMY[idx]] += 1
        # copy image + write pre-label (empty file if no detections — still reviewable)
        dst_img = out / "images" / img.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        (out / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        labelled += 1

    # data.yaml for the labeling tool / later training
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\n"
        f"nc: {len(cfg.PPE_TAXONOMY)}\nnames: {list(cfg.PPE_TAXONOMY)}\n"
    )
    print(f"\n✅ pre-labelled {labelled} images → {out}")
    print(f"   mapped instances: {dict(sorted(mapped.items()))}")
    if dropped:
        print(f"   ⚠️ dropped (unmapped) model classes: {dict(dropped)}")
    print("\nNEXT: open in a labeling tool, CORRECT boxes (focus weak classes),")
    print("      then split into realval/ (frozen) + train/.")


if __name__ == "__main__":
    main()
