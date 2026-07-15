#!/usr/bin/env python3
"""Stage 1 — fine-tune person detection, and score it honestly.

Baseline to beat (COCO yolo11s, no fine-tune, on this leak-free val):
    mAP50 0.598 · mAP50-95 0.305 · P 0.727 · R 0.623

Recall 0.62 is the number that matters: nearly 4 in 10 annotated people are
missed, and PPE, zone and fall all stand on the person box. A miss here is
invisible downstream — the person simply does not exist to the rest of the system.

The model is evaluated at BOTH the training resolution and the deployed
resolution (cfg.PERSON_IMGSZ), because those differ and a score at one says
little about the other.

Usage:
    python3 scripts/train_person.py --epochs 40
    python3 scripts/train_person.py --eval-only backend/models/person_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "backend/data/train_dataset/person_v1/data.yaml"
OUT = REPO / "backend/models"


def evaluate(model, tag: str, imgsz: int, split: str, **kw) -> dict:
    r = model.val(data=str(DATA), split=split, imgsz=imgsz, batch=16,
                  device="mps", verbose=False, plots=False, **kw)
    m = {"tag": tag, "split": split, "imgsz": imgsz,
         "mAP50": round(float(r.box.map50), 4),
         "mAP50_95": round(float(r.box.map), 4),
         "precision": round(float(r.box.mp), 4),
         "recall": round(float(r.box.mr), 4)}
    print(f"  {tag:<28} {split:<5} @{imgsz}  "
          f"mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50_95']:.4f}  "
          f"P={m['precision']:.4f}  R={m['recall']:.4f}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--base", default=str(OUT / "yolo11s.pt"))
    ap.add_argument("--name", default="person_v1")
    ap.add_argument("--deploy-imgsz", type=int, default=960)  # cfg.PERSON_IMGSZ
    ap.add_argument("--eval-only", type=Path)
    args = ap.parse_args()

    from ultralytics import YOLO

    results: list[dict] = []

    if args.eval_only:
        model = YOLO(str(args.eval_only))
    else:
        print(f"training {args.base} → {args.name}  "
              f"({args.epochs} epochs, imgsz {args.imgsz}, batch {args.batch})\n")
        t0 = time.time()
        model = YOLO(args.base)
        model.train(
            data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device="mps", workers=8, patience=args.patience,
            project=str(OUT), name=args.name, exist_ok=True,
            pretrained=True, optimizer="auto", cos_lr=True, seed=0,
            # Roboflow already baked ~3.2 augmented copies per photo into the
            # source dataset; we kept one each (see build_person_dataset.py) and
            # let ultralytics do the augmentation, which it varies every epoch.
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
            scale=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=10,
            plots=True, val=True,
        )
        mins = (time.time() - t0) / 60
        print(f"\ntrained in {mins:.1f} min")

    best = OUT / args.name / "weights" / "best.pt"
    if not args.eval_only and best.exists():
        model = YOLO(str(best))

    print("\n===== Stage 1 results (leak-free splits) =====")
    # Baseline for comparison, restricted to COCO's person class.
    base = YOLO(args.base)
    results.append(evaluate(base, "baseline COCO yolo11s", args.imgsz, "val", classes=[0]))
    results.append(evaluate(model, "fine-tuned person_v1", args.imgsz, "val"))
    results.append(evaluate(model, "fine-tuned person_v1", args.deploy_imgsz, "val"))
    # The test split is touched ONCE, at the end, and never used to pick anything.
    results.append(evaluate(model, "fine-tuned person_v1", args.deploy_imgsz, "test"))

    rep = OUT / args.name / "stage1_metrics.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(results, indent=2))
    print(f"\nmetrics → {rep}")

    gate = next((r for r in results if r["tag"].startswith("fine-tuned")
                 and r["split"] == "val" and r["imgsz"] == args.deploy_imgsz), None)
    if gate:
        ok = gate["mAP50"] >= 0.75
        print(f"\nGate 1 (mAP50 ≥ 0.75 @ deploy imgsz): "
              f"{'✅ PASS' if ok else '❌ FAIL'} — {gate['mAP50']:.4f}")


if __name__ == "__main__":
    main()
