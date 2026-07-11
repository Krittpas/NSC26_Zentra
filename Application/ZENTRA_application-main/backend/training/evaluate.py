# training/evaluate.py — ZENTRA model evaluation on a FROZEN test/holdout set
"""
Measure a trained model's accuracy against a FIXED held-out test set that is
never used for training, so numbers are comparable across model versions and
provable for judging (eval-gated promotion later builds on this).

Persists metrics JSON in the same format the app already reads
(server/api.py /api/training/metrics → Settings "ความแม่นยำโมเดล" card),
tagged source="holdout" so it's distinguishable from training-time metrics.

Usage:
  python -m training.evaluate --task ppe --data data/eval_holdout/data.yaml
  python -m training.evaluate --task ppe --model models/ppe_finetuned.pt --data <yaml>

Create the frozen set once (label by hand, NEVER train on it), e.g.:
  data/eval_holdout/
    images/*.jpg
    labels/*.txt        (YOLO format)
    data.yaml           (path/val -> images, names matching the model)
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path

import config as cfg


def evaluate(task: str, data_yaml: str, model_path: str | None = None,
             tag: str = "holdout") -> dict:
    """Run ultralytics validation on a frozen set and persist the metrics."""
    m_path = model_path or (cfg.PPE_LOCAL_MODEL if task == "ppe" else cfg.FALL_LOCAL_MODEL)
    if not Path(m_path).exists():
        print(f"[Eval] ❌ model not found: {m_path}")
        return {}
    if not Path(data_yaml).exists():
        print(f"[Eval] ❌ frozen test data.yaml not found: {data_yaml}")
        return {}

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    model   = YOLO(m_path)
    res     = model.val(data=data_yaml, imgsz=cfg.TRAIN_IMG_SIZE, verbose=True)
    metrics = {
        "mAP50":     float(res.box.map50),
        "mAP50-95":  float(res.box.map),
        "precision": float(res.box.mp),
        "recall":    float(res.box.mr),
    }

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp":  ts,
        "task":       task,
        "source":     tag,          # 'holdout' = measured on the frozen test set
        "model_path": str(m_path),
        "data":       data_yaml,
        **metrics,
    }
    mfile = Path(cfg.LOGS_DIR) / f"metrics_{task}_{ts}.json"
    mfile.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[Eval] ✅ metrics → {mfile}")
    print(f"[Eval] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50-95']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ZENTRA model evaluation on a frozen test set")
    ap.add_argument("--task",  default="ppe", choices=["ppe", "fall"])
    ap.add_argument("--data",  required=True, help="path to frozen test data.yaml")
    ap.add_argument("--model", default=None,  help="model .pt (default: configured fine-tuned model)")
    ap.add_argument("--tag",   default="holdout")
    args = ap.parse_args()
    evaluate(args.task, args.data, args.model, args.tag)
