# training/upload.py — push auto-collected frames to a Roboflow project
# Usage:
#   python -m training.upload --task ppe
#   python -m training.upload --task fall --project my-fall-project
from __future__ import annotations
import argparse
from pathlib import Path

import config as cfg
from training.trainer import ZENTRATrainer


_CATEGORY = {"ppe": "ppe_violations", "fall": "fall_events", "zone": "zone_intrusions"}


def main():
    ap = argparse.ArgumentParser(description="Upload collected frames to Roboflow for labeling")
    ap.add_argument("--task", default="ppe", choices=["ppe", "fall", "zone"])
    ap.add_argument("--project", default=None, help="Roboflow project slug (default from config)")
    args = ap.parse_args()

    category = _CATEGORY[args.task]
    src = Path(cfg.COLLECTED_DIR) / category
    imgs = list(src.glob("*.jpg"))
    if not imgs:
        print(f"[Upload] No images in {src} — run the app to collect data first.")
        return

    project = args.project or (
        cfg.ROBOFLOW_PPE_PROJECT if args.task == "ppe" else cfg.ROBOFLOW_FALL_PROJECT
    )
    print(f"[Upload] {len(imgs)} images from '{category}' → Roboflow project '{project}'")
    print(f"[Upload] workspace: {cfg.ROBOFLOW_WORKSPACE}")

    trainer = ZENTRATrainer(task="ppe" if args.task != "fall" else "fall")
    trainer.upload_to_roboflow(str(src), project)
    print("[Upload] Done. Open Roboflow → Annotate to correct the labels.")


if __name__ == "__main__":
    main()
