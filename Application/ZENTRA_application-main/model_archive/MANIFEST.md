# ZENTRA — Model Archive

Archived 2026-07-10. Every trained artifact produced before the
from-scratch retraining effort, moved (not copied) out of the live tree so
retraining cannot overwrite them. Nothing here is loaded by the running app.

Live weights still in `backend/models/`: ppe_finetuned.pt, yolo11s.pt,
yolo11n-pose.pt (+ backend/yolov8s.pt, backend/assets/models/*.tflite).

> ⚠️  mAP figures in these filenames were measured on a Roboflow val split
> with **augmentation leakage** (~24.5% of val base-images also appear in
> train). They are optimistic. See docs/TRAINING_PIPELINE.md § Stage 0.

## Files

| SHA256 (12) | Size | Path |
|---|---:|---|
| `d5ffc1a67495` | 39M | `backbones_unused/yolo11m.pt` |
| `8529c383197a` | 9.4M | `backbones_unused/yolo26n-reid.onnx` |
| `5d4a90cdc7a2` | 50M | `backbones_unused/yolov8m.pt` |
| `08ba8b426ec5` | 85M | `ppe_finetunes/ppe_224142_best_0.29_SAFE.pt` |
| `41a0e17ad92f` | 22M | `ppe_finetunes/ppe_finetuned_backup_213947_0.19.pt` |
| `fc1568316e5e` | 22M | `ppe_finetunes/ppe_finetuned_prev.pt` |
| `08ba8b426ec5` | 85M | `ppe_finetunes/ppe_subset_v3_e16_map0308.pt` |
| `7d4020377528` | 22M | `ppe_finetunes/ppe_v3_full20ep_map0324.pt` |
| `f73c5420db33` | 22M | `ppe_finetunes/ppe_v3ext_full_map0698.pt` |
| `fc1568316e5e` | 22M | `ppe_finetunes/ppe_v3ext_smoke_map0624.pt` |
| `41a0e17ad92f` | 22M | `train_runs/ppe_20260703_213947/ppe/weights/best.pt` |
| `f4a64dac7277` | 85M | `train_runs/ppe_20260703_213947/ppe/weights/epoch0.pt` |
| `41fb20c5d66c` | 22M | `train_runs/ppe_20260703_213947/ppe/weights/last.pt` |
| `7d4020377528` | 22M | `train_runs/ppe_20260703_224142/ppe/weights/best.pt` |
| `19520176bfcb` | 85M | `train_runs/ppe_20260703_224142/ppe/weights/epoch0.pt` |
| `bcee5051e8e0` | 85M | `train_runs/ppe_20260703_224142/ppe/weights/epoch10.pt` |
| `62cf18bc47b9` | 22M | `train_runs/ppe_20260703_224142/ppe/weights/last.pt` |
| `142f7cebfa93` | 21M | `train_runs/runs_detect/detect/ab/cont_freeze10/weights/best.pt` |
| `e2679a259271` | 21M | `train_runs/runs_detect/detect/ab/cont_freeze10/weights/last.pt` |
| `5ee3b472e431` | 85M | `train_runs/runs_detect/detect/ab/cont_unfrozen/weights/best.pt` |
| `7f4dc090566b` | 85M | `train_runs/runs_detect/detect/ab/cont_unfrozen/weights/last.pt` |
| `f73c5420db33` | 22M | `train_runs/runs_detect/detect/models/phase2_full/ppe/weights/best.pt` |
| `fa3a84366459` | 22M | `train_runs/runs_detect/detect/models/phase2_full/ppe/weights/last.pt` |
| `fc1568316e5e` | 22M | `train_runs/runs_detect/detect/models/phase2_smoke/ppe/weights/best.pt` |
| `313020e2589f` | 22M | `train_runs/runs_detect/detect/models/phase2_smoke/ppe/weights/last.pt` |
| `cd22220fcc31` | 22M | `train_runs/runs_detect/detect/models/phase3_960/ppe/weights/best.pt` |
| `e2f307da56c0` | 22M | `train_runs/runs_detect/detect/models/phase3_960/ppe/weights/last.pt` |
| `ec3d0f849cfc` | 22M | `train_runs/runs_detect/detect/models/phase3_boots/ppe/weights/best.pt` |
| `f8002e772bb0` | 22M | `train_runs/runs_detect/detect/models/phase3_boots/ppe/weights/last.pt` |

## Layout

```
ppe_finetunes/     standalone PPE snapshots (name encodes reported mAP)
train_runs/        full ultralytics run dirs — weights + curves + confusion matrices
backbones_unused/  COCO backbones no code path loads
logs/              training metric JSON
```

## Restore

```bash
cp model_archive/ppe_finetunes/ppe_v3ext_full_map0698.pt backend/models/ppe_finetuned.pt
```

## Provenance warnings (found by checksum, not by trust)

**The deployed model is identified:** `backend/models/ppe_finetuned.pt` is
byte-identical to `ppe_finetunes/ppe_v3ext_full_map0698.pt`
(= `train_runs/runs_detect/detect/models/phase2_full/ppe/weights/best.pt`).

**Filenames disagree with each other.** These pairs are the SAME FILE under names
claiming different scores, so at least one number in each pair is wrong:

| SHA256 (12) | Names claiming to differ |
|---|---|
| `08ba8b426ec5` | `ppe_224142_best_0.29_SAFE.pt` **vs** `ppe_subset_v3_e16_map0308.pt` |
| `fc1568316e5e` | `ppe_finetuned_prev.pt` **vs** `ppe_v3ext_smoke_map0624.pt` |

Also note `ppe_224142_best_0.29_SAFE.pt` and `ppe_subset_v3_e16_map0308.pt` are
85 MB, i.e. checkpoints carrying an optimizer state, not stripped weights.

**Do not trust an mAP baked into a filename.** Re-measure on the leak-free split
(Stage 0) before comparing anything to anything.
