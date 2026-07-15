# training/trainer.py — ZENTRA YOLOv8 Fine-tuning Pipeline
# รองรับ:
#   1. Fine-tune จาก Roboflow dataset
#   2. Fine-tune จาก auto-collected data (utils/collector.py)
#   3. Data augmentation (mosaic, mixup, flipping, HSV)
#   4. Export ONNX / TensorRT INT8
#   5. Upload ผลไป Roboflow (optional)
#   6. Auto-validate และ compare ก่อน deploy

from __future__ import annotations
import os
import json
import shutil
import time
import yaml
import random
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Optional


def _cfg():
    import config as c
    return c


# ──────────────────────────────────────────────────────────────
# Dataset Preparation
# ──────────────────────────────────────────────────────────────
class DatasetPreparer:
    """
    เตรียม dataset จาก:
    A) data/collected/ (auto-collected frames)
    B) Roboflow export (zip)
    C) ผสม A+B
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.cfg = _cfg()
        self.out = Path(output_dir or self.cfg.DATA_DIR / "train_dataset")
        self.out.mkdir(parents=True, exist_ok=True)

    # ── A: Prepare from collected ────────────────────────────
    def prepare_from_collected(self, val_split: float = 0.15) -> str:
        """
        แบ่ง train/val จาก data/collected/
        คืน path ไปยัง dataset.yaml
        """
        cfg      = self.cfg
        src_root = Path(cfg.COLLECTED_DIR)
        categories = ["ppe_violations", "zone_intrusions", "fall_events", "normal"]

        all_pairs: list[tuple[Path, Path]] = []
        for cat in categories:
            cat_dir = src_root / cat
            if not cat_dir.exists():
                continue
            imgs = sorted(cat_dir.glob("*.jpg"))
            for img in imgs:
                lbl = img.with_suffix(".txt")
                if lbl.exists():
                    all_pairs.append((img, lbl))

        if not all_pairs:
            raise ValueError("ไม่พบข้อมูลใน data/collected/ — รันระบบก่อนเพื่อเก็บ data")

        random.shuffle(all_pairs)
        n_val   = max(1, int(len(all_pairs) * val_split))
        val_set = all_pairs[:n_val]
        trn_set = all_pairs[n_val:]

        print(f"[Trainer] Dataset: {len(trn_set)} train, {len(val_set)} val")

        for split, pairs in [("train", trn_set), ("val", val_set)]:
            img_dir = self.out / "images" / split
            lbl_dir = self.out / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for img_p, lbl_p in pairs:
                shutil.copy2(img_p, img_dir / img_p.name)
                shutil.copy2(lbl_p, lbl_dir / lbl_p.name)

        return self._write_yaml()

    # ── B: Prepare from Roboflow export zip ──────────────────
    def prepare_from_roboflow_zip(self, zip_path: str) -> str:
        """Extract Roboflow YOLOv8 export zip แล้วคืน dataset.yaml path"""
        import zipfile
        extract_dir = self.out / "roboflow_raw"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        # Roboflow export มี data.yaml อยู่แล้ว
        yaml_candidates = list(extract_dir.rglob("data.yaml"))
        if yaml_candidates:
            src_yaml = yaml_candidates[0]
            # Update path ให้ชี้ถูก
            with open(src_yaml) as f:
                d = yaml.safe_load(f)
            d["path"] = str(src_yaml.parent.resolve())
            out_yaml = self.out / "dataset.yaml"
            with open(out_yaml, "w") as f:
                yaml.dump(d, f)
            print(f"[Trainer] Roboflow dataset ready: {out_yaml}")
            return str(out_yaml)
        raise FileNotFoundError("ไม่พบ data.yaml ใน zip")

    # ── C: Download from Roboflow API ────────────────────────
    def download_from_roboflow(self, project_name: str, version: int = 1) -> str:
        """
        ดาวน์โหลด dataset จาก Roboflow โดยตรง
        ต้องติดตั้ง roboflow package: pip install roboflow
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError("pip install roboflow")

        cfg = self.cfg
        location = self.out / "roboflow_dl"

        # ── Cache guard: reuse an already-downloaded dataset ──────────
        # Roboflow's own download uses overwrite=True → would re-pull the whole
        # (huge) dataset every run. Skip if a valid export is already present,
        # so smoke → subset → full runs download only ONCE. Force a fresh pull
        # with ZENTRA_FORCE_DOWNLOAD=1.
        force = os.getenv("ZENTRA_FORCE_DOWNLOAD", "0") == "1"
        cached_yaml = location / "data.yaml"
        has_imgs = any((location / "train" / "images").glob("*")) \
            if (location / "train" / "images").exists() else False
        if cached_yaml.exists() and has_imgs and not force:
            print(f"[Trainer] Reusing cached dataset at {location} "
                  f"(set ZENTRA_FORCE_DOWNLOAD=1 to re-download)")
            return self._finalize_yolo_dataset(str(location))

        rf  = Roboflow(api_key=cfg.ROBOFLOW_API_KEY)
        proj = rf.workspace(cfg.ROBOFLOW_WORKSPACE).project(project_name)
        ds   = proj.version(version).download(
            "yolov8",
            location=str(location),
            overwrite=True,
        )
        print(f"[Trainer] Downloaded: {ds.location}")
        return self._finalize_yolo_dataset(ds.location)

    @staticmethod
    def _finalize_yolo_dataset(location: str) -> str:
        """Make a Roboflow YOLOv8 export trainable regardless of how the version
        was generated: write ABSOLUTE paths (avoids ultralytics datasets_dir
        confusion) and create a val split if the export has none."""
        loc = Path(location)
        yml = loc / "data.yaml"
        d   = yaml.safe_load(yml.read_text()) if yml.exists() else {}
        names = d.get("names", [])

        train_img = loc / "train" / "images"
        train_lbl = loc / "train" / "labels"
        valid_img = loc / "valid" / "images"
        valid_lbl = loc / "valid" / "labels"

        has_valid = valid_img.exists() and any(valid_img.glob("*.*"))
        if not has_valid and train_img.exists():
            valid_img.mkdir(parents=True, exist_ok=True)
            valid_lbl.mkdir(parents=True, exist_ok=True)
            imgs = sorted(p for p in train_img.glob("*")
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            random.seed(0)
            random.shuffle(imgs)
            n_val = max(1, int(len(imgs) * 0.2))
            for p in imgs[:n_val]:
                shutil.move(str(p), str(valid_img / p.name))
                lp = train_lbl / f"{p.stem}.txt"
                if lp.exists():
                    shutil.move(str(lp), str(valid_lbl / lp.name))
            print(f"[Trainer] No val split in export → carved {n_val} images for validation")

        out = {
            "path":  str(loc.resolve()),
            "train": "train/images",
            "val":   "valid/images",
            "nc":    d.get("nc", len(names)),
            "names": names,
        }
        yml.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False))
        print(f"[Trainer] data.yaml normalised (absolute path, {out['nc']} classes)")
        return str(yml)

    # ── Augmentation (offline) ───────────────────────────────
    def augment_dataset(self, multiplier: int = 3):
        """
        Augment images ใน train set (offline, เก็บไว้ใน disk)
        multiplier = จำนวนรูป augmented ต่อ 1 รูปจริง
        """
        img_dir = self.out / "images" / "train"
        lbl_dir = self.out / "labels" / "train"
        if not img_dir.exists():
            print("[Trainer] ไม่พบ train images — ข้าม augmentation")
            return

        orig_imgs = sorted(img_dir.glob("*.jpg"))
        added = 0
        for img_path in orig_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            labels   = lbl_path.read_text() if lbl_path.exists() else ""

            for k in range(multiplier):
                aug_img, aug_lbl = self._augment(img, labels)
                stem  = f"{img_path.stem}_aug{k}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), aug_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                (lbl_dir / f"{stem}.txt").write_text(aug_lbl)
                added += 1

        print(f"[Trainer] Augmented {added} images added")

    @staticmethod
    def _augment(img: np.ndarray, labels: str) -> tuple[np.ndarray, str]:
        """Augment image + labels (horizontal flip, HSV, brightness)"""
        h, w = img.shape[:2]
        aug   = img.copy()
        new_labels = labels

        # Horizontal flip (50%)
        if random.random() < 0.5:
            aug = cv2.flip(aug, 1)
            new_lines = []
            for line in labels.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[1] = str(1.0 - float(parts[1]))   # cx flip
                    new_lines.append(" ".join(parts))
            new_labels = "\n".join(new_lines)

        # HSV augmentation
        if random.random() < 0.7:
            hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] += random.uniform(-18, 18)   # Hue
            hsv[:, :, 1] *= random.uniform(0.6, 1.4)  # Saturation
            hsv[:, :, 2] *= random.uniform(0.6, 1.4)  # Value
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Random crop (zoom in 0–20%)
        if random.random() < 0.3:
            margin = random.uniform(0, 0.15)
            x1 = int(w * margin)
            y1 = int(h * margin)
            aug = aug[y1:h - y1, x1:w - x1]
            aug = cv2.resize(aug, (w, h))

        return aug, new_labels

    # ── Private ───────────────────────────────────────────────
    def _write_yaml(self) -> str:
        """data.yaml for a dataset built from data/collected/.

        The class list MUST be cfg.PPE_TAXONOMY, in that exact order: the .txt
        labels being trained on were written by utils/collector.py, which emits
        indices in the taxonomy's index space (cfg.ppe_taxo_index). Building the
        names from sorted(PPE_CLASSES display labels) instead — as this did —
        produces a DIFFERENT class space (13 names, alphabetical), so every label
        index silently points at the wrong class and the run trains on noise
        without raising anything. The taxonomy is the single source of truth for
        indices; both sides must read it.
        """
        cfg         = self.cfg
        class_names = list(cfg.PPE_TAXONOMY)
        yaml_path   = self.out / "dataset.yaml"

        content = {
            "path":  str(self.out.resolve()),
            "train": "images/train",
            "val":   "images/val",
            "nc":    len(class_names),
            "names": class_names,
        }
        with open(yaml_path, "w") as f:
            yaml.dump(content, f, allow_unicode=True)
        print(f"[Trainer] dataset.yaml → {yaml_path}")
        return str(yaml_path)


# ──────────────────────────────────────────────────────────────
# YOLOv8 Fine-tuner
# ──────────────────────────────────────────────────────────────
class ZENTRATrainer:
    """
    Fine-tune YOLOv8 บน ZENTRA dataset

    Usage:
        trainer = ZENTRATrainer(task="ppe")
        yaml_path = trainer.prepare_dataset()
        trainer.train(yaml_path)
        trainer.export()
        trainer.validate(yaml_path)
    """

    def __init__(self, task: str = "ppe"):
        """
        task: 'ppe' | 'fall'
        """
        assert task in ("ppe", "fall"), "task ต้องเป็น 'ppe' หรือ 'fall'"
        self.task   = task
        self.cfg    = _cfg()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(self.cfg.MODELS_DIR) / f"{task}_{self.run_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Main Pipeline ────────────────────────────────────────
    def prepare_dataset(
        self,
        roboflow_zip:     Optional[str] = None,
        roboflow_project: Optional[str] = None,
        roboflow_version: int = 1,
        augment:          bool = True,
        aug_multiplier:   int  = 2,
    ) -> str:
        """
        เตรียม dataset — เลือก source:
        1. roboflow_zip   → จาก zip ที่ดาวน์โหลดไว้
        2. roboflow_project → ดาวน์โหลดจาก API
        3. ไม่ระบุ       → ใช้ auto-collected data
        """
        preparer = DatasetPreparer()

        used_roboflow = bool(roboflow_zip or roboflow_project)
        if roboflow_zip:
            yaml_path = preparer.prepare_from_roboflow_zip(roboflow_zip)
        elif roboflow_project:
            yaml_path = preparer.download_from_roboflow(roboflow_project, roboflow_version)
        else:
            yaml_path = preparer.prepare_from_collected()

        # Roboflow already applies augmentation when generating the version, and
        # its data lives in roboflow_dl/ — so only augment the locally-prepared set.
        if augment and not used_roboflow:
            preparer.augment_dataset(aug_multiplier)

        return yaml_path

    def train(self, dataset_yaml: str, resume: bool = False,
              epochs: Optional[int] = None, fraction: float = 1.0,
              pretrained: Optional[str] = None, freeze: Optional[int] = None) -> str:
        """
        รัน YOLOv8 fine-tuning

        epochs:     override จำนวน epochs (None = ใช้ cfg.TRAIN_EPOCHS).
        fraction:   สัดส่วน train set ที่ใช้ (1.0 = ทั้งหมด, 0.3 = subset run).
        pretrained: path ไป .pt เพื่อ fine-tune ต่อจากโมเดลนั้น (เช่น ppe_finetuned.pt
                    ตัว 0.324) — คนละอย่างกับ resume: เริ่มเทรน "ใหม่" บน dataset ใหม่
                    แต่ใช้ weights นี้เป็นจุดตั้งต้น (two-stage fine-tune).
        freeze:     freeze N เลเยอร์แรก (เช่น 10 = freeze backbone) → เร็วขึ้นบน MPS
                    และกัน overfit บน subset เล็ก.

        Returns: path ไปยัง best.pt
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        cfg = self.cfg
        n_epochs = int(epochs) if epochs else cfg.TRAIN_EPOCHS

        # เลือก base model
        if pretrained:
            # Two-stage fine-tune: start from these weights on a NEW dataset.
            # NOTE: does NOT set ultralytics resume (that would continue an old run).
            weights = pretrained
        elif self.task == "ppe":
            weights = cfg.PPE_LOCAL_MODEL if (
                Path(cfg.PPE_LOCAL_MODEL).exists() and resume
            ) else cfg.YOLO_BASE_MODEL
        else:
            weights = cfg.FALL_LOCAL_MODEL if (
                Path(cfg.FALL_LOCAL_MODEL).exists() and resume
            ) else cfg.YOLO_BASE_MODEL

        model = YOLO(weights)

        print(f"\n{'='*60}")
        print(f"  ZENTRA Training — Task: {self.task.upper()}")
        print(f"  Base weights : {weights}")
        print(f"  Dataset      : {dataset_yaml}")
        print(f"  Epochs       : {n_epochs}")
        print(f"  Batch        : {cfg.TRAIN_BATCH_SIZE}")
        print(f"  Device       : {cfg.TRAIN_DEVICE}")
        print(f"{'='*60}\n")

        results = model.train(
            data       = dataset_yaml,
            epochs     = n_epochs,
            batch      = cfg.TRAIN_BATCH_SIZE,
            imgsz      = cfg.TRAIN_IMG_SIZE,
            device     = cfg.TRAIN_DEVICE,
            workers    = cfg.TRAIN_WORKERS,
            project    = str(self.out_dir),
            name       = self.task,
            lr0        = cfg.TRAIN_LR0,
            lrf        = cfg.TRAIN_LRF,
            momentum   = cfg.TRAIN_MOMENTUM,
            weight_decay = cfg.TRAIN_WEIGHT_DECAY,
            warmup_epochs = cfg.TRAIN_WARMUP_EPOCHS,
            cos_lr     = True,        # cosine LR schedule → converge นิ่งขึ้นในรอบยาว
            fraction   = fraction,    # subset run (เช่น 0.3) เพื่อดู trend เร็วบน MPS
            freeze     = freeze,      # freeze N เลเยอร์แรก (None = ไม่ freeze)
            # Augmentation
            augment    = cfg.TRAIN_AUG,
            mosaic     = 1.0,
            mixup      = 0.1,
            copy_paste = 0.3,         # เพิ่มจาก 0.1 → ช่วยคลาสหายาก (glasses/no_glasses/boots)
            flipud     = 0.0,
            fliplr     = 0.5,
            hsv_h      = 0.015,
            hsv_s      = 0.7,
            hsv_v      = 0.4,
            degrees    = 5.0,
            translate  = 0.1,
            scale      = 0.5,
            # Validation
            val        = True,
            plots      = True,
            save       = True,
            save_period = 10,
            resume     = resume,
            patience   = getattr(cfg, "TRAIN_PATIENCE", 15),    # Early stopping
            # Logging
            verbose    = True,
        )

        best_pt = self.out_dir / self.task / "weights" / "best.pt"
        if best_pt.exists():
            # Copy to models/
            target = Path(self.cfg.MODELS_DIR) / (
                "ppe_finetuned.pt" if self.task == "ppe" else "fall_finetuned.pt"
            )
            # Safety: never silently clobber the currently-deployed model. Back it
            # up to *_prev.pt first so a weak/experimental run can always be rolled
            # back (the raw best.pt also stays in this run's timestamped dir).
            if target.exists():
                prev = target.with_name(target.stem + "_prev.pt")
                shutil.copy2(target, prev)
                print(f"↩︎  Previous model backed up → {prev}")
            shutil.copy2(best_pt, target)
            print(f"\n✅ Best model saved → {target}")
            self._save_training_log(results, str(target))
            return str(target)
        else:
            raise FileNotFoundError(f"ไม่พบ best.pt ใน {best_pt}")

    def validate(self, dataset_yaml: str, model_path: Optional[str] = None) -> dict:
        """Validate model บน val set"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        cfg    = self.cfg
        m_path = model_path or (
            cfg.PPE_LOCAL_MODEL if self.task == "ppe" else cfg.FALL_LOCAL_MODEL
        )

        if not Path(m_path).exists():
            print(f"[Trainer] ไม่พบ model: {m_path}")
            return {}

        model   = YOLO(m_path)
        metrics = model.val(data=dataset_yaml, imgsz=cfg.TRAIN_IMG_SIZE, verbose=True)

        result = {
            "mAP50":    float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall":    float(metrics.box.mr),
        }
        print(f"\n📊 Validation Results ({self.task.upper()}):")
        for k, v in result.items():
            print(f"   {k}: {v:.4f}")
        return result

    def export(self, model_path: Optional[str] = None, formats: list[str] | None = None) -> dict[str, str]:
        """
        Export model เป็น format ต่างๆ
        formats: ['onnx', 'engine', 'coreml', 'tflite']
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        cfg    = self.cfg
        m_path = model_path or (
            cfg.PPE_LOCAL_MODEL if self.task == "ppe" else cfg.FALL_LOCAL_MODEL
        )

        if not Path(m_path).exists():
            print(f"[Trainer] ไม่พบ model: {m_path} — ข้าม export")
            return {}

        formats = formats or ["onnx"]
        model   = YOLO(m_path)
        paths   = {}

        for fmt in formats:
            try:
                print(f"[Trainer] Exporting {fmt.upper()}...")
                kwargs: dict = {"format": fmt, "imgsz": cfg.TRAIN_IMG_SIZE}
                if fmt == "engine":
                    kwargs.update({"half": True, "int8": True, "device": cfg.TRAIN_DEVICE})
                elif fmt == "onnx":
                    kwargs.update({"simplify": True, "opset": 17})

                out = model.export(**kwargs)
                paths[fmt] = str(out)
                print(f"   ✅ {fmt}: {out}")
            except Exception as e:
                print(f"   ❌ {fmt} export failed: {e}")

        return paths

    def upload_to_roboflow(self, dataset_dir: str, project_name: str):
        """Upload collected data ขึ้น Roboflow project"""
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError("pip install roboflow")

        cfg = self.cfg
        rf  = Roboflow(api_key=cfg.ROBOFLOW_API_KEY)
        proj = rf.workspace(cfg.ROBOFLOW_WORKSPACE).project(project_name)

        img_dir = Path(dataset_dir)
        imgs    = list(img_dir.glob("**/*.jpg"))
        print(f"[Trainer] Uploading {len(imgs)} images to Roboflow '{project_name}'...")

        for img_path in imgs:
            ann_path = img_path.with_suffix(".txt")
            try:
                proj.upload(
                    image_path       = str(img_path),
                    annotation_path  = str(ann_path) if ann_path.exists() else None,
                    split            = "train",
                    num_retry_uploads = 3,
                )
            except Exception as e:
                print(f"   ⚠️ Upload failed {img_path.name}: {e}")

        print(f"[Trainer] Upload complete")

    # ── Private ───────────────────────────────────────────────
    def _save_training_log(self, results, model_path: str):
        log = {
            "timestamp":   self.run_id,
            "task":        self.task,
            "model_path":  model_path,
            "epochs":      self.cfg.TRAIN_EPOCHS,
        }
        log_file = Path(self.cfg.LOGS_DIR) / f"training_{self.task}_{self.run_id}.json"
        log_file.write_text(json.dumps(log, indent=2))
        print(f"[Trainer] Log saved → {log_file}")


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────
def run_training_pipeline(
    task:               str  = "ppe",
    roboflow_zip:       Optional[str] = None,
    roboflow_project:   Optional[str] = None,
    roboflow_version:   int  = 1,
    epochs:             Optional[int] = None,
    augment:            bool = True,
    export_onnx:        bool = True,
    upload_roboflow:    bool = False,
    fraction:           float = 1.0,
    dataset_yaml:       Optional[str] = None,
    resume:             bool = False,
    pretrained:         Optional[str] = None,
    freeze:             Optional[int] = None,
):
    """
    One-shot training pipeline

    Examples
    --------
    # Train PPE จาก auto-collected data
    run_training_pipeline(task="ppe")

    # Train จาก Roboflow zip
    run_training_pipeline(task="ppe", roboflow_zip="dataset.zip")

    # Download + train จาก Roboflow API
    run_training_pipeline(task="ppe", roboflow_project="zentra-ppe")
    """
    trainer   = ZENTRATrainer(task=task)
    if dataset_yaml:
        # Pre-built dataset (e.g. the class-balanced subset from
        # training.balance_subset). Skip the Roboflow download/prepare step and
        # train on it directly — its names/order must already match the deployed
        # 11-class taxonomy (balance_subset copies them verbatim).
        yaml_path = str(Path(dataset_yaml).resolve())
        print(f"[Trainer] Using pre-built dataset yaml: {yaml_path}")
    else:
        yaml_path = trainer.prepare_dataset(
            roboflow_zip     = roboflow_zip,
            roboflow_project = roboflow_project,
            roboflow_version = roboflow_version,
            augment          = augment,
        )
    model_path = trainer.train(yaml_path, epochs=epochs, fraction=fraction,
                               resume=resume, pretrained=pretrained, freeze=freeze)
    metrics    = trainer.validate(yaml_path, model_path)

    # Persist metrics so the app can display model accuracy (NSC proof)
    if metrics:
        try:
            cfg = _cfg()
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = {
                "timestamp":  ts,
                "task":       task,
                "model_path": model_path,
                **metrics,
            }
            mfile = Path(cfg.LOGS_DIR) / f"metrics_{task}_{ts}.json"
            mfile.write_text(json.dumps(out, indent=2))
            print(f"[Trainer] Metrics saved → {mfile}")
        except Exception as e:
            print(f"[Trainer] metrics save failed: {e}")

    if export_onnx:
        trainer.export(model_path, formats=["onnx"])

    if upload_roboflow:
        project = _cfg().ROBOFLOW_PPE_PROJECT if task == "ppe" else _cfg().ROBOFLOW_FALL_PROJECT
        trainer.upload_to_roboflow(str(Path(_cfg().COLLECTED_DIR) / "ppe_violations"), project)

    print(f"\n🎉 Training pipeline complete — {task.upper()}")
    print(f"   Model: {model_path}")
    if metrics:
        print(f"   mAP50: {metrics.get('mAP50', 0):.4f}")
    return model_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ZENTRA YOLOv8 Trainer")
    parser.add_argument("--task",    default="ppe", choices=["ppe", "fall"])
    parser.add_argument("--zip",     default=None,  help="Roboflow dataset zip path")
    parser.add_argument("--project", default=None,  help="Roboflow project name")
    parser.add_argument("--version", type=int, default=1, help="Roboflow version number (เลข ไม่ใช่ชื่อ)")
    parser.add_argument("--epochs",  type=int, default=None, help="override epochs (เช่น 3 สำหรับ smoke test)")
    parser.add_argument("--fraction", type=float, default=1.0, help="สัดส่วน train set (เช่น 0.3 = subset run เร็วๆ)")
    parser.add_argument("--data",    default=None, help="pre-built data.yaml (เช่น balanced subset) — ข้าม Roboflow download")
    parser.add_argument("--resume",  action="store_true", help="resume จาก last.pt ของรันก่อนหน้า (รันยาวข้าม session)")
    parser.add_argument("--pretrained", default=None, help="fine-tune ต่อจาก .pt นี้ (เช่น models/ppe_finetuned.pt ตัว 0.324)")
    parser.add_argument("--freeze",  type=int, default=None, help="freeze N เลเยอร์แรก (เช่น 10 = freeze backbone)")
    parser.add_argument("--no-aug",  action="store_true")
    parser.add_argument("--export",  action="store_true")
    parser.add_argument("--upload",  action="store_true")
    args = parser.parse_args()

    run_training_pipeline(
        task             = args.task,
        roboflow_zip     = args.zip,
        roboflow_project = args.project,
        roboflow_version = args.version,
        epochs           = args.epochs,
        augment          = not args.no_aug,
        export_onnx      = args.export,
        upload_roboflow  = args.upload,
        fraction         = args.fraction,
        dataset_yaml     = args.data,
        resume           = args.resume,
        pretrained       = args.pretrained,
        freeze           = args.freeze,
    )
