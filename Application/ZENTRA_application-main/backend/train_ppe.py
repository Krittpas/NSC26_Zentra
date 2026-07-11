#!/usr/bin/env python3
# train_ppe.py — ZENTRA PPE Detection Training Script (Standalone)
# ใช้งานได้ทันทีไม่ต้องมี data เดิม
#
# วิธีใช้:
#   python3 train_ppe.py --mode download   ← ดาวน์โหลด dataset จาก Roboflow
#   python3 train_ppe.py --mode train      ← เทรนจาก dataset ที่มีอยู่
#   python3 train_ppe.py --mode all        ← ดาวน์โหลด + merge + augment + เทรน
#   python3 train_ppe.py --mode validate   ← ทดสอบ model ที่เทรนแล้ว
#   python3 train_ppe.py --mode export     ← export เป็น ONNX / TensorRT

import os
import sys
import yaml
import json
import shutil
import random
import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════
ROBOFLOW_API_KEY  = os.getenv("ROBOFLOW_API_KEY",  "8xTIheqbzg4mkSLOdFe6")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "pholawats-workspace")

BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
MODELS_DIR   = BASE_DIR / "models"
RUNS_DIR     = BASE_DIR / "runs" / "ppe"
MERGED_DIR   = DATA_DIR / "merged_ppe"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Class Mapping สำหรับ ZENTRA ──────────────────────────────
# รวม class จากหลาย dataset → class กลาง
ZENTRA_CLASSES = [
    "helmet",        # 0
    "no_helmet",     # 1
    "vest",          # 2
    "no_vest",       # 3
    "gloves",        # 4
    "no_gloves",     # 5
    "goggles",       # 6
    "no_goggles",    # 7
    "safety_boots",  # 8
    "no_safety_boots", # 9
    "person",        # 10
]
NUM_CLASSES = len(ZENTRA_CLASSES)
CLS_MAP = {name: i for i, name in enumerate(ZENTRA_CLASSES)}

# Alias map — แมป class ชื่อต่างกันจากหลาย dataset → ZENTRA class
ALIAS_MAP = {
    # helmet variants
    "hard-hat": "helmet", "hardhat": "helmet", "hard_hat": "helmet",
    "helmet": "helmet", "head": "helmet",
    # no_helmet
    "no-hardhat": "no_helmet", "no_hardhat": "no_helmet",
    "no-helmet": "no_helmet", "no_helmet": "no_helmet",
    "no helmet": "no_helmet",
    # vest variants
    "safety-vest": "vest", "safety_vest": "vest",
    "vest": "vest", "reflective-vest": "vest",
    "hi-vis": "vest", "hi_vis": "vest",
    # no_vest
    "no-vest": "no_vest", "no_vest": "no_vest", "no vest": "no_vest",
    # gloves
    "gloves": "gloves", "safety-gloves": "gloves",
    # no_gloves
    "no-gloves": "no_gloves", "no_gloves": "no_gloves", "no gloves": "no_gloves",
    # goggles / glasses
    "goggles": "goggles", "safety-glasses": "goggles",
    "glasses": "goggles", "eye-protection": "goggles",
    # no_goggles
    "no-goggles": "no_goggles", "no_goggles": "no_goggles",
    "no-glasses": "no_goggles", "no glasses": "no_goggles",
    # boots
    "safety-boots": "safety_boots", "safety_boots": "safety_boots",
    "boots": "safety_boots",
    # no_boots
    "no-boots": "no_safety_boots", "no_boots": "no_safety_boots",
    "no boots": "no_safety_boots",
    # person
    "person": "person", "worker": "person", "human": "person",
    "people": "person",
}


# ════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD DATASETS
# ════════════════════════════════════════════════════════════════
def download_datasets() -> list[str]:
    """
    ดาวน์โหลด PPE datasets จาก Roboflow Universe
    คืน list ของ path ที่ดาวน์โหลด
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    # รายการ dataset ที่จะดาวน์โหลด (workspace/project/version)
    # เพิ่ม/ลด dataset ได้ที่นี่
    DATASETS = [
        # Dataset หลัก (ของทีม)
        {
            "workspace": ROBOFLOW_WORKSPACE,
            "project":   "ppe-cpxsz",
            "version":   2,
            "dest":      "data/ds_main",
        },
        # Roboflow Universe — PPE ทั่วไป (public)
        {
            "workspace": "roboflow-universe-projects",
            "project":   "construction-site-safety",
            "version":   1,
            "dest":      "data/ds_construction",
        },
        # Hard Hat Workers (7,000+ images)
        {
            "workspace": "joseph-nelson",
            "project":   "hard-hat-workers",
            "version":   2,
            "dest":      "data/ds_hardhat",
        },
        # PPE Dataset (Kaggle-sourced, ขึ้น Roboflow)
        {
            "workspace": "roboflow-universe-projects",
            "project":   "ppe-detection-ljb7d",
            "version":   4,
            "dest":      "data/ds_ppe",
        },
    ]

    downloaded = []
    for ds in DATASETS:
        dest = BASE_DIR / ds["dest"]
        if dest.exists() and any(dest.rglob("*.jpg")):
            print(f"✅ Already exists: {ds['project']} → {dest}")
            downloaded.append(str(dest))
            continue

        print(f"\n📥 Downloading: {ds['workspace']}/{ds['project']} v{ds['version']}...")
        try:
            proj = rf.workspace(ds["workspace"]).project(ds["project"])
            result = proj.version(ds["version"]).download(
                "yolov8",
                location=str(dest),
                overwrite=True,
            )
            print(f"   ✅ Downloaded to {dest}")
            downloaded.append(str(dest))
        except Exception as e:
            print(f"   ⚠️  ไม่สามารถดาวน์โหลด {ds['project']}: {e}")
            print(f"      ข้ามและใช้ dataset อื่นแทน")

    print(f"\n📦 Downloaded {len(downloaded)} datasets")
    return downloaded


# ════════════════════════════════════════════════════════════════
# STEP 2: MERGE & REMAP DATASETS
# ════════════════════════════════════════════════════════════════
def read_dataset_yaml(ds_path: str) -> dict:
    """อ่าน data.yaml จาก dataset directory"""
    ds_dir = Path(ds_path)
    for yaml_name in ["data.yaml", "dataset.yaml", "_annotations.yaml"]:
        y = ds_dir / yaml_name
        if y.exists():
            with open(y) as f:
                return yaml.safe_load(f)
    # ค้นหา subdirectory
    for y in ds_dir.rglob("data.yaml"):
        with open(y) as f:
            return yaml.safe_load(f)
    return {}


def remap_label(src_class_id: int, src_names: list[str]) -> int | None:
    """
    แปลง class ID จาก source dataset → ZENTRA class ID
    คืน None ถ้า class นั้นไม่ต้องการ
    """
    if src_class_id >= len(src_names):
        return None
    src_name = src_names[src_class_id].lower().strip()

    # ค้นหาใน alias map
    zentra_name = ALIAS_MAP.get(src_name)
    if zentra_name:
        return CLS_MAP.get(zentra_name)

    # ลอง partial match
    for alias, target in ALIAS_MAP.items():
        if alias in src_name or src_name in alias:
            return CLS_MAP.get(target)

    return None


def merge_datasets(dataset_paths: list[str], val_split: float = 0.15) -> str:
    """
    Merge หลาย dataset → single ZENTRA dataset
    - Remap class IDs
    - Train/Val split
    - คืน path ของ merged dataset.yaml
    """
    print(f"\n🔀 Merging {len(dataset_paths)} datasets...")

    merged = MERGED_DIR
    for split in ["train", "val"]:
        (merged / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats       = {c: 0 for c in ZENTRA_CLASSES}
    total_imgs  = 0
    skipped     = 0

    for ds_path in dataset_paths:
        ds_dir  = Path(ds_path)
        ds_info = read_dataset_yaml(ds_path)
        src_names = ds_info.get("names", [])
        if not src_names:
            # พยายามอ่าน names จากโฟลเดอร์
            src_names = _infer_class_names(ds_dir)

        print(f"\n  📁 {ds_dir.name} — classes: {src_names[:5]}{'...' if len(src_names)>5 else ''}")

        # หา images
        img_files = []
        for split_name in ["train", "valid", "val", "test", ""]:
            img_dir = ds_dir / "images" / split_name if split_name else ds_dir / "images"
            if not img_dir.exists():
                img_dir = ds_dir / split_name if split_name else ds_dir
            img_files += list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        print(f"     Found {len(img_files)} images")
        random.shuffle(img_files)

        for img_path in img_files:
            # หา label file
            lbl_path = _find_label(img_path, ds_dir)
            if not lbl_path or not lbl_path.exists():
                skipped += 1
                continue

            # อ่านและ remap labels
            new_lines = []
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                src_id = int(parts[0])
                new_id = remap_label(src_id, src_names)
                if new_id is None:
                    continue
                new_lines.append(f"{new_id} " + " ".join(parts[1:5]))
                stats[ZENTRA_CLASSES[new_id]] += 1

            if not new_lines:
                skipped += 1
                continue

            # Split train/val
            split = "val" if random.random() < val_split else "train"
            stem  = f"{ds_dir.name}_{total_imgs:06d}"

            # Copy + convert image
            out_img = merged / "images" / split / f"{stem}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Write label
            out_lbl = merged / "labels" / split / f"{stem}.txt"
            out_lbl.write_text("\n".join(new_lines))
            total_imgs += 1

    print(f"\n  ✅ Merged: {total_imgs} images, {skipped} skipped")
    print(f"  Class distribution:")
    for cls, cnt in stats.items():
        bar = "█" * min(cnt // 20, 40)
        print(f"    {cls:<20} {cnt:>5} {bar}")

    # Write merged dataset.yaml
    yaml_path = merged / "dataset.yaml"
    yaml.dump({
        "path":  str(merged.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NUM_CLASSES,
        "names": ZENTRA_CLASSES,
    }, open(yaml_path, "w"), allow_unicode=True)

    print(f"\n  📄 dataset.yaml → {yaml_path}")
    return str(yaml_path)


def _find_label(img_path: Path, ds_root: Path) -> Path | None:
    """หา label .txt จาก image path"""
    # ลอง images/ → labels/
    for part in img_path.parts:
        if part == "images":
            lbl = Path(str(img_path).replace("/images/", "/labels/", 1)).with_suffix(".txt")
            if lbl.exists():
                return lbl
    # ลองข้างๆ
    lbl = img_path.with_suffix(".txt")
    return lbl if lbl.exists() else None


def _infer_class_names(ds_dir: Path) -> list[str]:
    """พยายาม infer class names จาก label files"""
    # ถ้าไม่มี yaml → คืน list ว่าง
    return []


# ════════════════════════════════════════════════════════════════
# STEP 3: AUGMENTATION (offline เพิ่มข้อมูลให้มากขึ้น)
# ════════════════════════════════════════════════════════════════
def augment_dataset(dataset_yaml: str, multiplier: int = 2):
    """
    เพิ่มข้อมูล train set ด้วย augmentation แบบ offline
    multiplier = กี่เท่าของ original (2 = images เพิ่มขึ้น 2x)
    """
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    base     = Path(cfg["path"])
    img_dir  = base / "images" / "train"
    lbl_dir  = base / "labels" / "train"

    orig_imgs = sorted(img_dir.glob("*.jpg"))
    print(f"\n🎨 Augmenting {len(orig_imgs)} images (x{multiplier})...")

    added = 0
    for img_path in orig_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        labels   = lbl_path.read_text() if lbl_path.exists() else ""

        for k in range(multiplier):
            aug_img, aug_lbl = _apply_augmentation(img, labels, k)
            stem  = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            (lbl_dir / f"{stem}.txt").write_text(aug_lbl)
            added += 1

    print(f"   ✅ Added {added} augmented images")
    print(f"   Total train: {len(list(img_dir.glob('*.jpg')))} images")


def _apply_augmentation(img: np.ndarray, labels: str, seed: int) -> tuple:
    """
    หลาย augmentation เลือกแบบสุ่มตาม seed
    """
    random.seed(seed + int(time.time()) % 1000)
    np.random.seed(seed)
    import time as _time; _ = _time  # noqa

    h, w   = img.shape[:2]
    aug    = img.copy()
    lines  = [l.strip() for l in labels.splitlines() if l.strip()]
    new_lines = list(lines)

    # ── 1. Horizontal Flip ──────────────────────────────────
    if random.random() < 0.5:
        aug = cv2.flip(aug, 1)
        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                parts[1] = f"{1.0 - float(parts[1]):.6f}"
            new_lines.append(" ".join(parts))

    # ── 2. HSV Jitter ───────────────────────────────────────
    if random.random() < 0.8:
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + random.uniform(-20, 20), 0, 179)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.5, 1.5), 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.5, 1.5), 0, 255)
        aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 3. Brightness / Contrast ────────────────────────────
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)   # contrast
        beta  = random.uniform(-30, 30)    # brightness
        aug   = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # ── 4. Gaussian Blur (simulate oof) ─────────────────────
    if random.random() < 0.2:
        ksize = random.choice([3, 5])
        aug   = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    # ── 5. Random Noise ─────────────────────────────────────
    if random.random() < 0.2:
        noise = np.random.normal(0, random.uniform(5, 20), aug.shape).astype(np.int16)
        aug   = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── 6. Small Rotation (±10°) ────────────────────────────
    if random.random() < 0.3:
        angle = random.uniform(-10, 10)
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug   = cv2.warpAffine(aug, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)
        # NOTE: bbox rotation ซับซ้อน — ใช้ original labels (approximation)

    # ── 7. Random Crop / Zoom ───────────────────────────────
    if random.random() < 0.3:
        margin_x = int(w * random.uniform(0, 0.1))
        margin_y = int(h * random.uniform(0, 0.1))
        if margin_x > 0 and margin_y > 0:
            aug = aug[margin_y:h-margin_y, margin_x:w-margin_x]
            aug = cv2.resize(aug, (w, h))
            # Adjust bbox
            ratio_x = w / (w - 2*margin_x)
            ratio_y = h / (h - 2*margin_y)
            off_x   = margin_x / w
            off_y   = margin_y / h
            adjusted = []
            for line in new_lines:
                parts = line.split()
                if len(parts) >= 5:
                    cx = (float(parts[1]) - off_x) * ratio_x
                    cy = (float(parts[2]) - off_y) * ratio_y
                    bw = float(parts[3]) * ratio_x
                    bh = float(parts[4]) * ratio_y
                    if 0.01 < cx < 0.99 and 0.01 < cy < 0.99:
                        cx = max(0.01, min(0.99, cx))
                        cy = max(0.01, min(0.99, cy))
                        bw = min(bw, 1.0)
                        bh = min(bh, 1.0)
                        adjusted.append(f"{parts[0]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    # ถ้า center ออกนอก frame → ข้ามไป
            new_lines = adjusted

    return aug, "\n".join(new_lines)


# ════════════════════════════════════════════════════════════════
# STEP 4: TRAIN
# ════════════════════════════════════════════════════════════════
def train(
    dataset_yaml:   str,
    base_model:     str  = "yolov8m.pt",
    epochs:         int  = 100,
    batch:          int  = 16,
    imgsz:          int  = 640,
    device:         str  = "0",
    resume:         bool = False,
    pretrained_pt:  str | None = None,
) -> str:
    """
    รัน YOLOv8 training

    คืน path ของ best.pt
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ pip install ultralytics")
        sys.exit(1)

    # เลือก weights
    if pretrained_pt and Path(pretrained_pt).exists():
        weights = pretrained_pt
        print(f"🔄 Fine-tune จาก: {weights}")
    elif (MODELS_DIR / "ppe_finetuned.pt").exists() and resume:
        weights = str(MODELS_DIR / "ppe_finetuned.pt")
        print(f"🔄 Resume จาก: {weights}")
    else:
        weights = base_model
        print(f"🆕 Base model: {weights}")

    model    = YOLO(weights)
    run_name = f"ppe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'═'*60}")
    print(f"  🚀 ZENTRA PPE Training")
    print(f"  Dataset  : {dataset_yaml}")
    print(f"  Epochs   : {epochs}")
    print(f"  Batch    : {batch}")
    print(f"  ImgSize  : {imgsz}")
    print(f"  Device   : {device}")
    print(f"  Classes  : {NUM_CLASSES}")
    print(f"{'═'*60}\n")

    results = model.train(
        data    = dataset_yaml,
        epochs  = epochs,
        batch   = batch,
        imgsz   = imgsz,
        device  = device,
        project = str(RUNS_DIR),
        name    = run_name,

        # ── Optimizer ──────────────────────────────────────
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 0.0005,
        warmup_epochs = 5,
        warmup_momentum = 0.8,

        # ── Augmentation (YOLOv8 built-in) ─────────────────
        augment    = True,
        mosaic     = 1.0,        # mosaic (ผสม 4 ภาพ)
        mixup      = 0.15,       # mixup 15%
        copy_paste = 0.1,        # copy-paste
        close_mosaic = 15,       # ปิด mosaic ช่วง 15 epochs สุดท้าย
        fliplr     = 0.5,        # flip horizontal
        flipud     = 0.0,
        hsv_h      = 0.015,      # hue
        hsv_s      = 0.7,        # saturation
        hsv_v      = 0.4,        # value
        degrees    = 5.0,        # rotation
        translate  = 0.1,
        scale      = 0.6,
        shear      = 2.0,
        perspective = 0.0005,
        erasing    = 0.4,        # random erasing (simulate occlusion)
        crop_fraction = 1.0,

        # ── Loss weights ───────────────────────────────────
        box   = 7.5,
        cls   = 0.5,
        dfl   = 1.5,

        # ── Training control ────────────────────────────────
        patience      = 30,      # Early stopping
        save          = True,
        save_period   = 10,      # บันทึกทุก 10 epochs
        val           = True,
        plots         = True,
        verbose       = True,
        workers       = 8,
        cache         = False,   # True ถ้า RAM มากพอ (>16GB)
        amp           = True,    # Mixed precision (เร็วขึ้น ~40%)
        multi_scale   = False,
        overlap_mask  = True,
        mask_ratio    = 4,
        dropout       = 0.0,
        seed          = 42,
        deterministic = True,
        resume        = resume,
    )

    # ── Copy best.pt ────────────────────────────────────────
    best_pt = RUNS_DIR / run_name / "weights" / "best.pt"
    if best_pt.exists():
        target = MODELS_DIR / "ppe_finetuned.pt"
        shutil.copy2(best_pt, target)
        print(f"\n✅ Best model → {target}")
        _save_run_info(run_name, dataset_yaml, epochs, results)
        return str(target)
    else:
        # fallback: หา best.pt ที่ไหนก็ได้
        candidates = list(RUNS_DIR.rglob("best.pt"))
        if candidates:
            shutil.copy2(candidates[-1], MODELS_DIR / "ppe_finetuned.pt")
            return str(MODELS_DIR / "ppe_finetuned.pt")
        raise FileNotFoundError("ไม่พบ best.pt")


# ════════════════════════════════════════════════════════════════
# STEP 5: VALIDATE
# ════════════════════════════════════════════════════════════════
def validate(dataset_yaml: str, model_path: str | None = None) -> dict:
    """ทดสอบ model และแสดงผล metrics"""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics")

    m_path = model_path or str(MODELS_DIR / "ppe_finetuned.pt")
    if not Path(m_path).exists():
        print(f"❌ ไม่พบ model: {m_path}")
        return {}

    print(f"\n🧪 Validating: {m_path}")
    model   = YOLO(m_path)
    metrics = model.val(data=dataset_yaml, imgsz=640, verbose=True, plots=True)

    result = {
        "mAP50":     float(metrics.box.map50),
        "mAP50-95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }

    print(f"\n{'═'*50}")
    print(f"  📊 Validation Results — PPE Detection")
    print(f"{'═'*50}")
    print(f"  mAP50     : {result['mAP50']:.4f}  ({result['mAP50']*100:.1f}%)")
    print(f"  mAP50-95  : {result['mAP50-95']:.4f}")
    print(f"  Precision : {result['precision']:.4f}")
    print(f"  Recall    : {result['recall']:.4f}")
    print(f"{'═'*50}")

    # Per-class metrics
    if hasattr(metrics.box, "ap_class_index"):
        print("\n  Per-Class AP50:")
        for i, cls_i in enumerate(metrics.box.ap_class_index):
            if cls_i < len(ZENTRA_CLASSES):
                ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
                bar = "█" * int(ap * 30)
                print(f"    {ZENTRA_CLASSES[cls_i]:<20} {ap:.3f} {bar}")

    return result


# ════════════════════════════════════════════════════════════════
# STEP 6: EXPORT
# ════════════════════════════════════════════════════════════════
def export_model(model_path: str | None = None, formats: list | None = None):
    """Export เป็น ONNX / TensorRT / CoreML"""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics")

    m_path  = model_path or str(MODELS_DIR / "ppe_finetuned.pt")
    formats = formats or ["onnx"]

    if not Path(m_path).exists():
        print(f"❌ ไม่พบ model: {m_path}")
        return

    model = YOLO(m_path)
    print(f"\n📦 Exporting: {m_path}")

    for fmt in formats:
        print(f"   → {fmt.upper()}...")
        try:
            if fmt == "onnx":
                out = model.export(format="onnx", imgsz=640, simplify=True, opset=17)
            elif fmt == "engine":   # TensorRT
                out = model.export(format="engine", imgsz=640,
                                   half=True, int8=False, device=0)
            elif fmt == "tflite":
                out = model.export(format="tflite", imgsz=640)
            else:
                out = model.export(format=fmt, imgsz=640)
            print(f"      ✅ {out}")
        except Exception as e:
            print(f"      ❌ {e}")


# ════════════════════════════════════════════════════════════════
# STEP 7: TEST ON CAMERA / VIDEO
# ════════════════════════════════════════════════════════════════
def test_live(model_path: str | None = None, source: str = "0"):
    """ทดสอบ model บน webcam หรือ video file"""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics")

    m_path = model_path or str(MODELS_DIR / "ppe_finetuned.pt")
    if not Path(m_path).exists():
        print(f"❌ ไม่พบ model: {m_path}")
        return

    model = YOLO(m_path)
    print(f"\n🎥 Live test: {m_path}")
    print(f"   Source: {source}")
    print(f"   กด Q เพื่อหยุด")

    # Convert source
    src = int(source) if source.isdigit() else source

    cap = cv2.VideoCapture(src)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.45, iou=0.45, verbose=False)
        annotated = results[0].plot()

        # แสดง class ที่ตรวจพบ
        detected = [ZENTRA_CLASSES[int(c)] for c in results[0].boxes.cls.cpu().numpy()
                    if int(c) < len(ZENTRA_CLASSES)] if len(results[0].boxes) > 0 else []
        violations = [c for c in detected if c.startswith("no_")]
        color = (0, 0, 220) if violations else (0, 200, 0)
        status = f"⚠ VIOLATION: {', '.join(violations)}" if violations else "✅ Compliant"
        cv2.putText(annotated, status, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("ZENTRA PPE — Live Test (Q=Quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ════════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════════
def _save_run_info(run_name: str, dataset_yaml: str, epochs: int, results):
    info = {
        "run_name":     run_name,
        "dataset":      dataset_yaml,
        "epochs":       epochs,
        "timestamp":    datetime.now().isoformat(),
        "classes":      ZENTRA_CLASSES,
        "num_classes":  NUM_CLASSES,
    }
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"train_{run_name}.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False)
    )


def show_dataset_stats(dataset_yaml: str):
    """แสดงสถิติ dataset ก่อน train"""
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["path"])

    print(f"\n📊 Dataset Statistics: {dataset_yaml}")
    print(f"{'─'*50}")

    total_cls = {c: 0 for c in ZENTRA_CLASSES}
    for split in ["train", "val"]:
        lbl_dir = base / "labels" / split
        if not lbl_dir.exists():
            continue
        labels   = list(lbl_dir.glob("*.txt"))
        n_imgs   = len(list((base / "images" / split).glob("*.jpg")))
        n_labels = 0
        for lf in labels:
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    if cid < len(ZENTRA_CLASSES):
                        total_cls[ZENTRA_CLASSES[cid]] += 1
                        n_labels += 1
        print(f"  {split:<8}: {n_imgs:>5} images, {n_labels:>6} annotations")

    print(f"\n  Class breakdown:")
    for cls, cnt in total_cls.items():
        if cnt > 0:
            bar = "█" * min(cnt // 50, 35)
            print(f"    {cls:<22} {cnt:>5} {bar}")
    print(f"{'─'*50}")


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════
import time

def main():
    parser = argparse.ArgumentParser(
        description="ZENTRA PPE Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python3 train_ppe.py --mode all
  python3 train_ppe.py --mode download
  python3 train_ppe.py --mode train --dataset data/merged_ppe/dataset.yaml
  python3 train_ppe.py --mode train --epochs 150 --batch 32
  python3 train_ppe.py --mode validate
  python3 train_ppe.py --mode export --export-format onnx engine
  python3 train_ppe.py --mode test --source 0
        """
    )
    parser.add_argument("--mode", default="all",
        choices=["download", "merge", "augment", "train", "all", "validate", "export", "test"],
        help="Pipeline step to run")
    parser.add_argument("--dataset",  default=None, help="Path to dataset.yaml")
    parser.add_argument("--model",    default="yolov8m.pt",
        help="Base model (yolov8n/s/m/l/x.pt) หรือ path ของ .pt ที่มีอยู่")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--device",   default="0", help="0=GPU, cpu=CPU, 0,1=Multi-GPU")
    parser.add_argument("--augment-x", type=int, default=2,
        help="Offline augmentation multiplier")
    parser.add_argument("--resume",   action="store_true", help="Resume จาก checkpoint")
    parser.add_argument("--no-aug",   action="store_true", help="ข้าม offline augmentation")
    parser.add_argument("--export-format", nargs="+", default=["onnx"],
        choices=["onnx", "engine", "tflite", "coreml"])
    parser.add_argument("--source",   default="0", help="Camera index หรือ video path")
    parser.add_argument("--pretrained", default=None,
        help="Path ของ .pt ที่ fine-tune แล้ว (สำหรับ continue training)")

    args = parser.parse_args()

    dataset_yaml = args.dataset

    # ── Download ─────────────────────────────────────────────
    if args.mode in ("download", "all"):
        paths = download_datasets()
        if not paths:
            print("❌ ไม่สามารถดาวน์โหลด dataset ได้")
            sys.exit(1)

    # ── Merge ────────────────────────────────────────────────
    if args.mode in ("merge", "all"):
        if not dataset_yaml:
            # ค้นหา dataset ที่ดาวน์โหลดไว้
            ds_dirs = [str(d) for d in (BASE_DIR / "data").iterdir()
                       if d.is_dir() and d.name.startswith("ds_")]
            if not ds_dirs:
                print("❌ ไม่พบ dataset — รัน --mode download ก่อน")
                sys.exit(1)
            dataset_yaml = merge_datasets(ds_dirs)
        else:
            dataset_yaml = merge_datasets([dataset_yaml])

    # ── Augment ──────────────────────────────────────────────
    if args.mode in ("augment", "all") and not args.no_aug:
        if dataset_yaml:
            augment_dataset(dataset_yaml, args.augment_x)

    # ── Stats ────────────────────────────────────────────────
    if dataset_yaml:
        show_dataset_stats(dataset_yaml)

    # ── Train ────────────────────────────────────────────────
    if args.mode in ("train", "all"):
        if not dataset_yaml:
            # ใช้ merged dataset ถ้ามี
            default_yaml = MERGED_DIR / "dataset.yaml"
            if default_yaml.exists():
                dataset_yaml = str(default_yaml)
            else:
                print("❌ ระบุ --dataset <path/to/dataset.yaml>")
                sys.exit(1)

        model_path = train(
            dataset_yaml  = dataset_yaml,
            base_model    = args.model,
            epochs        = args.epochs,
            batch         = args.batch,
            imgsz         = args.imgsz,
            device        = args.device,
            resume        = args.resume,
            pretrained_pt = args.pretrained,
        )
        print(f"\n🎉 Training complete! Model: {model_path}")

    # ── Validate ─────────────────────────────────────────────
    if args.mode == "validate":
        if not dataset_yaml:
            default_yaml = MERGED_DIR / "dataset.yaml"
            dataset_yaml = str(default_yaml) if default_yaml.exists() else None
        if dataset_yaml:
            validate(dataset_yaml)
        else:
            print("❌ ระบุ --dataset")

    # ── Export ───────────────────────────────────────────────
    if args.mode == "export":
        export_model(formats=args.export_format)

    # ── Live Test ────────────────────────────────────────────
    if args.mode == "test":
        test_live(source=args.source)


if __name__ == "__main__":
    main()
