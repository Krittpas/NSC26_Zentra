#!/usr/bin/env python3.11
"""
train_ppe_local.py — ZENTRA PPE Training (Windows 11 + NVIDIA GPU)
==================================================================
วิธีใช้:

  # ตรวจสอบระบบก่อน
  python train_ppe_local.py --check

  # รันทุกขั้นตอนอัตโนมัติ (แนะนำ)
  python train_ppe_local.py --mode all

  # ทีละขั้น
  python train_ppe_local.py --mode download
  python train_ppe_local.py --mode merge
  python train_ppe_local.py --mode augment
  python train_ppe_local.py --mode train
  python train_ppe_local.py --mode validate
  python train_ppe_local.py --mode export --formats onnx
  python train_ppe_local.py --mode test --source 1

  # ตัวเลือกเพิ่มเติม
  python train_ppe_local.py --mode all --epochs 150 --batch 16
  python train_ppe_local.py --mode train --resume
  python train_ppe_local.py --mode continue --extra-epochs 50
==================================================================
"""

import os, sys, json, yaml, shutil, random, time, argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ================================================================
# CONFIG
# ================================================================
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY",   "8xTIheqbzg4mkSLOdFe6")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "pholawats-workspace")

# Training hyperparameters
CFG = {
    "base_model":     "yolov8m.pt",   # n=เร็วสุด s=กลาง m=แนะนำ l=แม่น x=ช้า/แม่นสุด
    "epochs":         100,
    "batch":          16,             # VRAM 8GB → 16 | 6GB → 8 | 4GB → 4
    "imgsz":          640,
    "device":         "0",            # "0"=GPU, "cpu", "0,1"=Multi-GPU
    "optimizer":      "AdamW",
    "lr0":            0.001,
    "lrf":            0.01,
    "warmup_epochs":  5,
    "patience":       30,
    "workers":        8,              # Windows: ลดเหลือ 0 ถ้า DataLoader error
    "cache":          False,          # True ถ้า RAM ≥ 32GB
    "aug_multiplier": 2,
    "val_split":      0.15,
    "target_map50":   0.85,           # เป้าหมาย mAP50
}

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RUNS_DIR   = BASE_DIR / "runs" / "ppe"
MERGED_DIR = DATA_DIR / "merged_ppe"

for _d in [DATA_DIR, MODELS_DIR, RUNS_DIR, MERGED_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ================================================================
# ZENTRA CLASS DEFINITIONS
# ================================================================
ZENTRA_CLASSES = [
    "helmet",           # 0 ✅
    "no_helmet",        # 1 ❌
    "vest",             # 2 ✅
    "no_vest",          # 3 ❌
    "gloves",           # 4 ✅
    "no_gloves",        # 5 ❌
    "goggles",          # 6 ✅
    "no_goggles",       # 7 ❌
    "safety_boots",     # 8 ✅
    "no_safety_boots",  # 9 ❌
    "person",           # 10 👤
]
NC      = len(ZENTRA_CLASSES)
CLS_MAP = {n: i for i, n in enumerate(ZENTRA_CLASSES)}

# แมปชื่อ class จากหลาย dataset → ZENTRA (ครอบคลุมหลาย variant)
ALIAS: dict[str, str] = {
    # helmet
    "hard-hat":"helmet","hardhat":"helmet","hard_hat":"helmet","helmet":"helmet",
    "head":"helmet","safety helmet":"helmet","protective helmet":"helmet",
    # no_helmet
    "no-hardhat":"no_helmet","no_hardhat":"no_helmet","no-helmet":"no_helmet",
    "no_helmet":"no_helmet","no helmet":"no_helmet","without helmet":"no_helmet",
    "w/o helmet":"no_helmet",
    # vest
    "safety-vest":"vest","safety_vest":"vest","vest":"vest","reflective-vest":"vest",
    "hi-vis":"vest","hi_vis":"vest","safety vest":"vest","high visibility":"vest",
    "visibility vest":"vest","reflective vest":"vest",
    # no_vest
    "no-vest":"no_vest","no_vest":"no_vest","no vest":"no_vest",
    "without vest":"no_vest","w/o vest":"no_vest",
    # gloves
    "gloves":"gloves","safety-gloves":"gloves","safety_gloves":"gloves",
    "glove":"gloves","protective gloves":"gloves",
    # no_gloves
    "no-gloves":"no_gloves","no_gloves":"no_gloves","no gloves":"no_gloves",
    "without gloves":"no_gloves","w/o gloves":"no_gloves",
    # goggles
    "goggles":"goggles","safety-glasses":"goggles","safety_glasses":"goggles",
    "glasses":"goggles","eye-protection":"goggles","eyewear":"goggles",
    "safety goggles":"goggles","protective glasses":"goggles",
    # no_goggles
    "no-goggles":"no_goggles","no_goggles":"no_goggles","no goggles":"no_goggles",
    "no-glasses":"no_goggles","no glasses":"no_goggles","without glasses":"no_goggles",
    # boots
    "safety-boots":"safety_boots","safety_boots":"safety_boots","boots":"safety_boots",
    "safety boots":"safety_boots","steel-toed boots":"safety_boots",
    # no_boots
    "no-boots":"no_safety_boots","no_boots":"no_safety_boots",
    "no boots":"no_safety_boots","without boots":"no_safety_boots",
    # person
    "person":"person","worker":"person","human":"person",
    "people":"person","man":"person","woman":"person","employee":"person",
}

def remap(src_id: int, src_names: list) -> int | None:
    if src_id >= len(src_names):
        return None
    raw    = src_names[src_id].lower().strip()
    target = ALIAS.get(raw)
    if target:
        return CLS_MAP.get(target)
    # partial match
    for alias, t in ALIAS.items():
        if alias in raw or raw in alias:
            return CLS_MAP.get(t)
    return None


# ================================================================
# SYSTEM CHECK
# ================================================================
def check_system():
    print("\n" + "═"*58)
    print("  ZENTRA — System Check (Windows 11)")
    print("═"*58)

    # Python
    pv = sys.version_info
    print(f"  Python   : {pv.major}.{pv.minor}.{pv.micro}  "
          f"{'✅' if pv >= (3,10) else '❌ ต้องการ 3.10+'}")

    # GPU
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  PyTorch  : {torch.__version__}  {'✅ CUDA' if cuda else '⚠️  CPU only'}")
        if cuda:
            for i in range(torch.cuda.device_count()):
                p    = torch.cuda.get_device_properties(i)
                vram = p.total_memory / 1e9
                rb   = 32 if vram>=10 else 16 if vram>=8 else 8 if vram>=6 else 4
                print(f"  GPU {i}    : {p.name}  VRAM={vram:.1f}GB  → batch={rb}")
        else:
            print("  ⚠️  ไม่พบ GPU — ตรวจสอบ CUDA driver")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("  PyTorch  : ❌  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # Packages
    for pkg in ["ultralytics", "roboflow", "cv2"]:
        try:
            m = __import__(pkg)
            v = getattr(m, "__version__", "?")
            print(f"  {pkg:<12}: {v}  ✅")
        except ImportError:
            print(f"  {pkg:<12}: ❌  pip install {pkg}")

    # Disk
    free = shutil.disk_usage(BASE_DIR).free / 1e9
    print(f"  Disk free: {free:.1f} GB  {'✅' if free>10 else '⚠️ ควรมี >10GB'}")
    print("═"*58 + "\n")


# ================================================================
# STEP 1: DOWNLOAD DATASETS
# ================================================================
DATASETS = [
    # dataset ของทีม
    {"ws": ROBOFLOW_WORKSPACE,              "proj": "ppe-cpxsz",                    "ver": 2, "dest": "ds_main",        "note": "ZENTRA main"},
    # Public datasets (Roboflow Universe)
    {"ws": "roboflow-universe-projects",    "proj": "construction-site-safety",      "ver": 1, "dest": "ds_construction","note": "~5k construction PPE"},
    {"ws": "joseph-nelson",                 "proj": "hard-hat-workers",              "ver": 2, "dest": "ds_hardhat",     "note": "~7k hard hat"},
    {"ws": "roboflow-universe-projects",    "proj": "ppe-detection-ljb7d",           "ver": 4, "dest": "ds_ppe2",        "note": "~3k PPE"},
    {"ws": "roboflow-universe-projects",    "proj": "safety-equipment-detection-vwckw","ver": 1,"dest":"ds_safety_eq",   "note": "~2k safety equipment"},
]


def download_datasets() -> list[str]:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("❌  pip install roboflow")

    rf         = Roboflow(api_key=ROBOFLOW_API_KEY)
    downloaded = []

    print(f"\n📥 Downloading {len(DATASETS)} datasets...")
    for ds in DATASETS:
        dest    = DATA_DIR / ds["dest"]
        existing = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
        if dest.exists() and len(existing) > 20:
            print(f"  ✅ {ds['proj']:<40} ({len(existing):,} images already)")
            downloaded.append(str(dest))
            continue
        print(f"  📥 {ds['proj']:<40} [{ds['note']}]", end=" ", flush=True)
        try:
            proj = rf.workspace(ds["ws"]).project(ds["proj"])
            proj.version(ds["ver"]).download("yolov8", location=str(dest), overwrite=True)
            n = len(list(dest.rglob("*.jpg")) + list(dest.rglob("*.png")))
            print(f"→ {n:,} images ✅")
            downloaded.append(str(dest))
        except Exception as e:
            print(f"→ ข้าม ({e})")

    total = sum(
        len(list((DATA_DIR/ds["dest"]).rglob("*.jpg")))
        for ds in DATASETS if (DATA_DIR/ds["dest"]).exists()
    )
    print(f"\n  📦 {len(downloaded)} datasets  |  ~{total:,} total images")
    return downloaded


# ================================================================
# STEP 2: MERGE + REMAP
# ================================================================
def _get_names(ds_path: str) -> list[str]:
    for name in ["data.yaml", "dataset.yaml"]:
        y = Path(ds_path) / name
        if y.exists():
            return yaml.safe_load(open(y)).get("names", [])
    for y in Path(ds_path).rglob("data.yaml"):
        return yaml.safe_load(open(y)).get("names", [])
    return []


def _find_label(img_path: Path) -> Path | None:
    lp = Path(
        str(img_path).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep, 1)
    ).with_suffix(".txt")
    if lp.exists(): return lp
    lp = img_path.with_suffix(".txt")
    return lp if lp.exists() else None


def merge_datasets(ds_paths: list[str]) -> str:
    print(f"\n🔀 Merging {len(ds_paths)} datasets → ZENTRA classes...")

    for split in ["train", "val"]:
        (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    cls_stats = {c: 0 for c in ZENTRA_CLASSES}
    total = skipped = 0
    random.seed(42)

    for ds_path in ds_paths:
        src_names = _get_names(ds_path)
        img_files = (list(Path(ds_path).rglob("*.jpg")) +
                     list(Path(ds_path).rglob("*.png")))
        random.shuffle(img_files)
        ds_name = Path(ds_path).name
        print(f"  📁 {ds_name:<25} {len(img_files):>5} imgs | classes:{src_names[:4]}")

        for img_path in img_files:
            lbl_path = _find_label(img_path)
            if not lbl_path: skipped += 1; continue

            new_lines = []
            for line in lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                new_id = remap(int(parts[0]), src_names)
                if new_id is None: continue
                new_lines.append(f"{new_id} {' '.join(parts[1:5])}")
                cls_stats[ZENTRA_CLASSES[new_id]] += 1

            if not new_lines: skipped += 1; continue

            img = cv2.imread(str(img_path))
            if img is None: skipped += 1; continue

            split = "val" if random.random() < CFG["val_split"] else "train"
            stem  = f"{ds_name}_{total:07d}"
            cv2.imwrite(
                str(MERGED_DIR / "images" / split / f"{stem}.jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            (MERGED_DIR / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(new_lines), encoding="utf-8"
            )
            total += 1

    # dataset.yaml
    yaml_path = MERGED_DIR / "dataset.yaml"
    yaml.dump({"path": str(MERGED_DIR.resolve()), "train": "images/train", "val": "images/val",
               "nc": NC, "names": ZENTRA_CLASSES}, open(yaml_path, "w"), allow_unicode=True)

    n_train = len(list((MERGED_DIR/"images"/"train").glob("*.jpg")))
    n_val   = len(list((MERGED_DIR/"images"/"val").glob("*.jpg")))
    print(f"\n  ✅ train={n_train:,}  val={n_val:,}  skipped={skipped:,}")
    print(f"  Class distribution:")
    for cls, cnt in sorted(cls_stats.items(), key=lambda x:-x[1]):
        if cnt > 0:
            icon = "✅" if not cls.startswith("no_") and cls!="person" else ("❌" if cls.startswith("no_") else "👤")
            bar  = "█" * min(cnt//60, 32)
            print(f"    {icon} {cls:<22} {cnt:>6,}  {bar}")
    return str(yaml_path)


# ================================================================
# STEP 3: AUGMENTATION (9 techniques)
# ================================================================
def _augment_one(img: np.ndarray, labels: str, seed: int) -> tuple[np.ndarray, str]:
    rng = random.Random(seed)
    np.random.seed(seed % (2**31))
    h, w  = img.shape[:2]
    aug   = img.copy()
    lines = [l.strip() for l in labels.splitlines() if l.strip()]
    new_lines = list(lines)

    # 1. Horizontal flip
    if rng.random() < 0.50:
        aug = cv2.flip(aug, 1)
        new_lines = []
        for line in lines:
            p = line.split()
            if len(p) >= 5: p[1] = f"{1.0 - float(p[1]):.6f}"
            new_lines.append(" ".join(p))
        lines = new_lines[:]

    # 2. HSV jitter (แสงโรงงาน)
    if rng.random() < 0.85:
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + rng.uniform(-22, 22),  0, 179)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * rng.uniform(0.45, 1.55), 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * rng.uniform(0.40, 1.60), 0, 255)
        aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Brightness / Contrast
    if rng.random() < 0.60:
        alpha = rng.uniform(0.55, 1.45)
        beta  = rng.uniform(-45, 45)
        aug   = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 4. CLAHE (โรงงานแสงไม่สม่ำเสมอ)
    if rng.random() < 0.35:
        lab = cv2.cvtColor(aug, cv2.COLOR_BGR2LAB)
        cl  = cv2.createCLAHE(clipLimit=rng.uniform(1.5, 4.0), tileGridSize=(8,8))
        lab[:,:,0] = cl.apply(lab[:,:,0])
        aug = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 5. Gaussian blur (simulate กล้อง out-of-focus)
    if rng.random() < 0.22:
        k   = rng.choice([3, 5, 7])
        aug = cv2.GaussianBlur(aug, (k, k), 0)

    # 6. Gaussian noise
    if rng.random() < 0.20:
        noise = np.random.normal(0, rng.uniform(5, 25), aug.shape).astype(np.int16)
        aug   = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 7. Salt-and-pepper (sensor noise)
    if rng.random() < 0.15:
        amount = rng.uniform(0.001, 0.006)
        n      = int(amount * aug.size)
        coords = [np.random.randint(0, d-1, n) for d in aug.shape[:2]]
        aug[coords[0], coords[1]] = 255
        coords = [np.random.randint(0, d-1, n) for d in aug.shape[:2]]
        aug[coords[0], coords[1]] = 0

    # 8. Rotation ±12°
    if rng.random() < 0.30:
        angle = rng.uniform(-12, 12)
        M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug   = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 9. Random crop / zoom-in (5-15%)
    if rng.random() < 0.30:
        pct  = rng.uniform(0.05, 0.15)
        mx   = int(w * pct); my = int(h * pct)
        if mx > 2 and my > 2:
            aug = aug[my:h-my, mx:w-mx]
            aug = cv2.resize(aug, (w, h))
            rx, ry = w / (w-2*mx), h / (h-2*my)
            ox, oy = mx/w, my/h
            adj = []
            for line in new_lines:
                p = line.split()
                if len(p) >= 5:
                    cx = (float(p[1]) - ox) * rx
                    cy = (float(p[2]) - oy) * ry
                    bw = float(p[3]) * rx
                    bh = float(p[4]) * ry
                    if 0.01 < cx < 0.99 and 0.01 < cy < 0.99:
                        adj.append(f"{p[0]} {max(0.01,min(0.99,cx)):.6f} {max(0.01,min(0.99,cy)):.6f} {min(bw,1.0):.6f} {min(bh,1.0):.6f}")
            new_lines = adj

    return aug, "\n".join(new_lines)


def augment_dataset(dataset_yaml: str):
    with open(dataset_yaml) as f:
        ds_cfg = yaml.safe_load(f)
    img_dir = Path(ds_cfg["path"]) / "images" / "train"
    lbl_dir = Path(ds_cfg["path"]) / "labels" / "train"
    orig    = sorted(img_dir.glob("*.jpg"))
    mult    = CFG["aug_multiplier"]

    print(f"\n🎨 Augmenting {len(orig):,} images × {mult}  (9 techniques)...")
    print(f"   Expected total: {len(orig)*(mult+1):,} images")

    added = 0; t0 = time.time()
    for i, img_path in enumerate(orig):
        img = cv2.imread(str(img_path))
        if img is None: continue
        lbl    = lbl_dir / img_path.with_suffix(".txt").name
        labels = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
        for k in range(mult):
            aug_img, aug_lbl = _augment_one(img, labels, i*100+k)
            stem = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(img_dir/f"{stem}.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            (lbl_dir/f"{stem}.txt").write_text(aug_lbl, encoding="utf-8")
            added += 1
        if (i+1) % 300 == 0:
            elapsed = time.time()-t0
            eta     = (len(orig)-i-1) / max((i+1)/elapsed, 0.01)
            print(f"   {i+1:>5}/{len(orig)}  ETA {eta/60:.1f}min")

    total = len(list(img_dir.glob("*.jpg")))
    print(f"  ✅ +{added:,} augmented  |  Total train: {total:,}")


# ================================================================
# STEP 4: TRAIN
# ================================================================
def train(
    dataset_yaml: str,
    epochs:       int  = None,
    batch:        int  = None,
    device:       str  = None,
    resume:       bool = False,
    pretrained:   str  = None,
) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    epochs = epochs or CFG["epochs"]
    batch  = batch  or CFG["batch"]
    device = device or CFG["device"]

    # เลือก weights
    if pretrained and Path(pretrained).exists():
        weights = pretrained; print(f"🔄 Fine-tune: {weights}")
    elif (MODELS_DIR/"ppe_finetuned.pt").exists() and resume:
        weights = str(MODELS_DIR/"ppe_finetuned.pt"); print(f"🔄 Resume: {weights}")
    else:
        weights = CFG["base_model"]; print(f"🆕 Base: {weights}")

    model    = YOLO(weights)
    run_name = f"zentra_ppe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'═'*58}")
    print(f"  🚀 ZENTRA PPE Training — Windows 11 GPU")
    print(f"  Dataset    : {dataset_yaml}")
    print(f"  Weights    : {weights}")
    print(f"  Epochs     : {epochs}  |  Batch: {batch}  |  Device: {device}")
    print(f"  Classes    : {NC}  {ZENTRA_CLASSES[:5]}...")
    print(f"{'═'*58}\n")

    results = model.train(
        data    = dataset_yaml,
        epochs  = epochs,
        batch   = batch,
        imgsz   = CFG["imgsz"],
        device  = device,
        project = str(RUNS_DIR),
        name    = run_name,

        # Optimizer
        optimizer      = CFG["optimizer"],
        lr0            = CFG["lr0"],
        lrf            = CFG["lrf"],
        momentum       = 0.937,
        weight_decay   = 0.0005,
        warmup_epochs  = CFG["warmup_epochs"],
        warmup_bias_lr = 0.1,

        # Built-in Augmentation (YOLOv8)
        augment      = True,
        mosaic       = 1.0,       # ผสม 4 ภาพ
        mixup        = 0.15,
        copy_paste   = 0.10,
        close_mosaic = 15,        # ปิด mosaic 15 epochs สุดท้าย
        fliplr       = 0.50,
        flipud       = 0.0,
        hsv_h        = 0.015,
        hsv_s        = 0.70,
        hsv_v        = 0.40,
        degrees      = 5.0,
        translate    = 0.10,
        scale        = 0.60,
        shear        = 2.0,
        perspective  = 0.0005,
        erasing      = 0.40,      # simulate occlusion

        # Loss weights
        box = 7.5,
        cls = 0.5,
        dfl = 1.5,

        # Control
        patience    = CFG["patience"],
        save        = True,
        save_period = 10,
        val         = True,
        plots       = True,
        verbose     = True,
        workers     = CFG["workers"],
        cache       = CFG["cache"],
        amp         = True,       # Mixed precision → ~40% faster
        seed        = 42,
        deterministic = True,
        resume      = resume,
    )

    # หา best.pt
    best_candidates = sorted(RUNS_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if not best_candidates:
        raise FileNotFoundError("ไม่พบ best.pt")

    best_pt = best_candidates[-1]
    target  = MODELS_DIR / "ppe_finetuned.pt"
    shutil.copy2(best_pt, target)

    # บันทึก log
    log = {"run": run_name, "dataset": dataset_yaml, "epochs": epochs,
           "model": str(target), "timestamp": datetime.now().isoformat()}
    (BASE_DIR/"logs").mkdir(exist_ok=True)
    (BASE_DIR/"logs"/f"train_{run_name}.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False))

    print(f"\n{'═'*58}")
    print(f"  ✅ Training complete!")
    print(f"  📁 Model → {target}")
    print(f"{'═'*58}\n")
    return str(target)


# ================================================================
# STEP 5: VALIDATE
# ================================================================
def validate(dataset_yaml: str, model_path: str = None) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    m_path = model_path or str(MODELS_DIR/"ppe_finetuned.pt")
    if not Path(m_path).exists():
        print(f"❌  ไม่พบ model: {m_path}"); return {}

    print(f"\n🧪 Validating: {Path(m_path).name}")
    model   = YOLO(m_path)
    metrics = model.val(data=dataset_yaml, imgsz=640, verbose=True, plots=True)

    map50 = float(metrics.box.map50)
    print(f"\n{'═'*55}")
    print(f"  📊 Validation — ZENTRA PPE")
    print(f"{'═'*55}")
    print(f"  mAP50     : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print(f"{'─'*55}")

    target = CFG["target_map50"]
    if map50 >= target:
        print(f"  🎉 ผ่านเป้าหมาย mAP50 ≥ {target*100:.0f}%  → พร้อม deploy!")
    else:
        print(f"  ⚠️  ยังต่ำกว่าเป้า {(target-map50)*100:.1f}%  → รัน --mode continue")

    # Per-class AP
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        print(f"\n  Per-Class AP50:")
        for i, ci in enumerate(metrics.box.ap_class_index):
            if ci < NC and i < len(metrics.box.ap50):
                ap   = float(metrics.box.ap50[i])
                bar  = "█" * int(ap*24)
                icon = "✅" if ap>=0.80 else ("⚠️" if ap>=0.60 else "❌")
                print(f"    {icon} {ZENTRA_CLASSES[ci]:<22} {ap:.3f}  {bar}")
    print("═"*55 + "\n")
    return {"mAP50": map50, "mAP50-95": metrics.box.map,
            "precision": metrics.box.mp, "recall": metrics.box.mr}


# ================================================================
# STEP 6: CONTINUE TRAINING
# ================================================================
def continue_training(dataset_yaml: str, extra: int = 50, device: str = None):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    device = device or CFG["device"]
    m_path = MODELS_DIR / "ppe_finetuned.pt"
    if not m_path.exists():
        print("❌  ไม่พบ ppe_finetuned.pt"); return

    print(f"\n🔄 Continue training +{extra} epochs (lr ลดลง)...")
    model = YOLO(str(m_path))
    rname = f"zentra_ppe_cont_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model.train(
        data=dataset_yaml, epochs=extra, batch=CFG["batch"],
        imgsz=CFG["imgsz"], device=device, project=str(RUNS_DIR), name=rname,
        optimizer="AdamW", lr0=0.0003, lrf=0.01, warmup_epochs=2,
        mosaic=0.5, mixup=0.05, close_mosaic=5, erasing=0.3,
        patience=20, amp=True, workers=CFG["workers"], verbose=True,
    )
    bests = sorted(RUNS_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if bests:
        shutil.copy2(bests[-1], MODELS_DIR/"ppe_finetuned.pt")
        print(f"✅ Model updated → {MODELS_DIR/'ppe_finetuned.pt'}")


# ================================================================
# STEP 7: EXPORT
# ================================================================
def export_model(formats: list = None, model_path: str = None):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    formats = formats or ["onnx"]
    m_path  = model_path or str(MODELS_DIR/"ppe_finetuned.pt")
    if not Path(m_path).exists():
        print(f"❌  ไม่พบ: {m_path}"); return

    model = YOLO(m_path)
    print(f"\n📦 Exporting: {Path(m_path).name}")
    for fmt in formats:
        print(f"  → {fmt.upper()}...", end=" ", flush=True)
        try:
            kw: dict = {"format": fmt, "imgsz": 640}
            if fmt == "onnx":    kw.update({"simplify": True, "opset": 17})
            elif fmt == "engine": kw.update({"half": True, "device": 0})
            out = model.export(**kw)
            dest = MODELS_DIR / Path(str(out)).name
            shutil.copy2(str(out), dest)
            print(f"✅  {dest}")
        except Exception as e:
            print(f"❌  {e}")


# ================================================================
# STEP 8: LIVE TEST
# ================================================================
def test_live(source: str = "0", model_path: str = None, conf: float = 0.45):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("❌  pip install ultralytics")

    m_path = model_path or str(MODELS_DIR/"ppe_finetuned.pt")
    if not Path(m_path).exists():
        print(f"❌  ไม่พบ: {m_path}"); return

    src   = int(source) if source.isdigit() else source
    model = YOLO(m_path)
    cap   = cv2.VideoCapture(src, cv2.CAP_DSHOW if isinstance(src, int) else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"❌  เปิด source ไม่ได้: {source}"); return

    print(f"\n🎥 Live test | Model: {Path(m_path).name} | conf={conf} | Q=quit\n")
    while True:
        ret, frame = cap.read()
        if not ret: break

        results   = model.predict(frame, conf=conf, iou=0.45, verbose=False)
        annotated = results[0].plot()

        detected   = [ZENTRA_CLASSES[int(c)] for c in results[0].boxes.cls.cpu().numpy()
                      if int(c) < NC] if len(results[0].boxes) > 0 else []
        violations = [c for c in detected if c.startswith("no_")]
        status     = f"⚠ VIOLATION: {', '.join(set(violations))}" if violations else "✅ Compliant"
        scolor     = (0, 0, 220) if violations else (0, 200, 50)
        cv2.putText(annotated, status, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, scolor, 2, cv2.LINE_AA)
        cv2.imshow("ZENTRA PPE Live Test — Q=Quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()


# ================================================================
# DATASET STATS
# ================================================================
def show_stats(dataset_yaml: str):
    with open(dataset_yaml) as f:
        ds_cfg = yaml.safe_load(f)
    base = Path(ds_cfg["path"])
    print(f"\n📊 Dataset: {dataset_yaml}")
    print("─"*55)
    cls_cnt = {c: 0 for c in ZENTRA_CLASSES}
    for split in ["train", "val"]:
        n_img = len(list((base/"images"/split).glob("*.jpg")))
        n_ann = 0
        for lf in (base/"labels"/split).glob("*.txt"):
            for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = line.strip().split()
                if p:
                    cid = int(p[0])
                    if cid < NC:
                        cls_cnt[ZENTRA_CLASSES[cid]] += 1
                        n_ann += 1
        print(f"  {split:<8}: {n_img:>6,} images | {n_ann:>7,} annotations")
    total_ann = sum(cls_cnt.values())
    print(f"\n  Class breakdown (total annotations={total_ann:,}):")
    for cls, cnt in sorted(cls_cnt.items(), key=lambda x:-x[1]):
        if cnt > 0:
            pct  = cnt/total_ann*100 if total_ann>0 else 0
            bar  = "█"*min(cnt//70, 28)
            icon = "✅" if not cls.startswith("no_") and cls!="person" else ("❌" if cls.startswith("no_") else "👤")
            print(f"    {icon} {cls:<22} {cnt:>6,} ({pct:4.1f}%) {bar}")
    print("─"*55)


def get_dataset_yaml() -> str:
    y = MERGED_DIR / "dataset.yaml"
    if y.exists(): return str(y)
    for y2 in DATA_DIR.rglob("dataset.yaml"):
        return str(y2)
    return ""


# ================================================================
# CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser(
        description="ZENTRA PPE Training — Windows 11",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--check",        action="store_true")
    ap.add_argument("--mode",         default="all",
        choices=["download","merge","augment","train","all",
                 "validate","continue","export","test"])
    ap.add_argument("--dataset",      default=None)
    ap.add_argument("--epochs",       type=int, default=None)
    ap.add_argument("--extra-epochs", type=int, default=50)
    ap.add_argument("--batch",        type=int, default=None)
    ap.add_argument("--device",       default=None, help="0=GPU  cpu=CPU  0,1=Multi")
    ap.add_argument("--resume",       action="store_true")
    ap.add_argument("--pretrained",   default=None)
    ap.add_argument("--no-aug",       action="store_true")
    ap.add_argument("--aug-x",        type=int, default=None)
    ap.add_argument("--formats",      nargs="+", default=["onnx"],
        choices=["onnx","engine","tflite"])
    ap.add_argument("--source",       default="0")
    ap.add_argument("--conf",         type=float, default=0.45)
    ap.add_argument("--model",        default=None)
    args = ap.parse_args()

    if args.aug_x:
        CFG["aug_multiplier"] = args.aug_x

    if args.check:
        check_system(); return

    check_system()

    dataset_yaml = args.dataset

    # download
    if args.mode in ("download", "all"):
        paths = download_datasets()
    else:
        paths = [str(d) for d in DATA_DIR.iterdir()
                 if d.is_dir() and d.name.startswith("ds_")]

    # merge
    if args.mode in ("merge", "all"):
        if not paths: sys.exit("❌  รัน --mode download ก่อน")
        dataset_yaml = merge_datasets(paths)
    elif not dataset_yaml:
        dataset_yaml = get_dataset_yaml()

    # augment
    if args.mode in ("augment","all") and not args.no_aug and dataset_yaml:
        augment_dataset(dataset_yaml)

    # stats
    if dataset_yaml and Path(dataset_yaml).exists():
        show_stats(dataset_yaml)

    # train
    if args.mode in ("train","all"):
        if not dataset_yaml or not Path(dataset_yaml).exists():
            sys.exit("❌  ไม่พบ dataset.yaml")
        train(dataset_yaml, args.epochs, args.batch, args.device,
              args.resume, args.pretrained)

    # continue
    if args.mode == "continue":
        if not dataset_yaml: dataset_yaml = get_dataset_yaml()
        if dataset_yaml: continue_training(dataset_yaml, args.extra_epochs, args.device)
        else: sys.exit("❌  ระบุ --dataset")

    # validate
    if args.mode == "validate":
        if not dataset_yaml: dataset_yaml = get_dataset_yaml()
        if dataset_yaml: validate(dataset_yaml, args.model)
        else: sys.exit("❌  ระบุ --dataset")

    # export
    if args.mode == "export":
        export_model(args.formats, args.model)

    # test
    if args.mode == "test":
        test_live(args.source, args.model, args.conf)


if __name__ == "__main__":
    main()
