# ZENTRA — Training Guide (คู่มือ Fine-tuning)

## ภาพรวม Pipeline

```
กล้อง IP
   ↓ RTSP Stream
Auto Collect (utils/collector.py)
   ↓ บันทึก frame + YOLO annotation อัตโนมัติ
data/collected/
├── ppe_violations/    ← เก็บเมื่อพบ violation
├── zone_intrusions/   ← เก็บเมื่อมีบุกรุก zone
├── fall_events/       ← เก็บเมื่อตรวจพบการล้ม
└── normal/            ← เก็บ normal ทุก 300 frames
   ↓ ผสมกับ Roboflow dataset
DatasetPreparer (training/trainer.py)
   ↓ Train/Val split + Augmentation
YOLOv8 Fine-tuning (ultralytics)
   ↓ best.pt
models/ppe_finetuned.pt
   ↓ ONNX Export + TensorRT INT8
Production Inference
```

---

## วิธีการ Train

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า .env

```bash
cp .env.example .env
# แก้ไข ROBOFLOW_API_KEY, LINE_OA_CHANNEL_ACCESS_TOKEN
```

### 3. รัน ZENTRA เพื่อเก็บ Training Data

```bash
python3.11 main.py
# รอจนเก็บข้อมูลพอ (ดูที่ data/collected/)
# กด S เพื่อดูจำนวน frame ที่เก็บแล้ว
```

### 4. Option A — Train จาก Auto-Collected Data

```bash
python3.11 -m training.trainer --task ppe
python3.11 -m training.trainer --task fall
```

### 5. Option B — Train จาก Roboflow Dataset

```bash
# ดาวน์โหลด dataset จาก Roboflow
python3.11 -m training.trainer --task ppe --project zentra-ppe

# หรือ export zip จาก Roboflow แล้ว train
python3.11 -m training.trainer --task ppe --zip /path/to/dataset.zip
```

### 6. Option C — Train ขณะรัน (กด T ใน main)

ขณะระบบทำงาน กด **T** เพื่อเริ่ม training ใน background thread

### 7. Export ONNX (สำหรับ Edge Deployment)

```bash
python3.11 -m training.trainer --task ppe --export
```

### 8. Upload Training Data ขึ้น Roboflow

```bash
python3.11 -m training.trainer --task ppe --upload
```

---

## Augmentation ที่ใช้

| Technique     | ค่า         | จุดประสงค์                            |
|---------------|-------------|---------------------------------------|
| Mosaic        | 1.0         | ผสม 4 ภาพ เพิ่ม context หลากหลาย     |
| MixUp         | 0.1         | ผสม 2 ภาพ ลด overfit                  |
| Horizontal Flip | 0.5       | เพิ่ม data ซ้าย-ขวา                   |
| HSV (H,S,V)   | ±18°, ±40%  | รองรับแสงหลากหลายในโรงงาน             |
| Scale         | ±50%        | รองรับระยะกล้องต่างๆ                  |
| Rotation      | ±5°         | รองรับมุมกล้องเอียง                   |
| Copy-Paste    | 0.1         | สังเคราะห์ PPE ในตำแหน่งใหม่          |
| Offline Aug   | 2x          | เพิ่มจาก DatasetPreparer.augment()    |

---

## ผลลัพธ์ที่คาดหวัง

| Stage         | mAP50 (PPE) | mAP50 (Fall) |
|---------------|-------------|--------------|
| Base model    | ~70%        | ~65%         |
| After 50 ep   | **≥85%**    | **≥80%**     |
| After 100 ep  | ≥88%        | ≥84%         |

---

## ไฟล์ที่เกี่ยวข้อง

```
zentra/
├── training/
│   └── trainer.py        ← ZENTRATrainer, DatasetPreparer, run_training_pipeline
├── utils/
│   ├── collector.py      ← DataCollector (auto-collect frames)
│   └── tracker.py        ← ByteTracker (multi-object tracking)
├── data/
│   ├── collected/        ← auto-collected frames (input ของ training)
│   ├── train_dataset/    ← dataset ที่เตรียมแล้ว (train/val split)
│   └── dataset.yaml      ← YOLO dataset config
├── models/
│   ├── ppe_finetuned.pt  ← fine-tuned PPE model
│   └── fall_finetuned.pt ← fine-tuned Fall model
└── logs/
    └── training_*.json   ← training history
```
