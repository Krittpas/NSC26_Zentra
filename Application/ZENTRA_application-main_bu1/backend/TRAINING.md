# ZENTRA — Training Guide (เทรนให้แม่นยำขึ้น)

> **กฎเหล็ก:** ความแม่นยำมาจาก **ข้อมูลจริง + label ที่ถูกต้อง** ไม่ใช่จำนวน epoch
> โมเดลเริ่มต้น (`ppe-cpxsz/2`, `fall-detection-ovjqo/5`) เป็นโมเดลสาธารณะ —
> เทรนกับข้อมูลโรงงานของคุณเองเท่านั้นจึงจะแม่นจริงในสภาพแวดล้อมจริง

GPU ในเครื่อง: **RTX 3050 (4 GB)** → ใช้ `yolov8s.pt` + `batch=4` (ตั้งใน `.env` แล้ว)

---

## ภาพรวม 5 ขั้น

```
1. COLLECT   รันแอป → ระบบเก็บภาพ+label อัตโนมัติลง data/collected/
2. UPLOAD    ส่งภาพขึ้น Roboflow project
3. LABEL ⭐  แก้/ตีกรอบให้ถูกต้องใน Roboflow (ขั้นที่ทำให้แม่นจริง)
4. TRAIN     ดึง dataset ที่ label แล้วมา fine-tune YOLOv8 บน GPU
5. DEPLOY    เปิด USE_LOCAL_MODEL=true → แอปใช้โมเดลใหม่
```

---

## ขั้น 1 — COLLECT (เก็บข้อมูล)

แค่ใช้งานแอปตามปกติ ระบบเก็บให้อัตโนมัติ ([utils/collector.py](utils/collector.py)):

```
data/collected/
├── ppe_violations/   ← ทุกครั้งที่เจอคนไม่ใส่ PPE (ภาพ + .txt YOLO label)
├── zone_intrusions/  ← คนเข้าเขตอันตราย
├── fall_events/      ← การล้ม
└── normal/           ← เฟรมปกติ (ทุก 300 เฟรม)
```

**เคล็ดความแม่นยำ:** เก็บให้ **หลากหลาย** — แสงเช้า/บ่าย/กลางคืน, มุมกล้องจริง,
คนใส่/ไม่ใส่ PPE, ระยะใกล้/ไกล ยิ่งหลากหลายโมเดลยิ่งทนสภาพจริง
เป้าหมายขั้นต่ำ: **~300–500 ภาพต่อคลาส** ที่ label ถูก

## ขั้น 2 — UPLOAD (ส่งขึ้น Roboflow)

```powershell
cd c:\ZENTRA\ZENTRA
python -m training.upload --task ppe          # อัปโหลด data/collected/ppe_violations
```
(หรือใช้เว็บ Roboflow ลากไฟล์จาก `data/collected/` ขึ้นไปตรงๆ ก็ได้)

## ขั้น 3 — LABEL ⭐ (สำคัญที่สุด)

ใน Roboflow:
1. เปิด project → **Annotate**
2. ตรวจทุกกรอบที่ระบบเดามา — **ลบกรอบผิด, เพิ่มกรอบที่ขาด, แก้คลาสให้ถูก**
3. ใช้ **Model-Assisted Labeling** ของ Roboflow ช่วยตีกรอบเร็วขึ้นได้
4. **Generate Version** → เลือก augmentation (flip, brightness, blur) → สร้าง dataset

> ขั้นนี้คือหัวใจ — โมเดลจะแม่นได้เท่าที่ label ถูกเท่านั้น

## ขั้น 4 — TRAIN (เทรนบน GPU)

**A) ดาวน์โหลด + เทรนจาก Roboflow (แนะนำ):**
```powershell
python -m training.trainer --task ppe --project zentra-ppe --export
```

**B) เทรนจาก zip ที่ export มาเอง:**
```powershell
python -m training.trainer --task ppe --zip path\to\dataset.zip --export
```

**C) เทรนจากข้อมูล local ที่ label แล้ว (ไม่ผ่าน Roboflow):**
```powershell
python -m training.trainer --task ppe --export
```

ผลลัพธ์:
- โมเดล → `models/ppe_finetuned.pt`
- รายงาน mAP50 / precision / recall พิมพ์ออกมา + กราฟใน `models/ppe_<timestamp>/`
- export ONNX (สำหรับ deploy เร็วขึ้น) ถ้าใส่ `--export`

> ถ้าเจอ **CUDA out of memory** → ลด `TRAIN_BATCH_SIZE` ใน `.env` เป็น 2 หรือ
> เปลี่ยน `YOLO_BASE_MODEL=yolov8n.pt`

## ขั้น 5 — DEPLOY (ใช้โมเดลใหม่)

ใน `.env`:
```
USE_LOCAL_MODEL=true
```
แอปจะโหลด `models/ppe_finetuned.pt` แทนโมเดลสาธารณะอัตโนมัติ
([main.py](main.py) / [pipeline](../ZENTRA_application/pipeline/pipeline.py) `_make_client`)

---

## วัดว่าแม่นขึ้นจริงไหม

ดูค่า **mAP50** จากขั้น train (ยิ่งสูงยิ่งดี, >0.85 = ดีมากตาม proposal):
```powershell
python -m training.trainer --task ppe         # จะ validate ให้ท้ายการเทรน
```
ถ้าค่ายังต่ำ → กลับไปขั้น 1–3: เก็บ + label เพิ่ม (เกือบทุกครั้งปัญหาอยู่ที่ข้อมูล ไม่ใช่ epoch)

## วงจรพัฒนาต่อเนื่อง (active learning)

รันแอป → เก็บเคสที่โมเดลพลาด → label เคสยากเหล่านั้น → เทรนซ้ำ → ทำซ้ำ
ทุกรอบโมเดลจะแม่นขึ้นกับโรงงานของคุณโดยเฉพาะ
