# ZENTRA — ทำงานต่อบน Mac (เทรนโมเดล PPE)

คู่มือ setup + เทรนโมเดลต่อบนเครื่อง Mac (Apple Silicon) หลังเพิ่ม dataset ใหม่ใน Roboflow

## โครงสร้างโปรเจกต์
| ส่วน | Repo | หน้าที่ |
|---|---|---|
| Backend (repo นี้) | ThePpoon/… | AI, เทรน, config, modules |
| App (เดสก์ท็อป) | `ThePpoon/ZENTRA_application` | UI (PyWebView + FastAPI) |

---

## 1) Setup บน Mac (Apple Silicon)

```bash
git clone <URL-ของ-repo-นี้>
cd ZENTRA
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # torch บน Mac ใช้ MPS (Metal) — ไม่มี CUDA
```

### สร้างไฟล์ `.env` (copy จาก `.env.example`)
```bash
cp .env.example .env
```
แก้ค่าสำคัญใน `.env` — **จุดที่ต่างจาก Windows:**
```ini
ROBOFLOW_API_KEY=<คีย์ของคุณ>        # copy จาก .env เครื่อง Windows หรือหน้า Roboflow > Settings > API
ROBOFLOW_WORKSPACE=pholawats-workspace
ROBOFLOW_PPE_PROJECT=zentra-ppe

# ── สำคัญที่สุดสำหรับ Mac ──
TRAIN_DEVICE=mps          # ⚠️ ใช้ mps (GPU ของ Apple) — ห้ามใช้ 0 (นั่นคือ CUDA ของ NVIDIA จะ error)
TRAIN_WORKERS=0           # MPS + multiprocessing มักมีปัญหา → ตั้ง 0 ปลอดภัยสุด
YOLO_BASE_MODEL=yolov8s.pt
TRAIN_BATCH_SIZE=8        # Mac ใช้ unified memory ปรับตาม RAM (8GB→4, 16GB→8-16)
TRAIN_EPOCHS=60
```
ถ้าเจอ error ว่า op ไม่รองรับบน MPS ให้รันด้วย fallback:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## 2) เทรนด้วย dataset ใหม่ที่เพิ่งเพิ่ม

**ขั้นแรก:** ไปที่ Roboflow → เมนู **Versions** → **Generate New Version** (รวม data ใหม่) → จดเลข **version number** ที่ได้ (เช่น 3)

> เทรนอ้างอิงด้วย **เลข version** ไม่ใช่ชื่อ

**Smoke test ก่อน (3 epochs — พิสูจน์ว่า flow ครบ):**
```bash
python -m training.trainer --task ppe --project zentra-ppe --version <N> --epochs 3 --export
```

**เทรนจริงเต็ม:**
```bash
python -m training.trainer --task ppe --project zentra-ppe --version <N> --export
```
ผลลัพธ์: `models/ppe_finetuned.pt` + `.onnx` + `logs/metrics_ppe_*.json`

> ⚠️ **ต่อ (resume) จากที่เทรนบน Windows ไม่ได้** — checkpoint (`models/…/last.pt`) เป็นไฟล์ local ที่ถูก gitignore ไม่ได้ push ขึ้นมา → บน Mac ให้ **เทรนใหม่จาก version ใหม่** เลย

### Trainer CLI flags (เพิ่มใน session ล่าสุด)
| flag | ความหมาย |
|---|---|
| `--version N` | เลข Roboflow version (default 1) |
| `--epochs N` | override จำนวน epochs (เช่น 3 = smoke test) |
| `--project <slug>` | ชื่อ Roboflow project |
| `--task ppe\|fall` | โมดูล |
| `--export` | export ONNX หลังเทรน |
| `--no-aug` | ปิด offline augmentation |

---

## 3) ทดสอบโมเดลบนเว็บแคม
```bash
yolo predict model=models/ppe_finetuned.pt source=0 show=True conf=0.25 imgsz=640
```
`source=0` = เว็บแคม Mac · กด `Q` ที่หน้าต่างวิดีโอเพื่อปิด

---

## 4) สถานะโมเดลปัจจุบัน (จาก session Windows)

- โมเดลดีสุดตอนนี้: smoke 3 epochs, **mAP50 0.65** (11 คลาส)
- **AP รายคลาส:** helmet/no_helmet/gloves = แม่น ✅ | **vest (0.54), no_vest (0.27), no_glasses (0.23) = อ่อน** ⬇️
- glasses ได้ 0.86 บน val แต่จริงไม่ค่อยขึ้น = **domain gap** (รูปเทรนเป็นแว่นนิรภัย แต่เทสด้วยแว่นสายตาปกติ)
- **สาเหตุหลัก = domain gap + คลาส negative อ่อน** → แก้ด้วยการเพิ่มรูป**บริบทจริง** (มุมกล้อง/แสงจริง) โดยเฉพาะ vest & glasses (ที่กำลังทำอยู่)
- คลาสไม่สมดุล: helmet/person รวมกัน ~70% ของกล่องทั้งหมด

### 11 คลาส (ลำดับสำคัญ — ต้องตรงกับ Roboflow)
`Vest, boots, glasses, gloves, helmet, no_boots, no_glasses, no_gloves, no_helmet, no_vest, person`

---

## เช็กว่า Mac เห็น GPU (MPS)
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
ควรได้ `True` — ถ้า `False` จะเทรนบน CPU (ช้ากว่ามาก)
