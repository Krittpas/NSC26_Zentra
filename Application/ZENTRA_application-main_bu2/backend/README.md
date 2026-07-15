# ZENTRA — PPE Detection System

ระบบตรวจจับอุปกรณ์ป้องกันภัยส่วนบุคคล (PPE) ด้วย AI แบบ Real-time

---

## สิ่งที่ต้องติดตั้งก่อน

| โปรแกรม | วิธีติดตั้ง |
|---------|------------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop) | ดาวน์โหลดและติดตั้งตามปกติ |
| Python 3.11 | `brew install python@3.11` |
| Roboflow Account (ฟรี) | สมัครที่ [app.roboflow.com](https://app.roboflow.com) |

---

## วิธีติดตั้ง

**1. Clone โปรเจกต์**
```bash
git clone <repo-url>
cd TestDeteckl
```

**2. สร้างไฟล์ .env**
```bash
cp .env.example .env
```

เปิดไฟล์ `.env` แล้วใส่ API Key ของตัวเอง:
```
ROBOFLOW_API_KEY=ใส่_key_ของคุณที่นี่
```

> หา API Key ได้ที่ [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api)

**3. ติดตั้ง Python dependencies**
```bash
pip3.11 install -r requirements.txt
```

---

## วิธีรัน

**ขั้นที่ 1 — เปิด Inference Server (ต้องเปิด Docker Desktop ก่อน)**
```bash
docker compose up inference -d
```

> Server จะรันที่ `http://localhost:9001` ซึ่งตรงกับค่าใน config.py อยู่แล้ว ไม่ต้องแก้อะไร

รอจนขึ้น `Application startup complete` ตรวจสอบได้ด้วย:
```bash
docker compose logs inference
```

**ขั้นที่ 2 — รันระบบ**
```bash
python3.11 main.py
```

กด `q` ในหน้าต่างกล้องเพื่อหยุดระบบ

---

## โครงสร้างโปรเจกต์

```
TestDeteckl/
├── main.py              # Entry point
├── config.py            # ตั้งค่าระบบ
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Docker setup
├── .env.example         # Template สำหรับ API Keys
├── modules/
│   ├── ppe.py           # Module 1: PPE Detection
│   ├── safety_zone.py   # Module 2: Safety Zone
│   └── heat_stroke.py   # Module 3: Heat Stroke
└── alerts/
    └── line_notify.py   # LINE Notification
```

---

## การตั้งค่ากล้อง

แก้ได้ที่ [config.py](config.py):

```python
CAMERA_SOURCE = "webcam"   # ใช้ webcam ของเครื่อง
# CAMERA_SOURCE = "rtsp"   # ใช้กล้อง IP/CCTV
```

ถ้าใช้ RTSP ให้ตั้งค่า URL:
```python
RTSP_URL = "rtsp://admin:password@192.168.x.x:554/stream1"
```

---

## แก้ปัญหาเบื้องต้น

| ปัญหา | วิธีแก้ |
|-------|---------|
| `can't open file 'main.py'` | ต้อง `cd` เข้าโฟลเดอร์โปรเจกต์ก่อน |
| `ไม่สามารถเปิดกล้องได้` | ตรวจสอบ `WEBCAM_INDEX = 0` ใน config.py |
| Inference Server ไม่ขึ้น | ตรวจสอบว่าเปิด Docker Desktop แล้ว |
| `401 Unauthorized` | ROBOFLOW_API_KEY ใน .env ไม่ถูกต้อง |
