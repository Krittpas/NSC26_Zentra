# ZENTRA — แผนพัฒนา UX/UI หน้า Dashboard (Live)

> สถานะ: **แผน — ยังไม่ลงมือ** · โฟกัสหน้า Dashboard ก่อน (หน้าแรกที่กรรมการ NSC เห็น)
> ทำในแอปจริง (ไม่ใช่ artifact) เพื่อเคารพ design system เดิม + ข้อจำกัด WebView2

## เป้าหมาย
ยกหน้า Live ให้ดู "ห้องควบคุมโรงงาน" ที่เป็นทางการ น่าเชื่อถือ อ่านสถานะได้ใน 2 วินาที
โดยคงธีม control-room โทนขาวเดิม (tokens ใน [style.css](../ui/assets/style.css)) ไม่รื้อ logic

## โครงปัจจุบัน (อ้างอิง)
[dashboard.html](../ui/screens/dashboard.html): navbar → KPI strip (5 tiles) → body[ วิดีโอ (ซ้าย, พื้นดำ) | right-panel: module status, emergency banner, alarm list, ปุ่ม ]
JS helper ที่ใช้ซ้ำได้: `_updateKPIs / _renderAlarms / _showEmergencyBanner / _updateCameraState` ใน [app.js](../ui/assets/app.js)
มี CSS `.modal-*` (evidence modal) อยู่แล้วใน style.css — reuse ได้

---

## ระดับ 1 — Polish (คงเลย์เอาต์เดิม, ปรับให้พรีเมียมขึ้น)

1. **HUD บนวิดีโอ (สร้างฟีล control-room)** — overlay มุมจอแบบบางเบา: ชื่อกล้อง + จุด **● LIVE** กระพริบ
   + เวลาเรียลไทม์ + badge FPS และ badge โมเดล (cloud/local จาก `status.ppe_model`). ปัจจุบันมีแค่ป้ายเล็ก `camera-label`
2. **KPI tiles** — เพิ่มไอคอนเล็ก/ลำดับความเด่น: ทำ tile "ฉุกเฉิน" เด่นสุด, คุมสีตาม semantic
   (warning=amber, emergency=red), เพิ่มหน่วยกำกับ. ปรับ `--shadow`/border-left ให้คม
3. **Alarm list** — เพิ่มไอคอนระดับ + **เวลาแบบสัมพัทธ์** ("2 นาทีที่แล้ว") + badge จำนวนใน section title;
   ปรับ rhythm/spacing ให้สแกนง่าย
4. **Module status** — ใส่ไอคอนประจำโมดูล (helmet/zone/fall) + จุดสถานะคมขึ้น; แสดง "อัปเดตล่าสุด"
5. **typography + spacing** — ไล่ระดับหัวข้อ/ตัวเลข KPI, ระยะ panel ให้สม่ำเสมอทั้งหน้า
6. **สถานะว่าง/กำลังต่อ** — no-signal วิดีโอแบบมีแบรนด์ (ตอนนี้พึ่ง `no-signal.svg` + onerror); overlay spinner มีแล้ว

## ระดับ 2 — UX (เพิ่มการใช้งาน, แตะ logic เล็กน้อย)

7. **คลิก alarm → เปิด evidence modal** — reuse `.modal-*` + `/api/history/snapshot/{id}` (ปัจจุบัน alarm ใน
   dashboard คลิกไม่ได้; ต้องเก็บ `event id` ตอน push เข้า `recentAlarms`)
8. **Emergency เด่นชัดเมื่อเกิดเหตุ** — ยกเป็นแถบเต็มความกว้างเหนือวิดีโอ + ปุ่ม "รับทราบ" (acknowledge)
   + auto-fade เมื่อหมดเหตุ (ตอนนี้อยู่แค่ right panel ค้างจนเปลี่ยนหน้า)
9. **ปุ่ม fullscreen บนวิดีโอ** — reuse `JsApi.toggle_fullscreen()` (มีแล้วใน [app.py](../app.py))
10. **badge โมเดล/FPS เรียลไทม์** — ดึงจาก `/api/status` ที่ poll อยู่แล้ว

## ระดับ 3 — Redesign (ออปชัน ถ้าอยากรื้อเลย์เอาต์ — เทสต์ละเอียด)

11. **เลย์เอาต์ 3 โซนห้องควบคุม**: rail ซ้าย (system health) · วิดีโอ hero ตรงกลาง · rail ขวา (live alarm feed)
    ย้าย KPI ลงเป็นแถบบางบนวิดีโอ → วิดีโอได้พื้นที่มากขึ้น
12. **Event ticker ด้านล่าง** — แถบเลื่อนเหตุการณ์ล่าสุดแบบจอมอนิเตอร์
13. **Multi-camera grid (placeholder)** — เล่าเรื่อง scalability ให้กรรมการ (รองรับหลายกล้องในอนาคต)

---

## ข้อจำกัด WebView2 (ต้องเคารพตอนลงมือ)
- script ใน screen ใช้ `var` ไม่ใช่ `let/const` (re-navigation จะ redeclare error)
- ใช้ tokens/คลาสจาก style.css เดิม ไม่ hardcode สี
- ทุก dynamic element ต้องมี id ให้ helper ใน app.js หาเจอ

## Verification (ตอนลงมือจริง)
- รันแอป (`run_zentra`) → ต่อกล้อง → เข้า/ออกหน้า Dashboard ซ้ำ ๆ ดูว่าไม่พัง/ไม่ค้าง
- ยิง alert จริง → ดู alarm list, emergency banner, คลิกดู evidence
- ปรับขนาดหน้าต่าง (min 1024×640) ดู responsive
- ตรวจ console ไม่มี error

## ขอบเขต
- **หน้า Dashboard เท่านั้นรอบนี้** (หน้าอื่นไว้ทีหลัง ใช้ภาษา/โทนเดียวกัน)
- ยังไม่เลือกระดับ (Polish vs Redesign) — เลือกตอนเริ่มลงมือ; แนะนำเริ่ม **ระดับ 1–2** ก่อน
