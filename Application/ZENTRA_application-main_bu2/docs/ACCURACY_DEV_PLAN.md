# ZENTRA — แผนพัฒนาความแม่นยำ 3 โมดูล (PPE / Dangerous Zone / Heat-Stress)

> สถานะ: **แผน — ยังไม่ลงมือ** · เน้น **ความถูกต้อง/แม่นยำ** · อิงไอเดีย Protex AI + best practice
> อ้างอิงไอเดียจาก `ZENTRA_Accuracy_System.md` + `ZENTRA_Protex_Inspired_Dev.md` ของผู้ใช้

## Context
ผู้ใช้อยากให้ ZENTRA แม่นและใช้งานดีระดับ Protex AI / ระบบ safety vision เชิงพาณิชย์
จากการค้นข้อมูล Protex AI: ทำงานบน CCTV เดิม · **edge + เบลอ/anonymize หน้า (ไม่เก็บ PII)** ·
**rules engine** แปลงนโยบายเป็นกฎ · analytics เชิงรุก. ฝั่ง best-practice PPE ตรงกันว่า
ความแม่นยำมาจาก **per-person association + persistent track ID + multi-frame confirmation +
ตรวจว่าส่วนร่างกายมองเห็นจริงก่อนตัดสิน** (= แหล่ง false-positive อันดับ 1)

**ข้อค้นพบสำคัญจากโค้ดปัจจุบัน (จุดที่ทำให้ "ไม่แม่น" ในงานหลายคน):**
ทุกโมดูลใช้ **state แบบ global ตัวเดียว** ไม่ได้แยกตามคน —
[ppe.py](../../ZENTRA/modules/ppe.py) `_violation_streak` ตัวเดียว, [safety_zone.py](../../ZENTRA/modules/safety_zone.py)
`_intrusion_streak` ตัวเดียว, [heat_stroke.py](../../ZENTRA/modules/heat_stroke.py) ใช้ MediaPipe
`_pose_states[0]` = **คนเดียวเท่านั้น**. ในโรงงานหลายคนสิ่งนี้ทำให้นับผิด/เตือนผิด
→ **การแก้ที่ให้ผลแม่นยำสูงสุดคือเปลี่ยนเป็น per-track-ID ทุกโมดูล** (ไม่ใช่สถาปัตยกรรมแปลก ๆ)

---

## ZENTRA ตอนนี้ เทียบ Protex / best practice (ช่องว่าง)

| ด้าน | ZENTRA ตอนนี้ | ช่องว่าง (สิ่งที่ควรมี) |
|------|----------------|--------------------------|
| Track ID | มี `ByteTracker` ([utils/tracker.py](../../ZENTRA/utils/tracker.py)) แต่ **ใช้แค่ Zone** | ใช้ track เดียวร่วมทุกโมดูล (single-pass) |
| PPE→Person | ตรวจคลาส no_* แบบ global ไม่ผูกกับคน | associate PPE เข้าแต่ละคน + เช็ค visibility |
| Confirm | streak global | per-track buffer (เช่น 3/5 เฟรม/คน) |
| Pose | MediaPipe **1 คน** | YOLOv8-Pose **หลายคน** |
| Heat-stress | จริง ๆ มีแต่ fall | indicators เชิงพฤติกรรม + temporal |
| Zone dwell | streak ไม่ใช่เวลา​จริง | entry/exit timestamp ต่อ track |
| Zone ขั้นสูง | ไม่มี | zone types / exclusion / schedule |
| Privacy | เก็บในเครื่อง (ดีแล้ว) | **เบลอหน้า (anonymize)** แบบ Protex |
| วัดผล | ไม่มี eval set จริง | frozen test set + report mAP/P/R/F1 |

---

## P0 — รากฐานความถูกต้อง (ผลกระทบสูงสุด ทำก่อน)

1. **Single-pass tracking ร่วมทุกโมดูล** — detect person + `ByteTracker.update()` ครั้งเดียวใน
   [pipeline.py](../pipeline/pipeline.py) `_process_loop` แล้วส่ง `tracks` (มี `track_id`) ให้ PPE/Zone/Heat ใช้ร่วม
   → ผลตรงกัน + เร็วขึ้น (reuse `ByteTracker` ที่มีอยู่)
2. **Per-track-ID state แทน global** — เปลี่ยน `_violation_streak/_intrusion_streak/_last_alert`
   เป็น dict keyed by `track_id` ทั้ง 3 โมดูล. คูลดาวน์/คอนเฟิร์มแยกรายคน → นับถูก เตือนถูก
3. **Person↔PPE association + visibility gate** — ผูกกล่อง PPE/ no_* เข้ากับ person bbox (overlap ≥ ~30%,
   ดู MD Layer 3); **ตัดสิน "ไม่สวม X" เฉพาะเมื่อบริเวณนั้นของคนมองเห็นจริง + person conf สูง**
   → ลด false positive อันดับ 1 ตาม best practice
4. **Multi-frame confirmation ต่อ track** — เตือนเมื่อพบ violation เดิม ≥ N/M เฟรมของ track เดียวกัน
   (เช่น 3/5). ปัจจุบันมีแต่ global streak
5. **Evaluation harness + frozen test set** — โฟลเดอร์ test ที่ label มือ ไม่เคยเทรน, สคริปต์วัด
   mAP/precision/recall (PPE), precision/false-alert (Zone), F1 (Heat) → **พิสูจน์ตัวเลขให้กรรมการ NSC**
   (ผูกกับแผน active-learning ใน plan file หลัก)

---

## Module 1 — PPE (เป้า mAP@0.5 ≥ 85%, false-positive ต่ำ)
- ทำ P0 ข้อ 3–4 (association + visibility + per-track confirm) ← ผลแม่นยำสูงสุด
- **ปรับ data/label + active-learning loop** (ดู plan file หลัก) — "label ถูก + หลากหลาย" ให้ผลมากกว่าทุกเทคนิค
- per-zone confidence (กฎต่างพื้นที่เข้มต่างกัน) — ต่อยอดจาก settings เดิม
- ⚙️ **ออปชัน/ROI ต่ำ (อย่าพึ่งทำ):** CBAM attention, ESPCN super-resolution, OAM-YOLO
  จาก MD — เป็น research-grade, ต้องแก้สถาปัตยกรรม/เทรนใหม่, ได้ผลเพิ่มเล็กเทียบแรง.
  ทำเฉพาะถ้ากล้องความละเอียดต่ำ/วัตถุเล็กจริง ๆ และหลังรากฐานเสร็จ

## Module 2 — Dangerous Zone (เป้า precision ≥ 90%)
- **Dwell time จริงต่อ track** — เก็บ entry/exit timestamp (MD Layer 3 `ZoneDwellTracker`) แทน streak
- **Zone types** RESTRICTED / CONTROLLED(เช็ค PPE) / MONITORED(นับจำนวน) — ต่อ `zones.json` + UI
- **Exclusion zone masking** — พื้นที่ไม่ตรวจ (โต๊ะ/จอ/นอกพื้น) → ลด false alert (Protex มี)
- **PPE-in-zone combo** — "อยู่โซน CONTROLLED + ไม่สวม PPE" = เตือน (ใช้ track ร่วม → ทำได้ตรง ๆ)
- foot-point + confirmed-track มีแล้ว ([safety_zone.py](../../ZENTRA/modules/safety_zone.py)) — คงไว้

## Module 3 — Heat-Stress Behavioral (เป้า F1 ≥ 80%)
- **เปลี่ยนเป็น YOLOv8-Pose (หลายคน)** แทน MediaPipe 1 คน — correctness ในงานหลายคน
- **Behavioral indicators ต่อ track + temporal** (MD Module 3): หยุดนิ่งผิดปกติ, ก้มหัวค้าง,
  ก้มตัวค้าง, เคลื่อนช้าลงเทียบ baseline, เดินเซ, ทรุด/ล้ม → รวมเป็น risk score (ถ่วงน้ำหนัก)
- คง **hybrid fall (YOLO+Pose)** ที่ทำไว้ เป็นส่วนหนึ่งของ collapse indicator
- ⚙️ **ออปชัน:** DHT22 sensor fusion → heat index (OSHA) ถ่วงกับ behavioral score — เป็น
  จุดขายเฉพาะตัว แต่ต้องมีฮาร์ดแวร์; ทำหลังสุด

---

## App / UX (อิง Protex — เรียงตาม ROI)
1. **เบลอหน้า/anonymize** ในภาพหลักฐาน (Protex ไม่เก็บ PII) — เสริม PDPA ที่เป็นจุดขาย, ทำไม่ยาก
2. **Per-zone config + วาด exclusion** ใน Zone Editor — หนุน Module 2
3. **Dashboard analytics** — เทรนด์ 7 วัน, โซนเสี่ยงสุด, PPE ที่ขาดบ่อย (มี SQLite store แล้ว)
4. **Zone utilization heatmap** — ภาพความหนาแน่นคน (Gaussian) — ดูโปรเฉพาะตอนเดโม
5. **Corrective action tracker** — สถานะแก้ไขเหตุการณ์ (open/in-progress/resolved)
6. ⚙️ **Drag-drop Rule Builder** (MD ระบบ 1) — เท่แต่ **งานใหญ่มาก**; สำหรับ NSC แนะนำเริ่มเป็น
   "rule แบบฟอร์ม/ดรอปดาวน์" ก่อน ไม่ใช่ canvas เต็ม

---

## คำแนะนำ ROI (ตรงไปตรงมา)
- **ทำก่อน (ได้ผลจริง):** P0 ทั้งหมด (per-track + association + visibility + eval) + data/label.
  นี่คือสิ่งที่ทำให้ "แม่นขึ้นจริง" และพิสูจน์ได้
- **ทำต่อ:** Module 2 (dwell/zone types/exclusion), Module 3 (YOLOv8-Pose + behavioral)
- **ออปชัน/อย่าเพิ่งทุ่ม:** CBAM/ESPCN/OAM-YOLO, DHT22, REBA, vehicle, near-miss, drag-drop rule builder
  — งานเยอะ ผลเพิ่มน้อยเทียบรากฐาน. ใส่เป็น "โรดแมป Sprint ถัดไป" เพื่อเล่าวิสัยทัศน์ให้กรรมการ

## Phasing
1. P0: single-pass tracking + per-track state (3 โมดูล)
2. P0: association + visibility gate (PPE) + per-track confirm
3. P0: eval harness + frozen test sets → วัด baseline
4. Module 2 ขั้นสูง (dwell/types/exclusion/combo)
5. Module 3: YOLOv8-Pose + behavioral indicators
6. App/UX: เบลอหน้า → analytics → per-zone config
7. ออปชันตาม ROI

## Verification (ต้องวัดเป็นตัวเลข)
- รัน eval harness บน frozen test set → รายงาน mAP@0.5 / precision / recall (PPE),
  precision + false-alert/ชม. (Zone), F1 (Heat) **ก่อน-หลัง** ทุกการเปลี่ยน
- ทดสอบหลายคนในเฟรม: นับ violation ต่อคนถูก, track ID ไม่สลับ
- ทดสอบ false positive: คนบังกัน / PPE ถูกบัง / ส่วนร่างกายมองไม่เห็น → ต้องไม่เตือนพลาด
- `py_compile` + รันแอปจริง (ข้อจำกัด WebView2 ตามเดิม)

## ขอบเขต
- เน้น **ความถูกต้อง/แม่นยำ** ก่อนฟีเจอร์หรูหรา
- **วางแผนเท่านั้น — รอผู้ใช้เลือกว่าจะเริ่มเฟส/โมดูลใดก่อน**
