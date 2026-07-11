# ZENTRA — UI Redesign Plan (Dark Professional · Protex-style)

> สถานะ: **แผน** · ทิศทางที่ผู้ใช้เลือก: **ธีมมืดโปร + Left sidebar + เริ่มรากฐาน+Dashboard ก่อน**
> อ้างอิงดีไซน์จาก `ZENTRA_UI_Design_Spec.md` (Protex AI / viAct / Visionify)

## Context
ต้องการให้แอปดู**ทางการ/โปรเหมือน Protex AI** และใช้ง่ายขึ้น
**ข้อจำกัดสำคัญ:** design spec เขียนสำหรับ **PyQt6** แต่แอปจริงเป็น **PyWebView + FastAPI + HTML/CSS/JS**
(เว็บใน WebView2). จะ **ไม่เขียนใหม่เป็น PyQt6** (จะทิ้ง backend/pipeline ที่ทำมาทั้งหมด) แต่จะ
**แปลงภาษาดีไซน์** (โทนสี/sidebar/cards/tokens/layout) มาทำใน **CSS/HTML เดิม** ให้ได้หน้าตาเดียวกัน

ไฟล์ที่เกี่ยว: [index.html](../ui/index.html) · [style.css](../ui/assets/style.css) ·
[app.js](../ui/assets/app.js) · [ui/screens/*.html](../ui/screens/)

---

## A. รากฐาน — Design System (โทนมืด) เป็น CSS variables
แอปใช้ CSS tokens อยู่แล้ว (`var(--bg)`, `--accent`, `--text`…) → **แค่เปลี่ยนค่าใน `:root`** ทั้งแอปก็พลิกเป็นมืดทันที แล้วค่อยเก็บรายละเอียด

| token | ค่าใหม่ (มืด) |
|-------|--------------|
| `--bg` | `#0D0F14` (ฐาน) |
| `--bg-panel` / `--bg-sidebar` | `#13161D` |
| `--bg-card` | `#1A1E28` |
| `--bg-card-alt` / hover | `#212636` |
| `--border` / `--border-mid` | `#252A37` / `#353C52` |
| `--text` / `--text-sub` / `--text-muted` | `#E8EAF0` / `#9AA0B8` / `#5C6480` |
| `--accent` (info/PPE) | `#3B82F6` |
| success/warning/danger | `#10B981` / `#F59E0B` / `#EF4444` |
| module: ppe/zone/heat | `#3B82F6` / `#F59E0B` / `#EF4444` |

- **Font:** Inter (ละติน) + Noto Sans Thai (ไทย) ตาม spec (โหลดผ่าน Google Fonts เหมือนที่โหลด Sarabun อยู่); fallback `sans-serif`
- เพิ่ม scale typography/spacing/radius ตาม spec (size xs–2xl, radius sm–xl)
- เงา/►scrollbar/► input/► toggle ปรับโทนมืด

## B. Left Sidebar Navigation (แทน navbar บน)
- เพิ่ม `renderSidebar(active)` ใน [app.js](../ui/assets/app.js) แทน `renderNavbar`
- โครง: โลโก้+เวอร์ชัน · หมวด **MAIN** (Dashboard, Live Monitor, Zone Editor) · **EVENTS** (Alert Feed/History) ·
  **SETTINGS** (Configuration) · footer สถานะ (🟢 System Online · AI ON · LINE ON)
- Active = พื้น `--bg-elevated` + เส้นซ้าย 3px สี accent; hover transition 150ms
- **โครงแอป (shell):** ปรับ [index.html](../ui/index.html) ให้มี sidebar ถาวร + `#content` ที่สลับเฉพาะเนื้อหา
  → sidebar ไม่กระพริบตอนเปลี่ยนหน้า. `navigate()` สลับเฉพาะ `#content`
- **splash / source** = เต็มจอ ไม่มี sidebar (เช็คชื่อหน้าก่อนแสดง shell)

```
┌────────────┬──────────────────────────────────────────┐
│ 🛡 ZENTRA  │  Topbar: [Page Title]      🕐  🔔  ⚙      │
│ v1.0       ├──────────────────────────────────────────┤
│ MAIN       │                                          │
│ ▸Dashboard │   CONTENT (สลับตามหน้า)                   │
│  Live      │                                          │
│  Zone      │                                          │
│ EVENTS     │                                          │
│  Alerts    │                                          │
│ SETTINGS   │                                          │
│  Config    │                                          │
│ 🟢 Online  │                                          │
└────────────┴──────────────────────────────────────────┘
```

## C. Dashboard ใหม่ (หน้าแรก หลังรากฐานเสร็จ)
- **KPI cards แถวบน** (4 ใบ): Alerts วันนี้ / PPE / Zone / Heat — ไอคอน + ตัวเลขใหญ่ + trend (▲/▼ %) +
  ขอบสีตามโมดูล (reuse data `/api/history/today`)
- **กราฟ**: Violations รายชั่วโมง (bar) · Top risk zones (bar list) · เทรนด์ 7 วัน (line) — ใช้ **Chart.js**
  (โหลดอยู่แล้วในหน้า History) · Camera status panel
- **Recent events**: ตารางเหตุการณ์ล่าสุด (reuse `/api/history/events`) + คลิกดู evidence
- การ์ด/พาเนลใช้คอมโพเนนต์มาตรฐาน (ดู D)

## D. คอมโพเนนต์ที่ทำใหม่ (reusable, มืด)
KPI card · panel/card · table (dark) · badge/level chip · **toast notification** (มุมขวาบน เด้งเมื่อมี alert) · status pill

---

## E. Phasing
1. **รากฐาน (รอบนี้):** dark tokens ใน style.css + ฟอนต์ + sidebar shell + ปรับ router ให้สลับเฉพาะ `#content`
   → ทุกหน้าหลักพลิกเป็นมืด + sidebar ทันที (Zone/History/Settings ใช้ของเดิมได้ในธีมมืด)
2. **Dashboard:** KPI cards + charts + recent events
3. (ภายหลัง) Live Monitor หลายกล้อง · Alert Feed · Event Log · Rule Builder · Toasts

## F. ข้อจำกัด WebView2 (ต้องเคารพ)
- script ในหน้าใช้ `var` ไม่ใช่ `let/const` · re-inject script ตอน navigate · ไม่มี SSE/blob download
- ฟอนต์ไทยต้องมี (Noto Sans Thai/Sarabun) · Chart.js โหลดผ่าน CDN (re-load ใน navigate)
- ทุก element ที่ JS อ้างต้องมี id

## G. Verification
- รันแอป → วนทุกหน้า: ธีมมืดสม่ำเสมอ, sidebar active ถูก, ไม่กระพริบ
- Dashboard: KPI/กราฟ/recent events มีข้อมูลจริง, คลิก event เปิด evidence
- ขนาด 1280×800: layout ไม่แตก · console ไม่มี error · splash/source ยังเต็มจอปกติ

## ขอบเขตรอบนี้
- **รากฐาน (A+B) + Dashboard (C)** ก่อน — เห็นผลชัดสุด ความเสี่ยงคุมได้
- หน้าอื่น (Live/Alert Feed/Rule Builder) เป็นเฟสถัดไป
