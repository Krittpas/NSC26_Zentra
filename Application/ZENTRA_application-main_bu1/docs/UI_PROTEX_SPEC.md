# ZENTRA — Protex-style UI Design Spec

> สเปกออกแบบ UI ของ ZENTRA ให้ดูทางการระดับ SaaS เหมือน **Protex AI**
> อ้างอิงจากสกรีนช็อตผลิตภัณฑ์จริงของ Protex (Dashboard / Cameras / Rule Builder / Reports)
> ใช้เป็นคู่มือพัฒนา UI ทีละส่วน (vanilla HTML/CSS/JS — PyWebView SPA)

---

## 0. เป้าหมาย & ภาพรวม

ทำให้ ZENTRA มี **โครงและภาษาภาพแบบ SaaS มืออาชีพ**: thin icon sidebar + topbar บริบท
+ การ์ดสถิติสะอาด + ตารางข้อมูลมีลำดับชั้น + กราฟวงกลม Safety Score + event card ลอยบนวิดีโอ

### ⚠️ การตัดสินใจเรื่องธีม (ต้องเลือกก่อน)
| | Protex (ในภาพ) | ZENTRA ตอนนี้ |
|---|---|---|
| ธีม | **Light** (ขาว/เทาฟ้าอ่อน) | **Dark** (`--bg:#0D0F14`) |

สกรีนช็อต Protex เป็น **ธีมสว่าง** ทั้งหมด แต่ ZENTRA เพิ่งทำธีมมืดไป
→ สเปกนี้ให้ **ทั้งสองพาเลต** (§3) เลือกได้ว่าจะสลับเป็น light ตาม Protex เป๊ะ
หรือคงธีมมืดแล้วใช้แค่ layout/component language ของ Protex

**คำแนะนำ:** ใช้ layout + component ของ Protex (ส่วนที่มีค่าจริง) แต่ทำ **โทเคนธีมสลับได้** (light/dark)
เพราะ ZENTRA วาง CSS variables ไว้แล้ว — สลับธีมทั้งแอปได้ด้วยการแก้ `:root`

---

## 1. อ้างอิงจากสกรีนช็อต Protex (สิ่งที่เห็น)

**ภาพ 1 — Dashboard**
- Thin **icon-only sidebar** (~56px) ซ้ายสุด: โลโก้ + ไอคอน grid/camera/sliders/lightning/doc
- **Topbar**: ตัวเลือกสถานที่ `📍 DSG · Dublin` (ซ้าย) + ธง/ภาษา + avatar ผู้ใช้ (ขวา)
- แถว **KPI cards**: Violations 12 (▲2% แดง), Activity 81.7% (▼0.75% เขียว), Cameras Online 1
  — แต่ละใบ: หัวข้อ + ไอคอนวงกลมมุมขวาบน + ตัวเลขใหญ่ + delta เล็กมีสี
- **Safety Score**: การ์ดกราฟวงกลม (donut) 83% เขียว + Yesterday 39% / This Week 78% ด้านล่าง
- **Live video** ขวา: overlay โซน exclusion สีส้ม/เหลือง + คนที่ตรวจจับ + event card "Exclusion Zone Violation Logged" ลอยมุมล่าง
- **Footer**: โลโก้ + `COPYRIGHT © 2021 ...`

**ภาพ 2 — Cameras**
- แถวการ์ด: **`+ Add Camera`** (เส้นประ), Online 12, Offline 0
- **ตาราง Cameras**: Name · Feed (thumbnail) · Location · Date Created · Rules Active · **Compliance (progress bar)** · Activity · **Status (badge)** · RTSP
- Pagination: `Showing 1 to 3 of 3 entries`

**ภาพ 3 — Rule Builder** (node graph)
- พาเนลซ้าย: RULES (Exclusion Zone, Speed Limit) + VISION BLOCKS (Camera/Object/Metric blocks: Person, Vehicle, PPE, Distance)
- แคนวาสกลาง: **flow diagram** ลากบล็อกต่อกันด้วยเส้นประ → ปลายทาง Notification / Report / Text

**ภาพ 4 — Reports**
- หัว: `Reports [10▼]  [+ New Report]` + ช่องค้นหา
- **ตาราง**: # · ไอคอนสถานะ · Created By (**avatar + ชื่อ + อีเมล**) · Period · Date Created

---

## 2. หลักการออกแบบ (Design Principles)
1. **โครงคงที่ (persistent shell)** — icon sidebar + topbar อยู่ทุกหน้า เนื้อหาเปลี่ยนตรงกลาง
2. **การ์ดสะอาด มีลำดับชั้น** — พื้นขาว/เทาเข้ม, มุมโค้ง, เงาบาง, ระยะหายใจเยอะ
3. **สีมีความหมาย** — เขียว=ปลอดภัย/ดีขึ้น, แดง=ละเมิด/แย่ลง, ฟ้า=ข้อมูล/แอคชัน, เหลือง/ส้ม=เฝ้าระวัง/โซน
4. **ตัวเลขเป็นพระเอก** — KPI ตัวใหญ่ + delta เล็กมีทิศทาง (▲▼)
5. **ตารางอ่านง่าย** — avatar, progress bar, status badge, pagination
6. **ภาษาภาพเดียวทั้งแอป** — icon line-style ชุดเดียว, radius/shadow/spacing จากโทเคนชุดเดียว

---

## 3. Color Tokens (ใส่ใน `:root` ของ style.css)

### 3A. Light palette (ตาม Protex)
```css
--bg:          #EAF2F6;   /* app base — ฟ้าเทาอ่อน           */
--bg-panel:    #FFFFFF;   /* sidebar / topbar / การ์ด         */
--bg-card:     #FFFFFF;
--bg-card-alt: #F5F8FA;   /* hover / แถวสลับ / input          */
--bg-video:    #0B0E13;   /* พื้นวิดีโอ (ดำเสมอ)              */

--accent:      #2D6BFF;   /* Protex blue — แอคชัน/ลิงก์/active */
--accent-dim:  rgba(45,107,255,0.10);
--green:       #22B07D;   /* ปลอดภัย / ดีขึ้น                 */
--red:         #EF4655;   /* ละเมิด / แย่ลง                   */
--yellow:      #F5A623;   /* เฝ้าระวัง                        */
--orange:      #FF7A45;   /* โซน exclusion                    */

--text:        #1A2B4A;   /* หัวข้อ/ตัวเลข — navy เข้ม         */
--text-sub:    #5B6B85;   /* รอง                              */
--text-muted:  #97A2B6;   /* จาง                              */
--border:      #E3E9F0;   /* เส้นการ์ด/ตาราง                  */
--shadow:      0 1px 3px rgba(16,30,54,0.06), 0 6px 18px rgba(16,30,54,0.05);
```

### 3B. Dark palette (ZENTRA ปัจจุบัน — ถ้าคงธีมมืด)
```css
--bg:#0D0F14; --bg-panel:#13161D; --bg-card:#1A1E28; --bg-card-alt:#212636;
--accent:#3B82F6; --green:#10B981; --red:#EF4444; --yellow:#F59E0B; --orange:#F97316;
--text:#E8EAF0; --text-sub:#9AA0B8; --text-muted:#5C6480; --border:#252A37;
```

### 3C. โทเคนร่วม
```css
--font: 'Sarabun', 'Inter', sans-serif;   /* Protex ใช้ geometric sans; Sarabun รองรับไทย */
--r-sm: 8px;  --r: 12px;  --r-lg: 16px;    /* การ์ด Protex โค้ง ~12-16px */
--sidebar-w: 60px;        /* thin icon rail (เดิม 210px → แคบลงแบบ Protex) */
--topbar-h: 60px;
```

---

## 4. Layout System (โครงรวม)

```
┌────┬─────────────────────────────────────────────┐
│ S  │  TOPBAR: 📍สถานที่            ภาษา · 🔔 · 👤  │  ← --topbar-h 60px
│ I  ├─────────────────────────────────────────────┤
│ D  │                                             │
│ E  │            CONTENT (ต่อหน้า)                 │  ← scroll เฉพาะตรงนี้
│ B  │                                             │
│ A  ├─────────────────────────────────────────────┤
│ R  │  FOOTER: โลโก้ · © 2026 ZENTRA              │
└────┴─────────────────────────────────────────────┘
 60px
```

- **Sidebar**: `position:fixed; left:0; width:60px` — icon-only, จัดกลางแนวตั้ง, active = ไฮไลต์ฟ้า
- **Topbar**: `position:fixed; top:0; left:60px; right:0; height:60px`
- **Content**: `margin:60px 0 0 60px; padding:24px; overflow:auto`
- หน้า splash/source ไม่ต้องมี shell (เหมือนตอนนี้)

---

## 5. Component Specs

### 5.1 Icon Sidebar (rail)
- กว้าง 60px, พื้น `--bg-panel`, เส้นขวา `--border`
- บนสุด: โลโก้ ZENTRA (กล่องฟ้า gradient + ไอคอนโล่)
- กลาง: ปุ่มไอคอน 44×44 (line-icon 20px) เว้น gap 4px
  - active: พื้น `--accent-dim` + ไอคอนสี `--accent` + แถบฟ้าซ้าย 3px
  - hover: พื้น `--bg-card-alt`
- **tooltip**: hover แล้วป้ายชื่อโผล่ข้าง ๆ (เพราะ icon-only)
- map ไอคอน → หน้า: `grid→Dashboard` · `camera→Live/Source` · `map→Zone` · `bell/⚡→Alerts/History` · `doc→Reports` · `sliders→Settings`

### 5.2 Topbar
- ซ้าย: **location selector** `📍 ZENTRA · โรงงาน A ▾` (dropdown, ตอนนี้ fix ค่าเดียวได้)
- ขวา: ปุ่มภาษา (TH/EN), ไอคอนแจ้งเตือน 🔔 (badge ตัวเลข), avatar ผู้ใช้ + ชื่อ
- พื้น `--bg-panel`, เส้นล่าง `--border`

### 5.3 KPI Stat Card  ← ปรับจาก `.kpi-tile` ที่มี
```
┌─────────────────────────────┐
│ Violations            (◔)   │  ← title (text-sub) + ไอคอนวงกลมมุมขวา
│                             │
│ 12                          │  ← ตัวเลขใหญ่ 32-36px bold (--text)
│ ▲ 2% since last week        │  ← delta: ▲แดง=แย่ลง / ▼เขียว=ดีขึ้น
└─────────────────────────────┘
```
- การ์ด: พื้น `--bg-card`, radius `--r`, shadow `--shadow`, padding 18-20px
- delta สี: ตามทิศทาง "ดี/แย่" ของ metric นั้น (violations เพิ่ม=แดง, activity ลด=แดง)
- ไอคอนมุมขวา: วงกลมจาง สีตาม metric

### 5.4 Donut Gauge — Safety Score
- วงแหวน SVG (`stroke-dasharray`) สีไล่จากแดง→เหลือง→เขียวตาม % (หรือเขียวล้วนถ้า ≥80)
- กลางวง: `83%` ใหญ่ + `Today` เล็ก
- ใต้การ์ด: 2 ช่อง `Yesterday 39%` · `This Week 78%`
- ZENTRA นิยาม Safety Score = `100 − (น้ำหนัก×เหตุการณ์/ชม.)` หรือ `% เวลาที่ไม่มีการละเมิด`

### 5.5 Live Video Panel + Event Toast
- กล่องวิดีโอพื้น `--bg-video`, radius `--r`, มุมมี label กล้อง
- **event card ลอย** (มุมล่างซ้าย): ไอคอนเตือน + หัวข้อ "Exclusion Zone Violation Logged" + เวลา/กล้อง
  - พื้นขาว/การ์ด, เงา, เด้งเข้า slide-up, หายเองใน ~5s (= Toast ที่วางแผนไว้)
- overlay โซน = วาดจาก backend อยู่แล้ว (สีส้ม/เหลือง)

### 5.6 Data Table (Cameras / Reports / History)
- หัวตาราง: `--text-muted` ตัวเล็ก uppercase, เส้นล่าง `--border`
- แถว: สูง ~56px, hover `--bg-card-alt`, เส้นคั่นบาง
- เซลล์พิเศษ:
  - **avatar + ชื่อ + อีเมล** (2 บรรทัด) — col "Created By / ผู้ใช้"
  - **thumbnail** กล้อง (รูปย่อ radius 6px)
  - **progress bar** Compliance (แถบเขียว + % ข้าง)
  - **status badge**: `Online` (เขียวพื้นจาง) / `Offline` (แดงพื้นจาง) — pill radius 99px
- **pagination**: `Showing X to Y of Z entries` + ปุ่มหน้า

### 5.7 Add Card (เส้นประ)
- การ์ดพื้นจาง เส้นประ `--border`, กลางมีไอคอน `+` วงกลมฟ้า + ข้อความ "เพิ่มกล้อง"
- hover: เส้น/ไอคอนเข้มขึ้น

### 5.8 Rule Builder (node graph) — เฟสหลัง
- พาเนลซ้าย: รายการ blocks (Camera / Person / PPE / Zone / Distance / Notification)
- แคนวาส: ลากบล็อกต่อด้วยเส้น → เงื่อนไข → แอคชัน (Notification/Report/Line)
- *ซับซ้อน — แนะนำทำทีหลังสุด หรือทำเวอร์ชันง่าย (ฟอร์มสร้างกฎ) ก่อน*

---

## 6. Mapping → หน้าจอ ZENTRA

| Protex | ZENTRA screen | งานที่ต้องทำ |
|--------|---------------|--------------|
| Dashboard | `dashboard.html` | ปรับ KPI เป็น stat card + เพิ่ม **Donut Safety Score** + event toast บนวิดีโอ |
| Cameras | `source.html` (+ ใหม่ `cameras`) | ตารางกล้อง + add-card + status badge (ตอนนี้กล้องเดียว → เริ่มจากการ์ดเดียว) |
| Rule Builder | `zone_editor.html` | เฟสหลัง — ตอนนี้คงตัว zone editor เดิม |
| Reports | `history.html` | ปรับเป็นตารางสไตล์ Protex + avatar + badge + pagination |
| (Settings) | `settings.html` | คงโครง group เดิม แต่ใช้โทเคน/การ์ดชุดใหม่ |
| Topbar/Shell | `app.js` (sidebar) | sidebar → icon rail 60px + เพิ่ม **topbar** ใหม่ |

---

## 7. แผนพัฒนาทีละส่วน (Phased — ทำ + เทสต์ทีละชิ้น)

| เฟส | งาน | ไฟล์ |
|-----|-----|------|
| **2a** ✅ | Sidebar SVG icons | `app.js`, `style.css` |
| **2b** | ตัดสินใจธีม + วางโทเคน light/dark สลับได้ | `style.css :root` |
| **2c** | Sidebar → **icon rail 60px** + tooltip | `app.js`, `style.css` |
| **2d** | **Topbar** (location + ภาษา + user + 🔔) | `app.js`, ทุก screen |
| **2e** | **KPI stat cards** + **Donut Safety Score** | `dashboard.html`, `style.css` |
| **2f** | **Event toast** บนวิดีโอ + Toast system | `app.js`, `dashboard.html` |
| **2g** | **Data table component** (ใช้ที่ History/Cameras) | `style.css`, `history.html` |
| **2h** | หน้า **Cameras** (การ์ด + ตาราง + add-card) | screen ใหม่ |
| **2i** | Footer (โลโก้ + copyright) | shell |
| **2j** | Rule Builder (node graph) | `zone_editor.html` — ท้ายสุด |

---

## 8. ช่องว่างเทียบ ZENTRA ปัจจุบัน (Gaps)
- Protex มี **หลายกล้อง + add camera** → ZENTRA ตอนนี้กล้องเดียว (ออกแบบเผื่อหลายกล้องได้)
- Protex มี **multi-user + avatar/อีเมล** → ZENTRA ยังไม่มีระบบ user (ใส่ placeholder ได้)
- Protex มี **Rule Builder graph** → ZENTRA มีแค่ zone editor (กฎ PPE/Zone hardcoded)
- Protex เป็น **web SaaS หลายผู้ใช้** → ZENTRA เป็น **desktop app เครื่องเดียว** (PDPA, on-device) — ปรับ "location/user" เป็น context ของเครื่องนั้น

---

## 9. หมายเหตุเทคนิค (vanilla stack — ต้องระวัง)
- สคริปต์ใน **screen .html ใช้ `var` ไม่ใช่ `let/const`** (WebView2 re-navigation redeclare error)
- `innerHTML` ไม่รัน `<script>` อัตโนมัติ — ต้อง re-create element (มี handler ใน `app.js` แล้ว)
- `cv2.putText` วาดภาษาไทยไม่ได้ → overlay บนวิดีโอใช้ ASCII/อังกฤษ (event card ที่เป็น HTML ใช้ไทยได้)
- ใช้ **CSS variables** เป็นหลัก → สลับธีม light/dark = แก้ `:root` ที่เดียว
- icon = inline SVG ผ่าน `ZENTRA.icon(name)` (มีระบบแล้วใน `app.js`)

---

*อ้างอิง: Protex AI product screenshots (Dashboard / Cameras / Rule Builder / Reports). ใช้เป็นแนวภาษาภาพ — ปรับให้เข้ากับบริบท desktop/on-device ของ ZENTRA*
