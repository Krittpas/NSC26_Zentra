"""
server/report.py — ZENTRA safety report (local PDF via matplotlib)

Builds an A4 safety report from the local event store, formatted to the Thai
official-document convention: national font TH Sarabun New, 1-inch margins on
all sides, 20/18/16 pt hierarchy (title / heading / body), and page numbers.
Content: header/identity, KPI summary, severity breakdown, trend chart, evidence
gallery, event log with a corrective-action column, and a signature block.

Font: prefer TH Sarabun New (installed on Thai machines); fall back to the
bundled OFL Sarabun so Thai still renders where it isn't (e.g. the container).
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")                      # headless backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

from server import store

_REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"
_BUNDLED_FONT = Path(__file__).parent.parent / "backend" / "assets" / "fonts" / "Sarabun-SemiBold.ttf"

_FONT_READY = False

_LVL_COLOR = {"warning": "#ea580c", "alert": "#2563eb", "emergency": "#dc2626"}
_LVL_TH    = {"warning": "เตือน (PPE)", "alert": "อันตราย (เขต)", "emergency": "ฉุกเฉิน"}
_TYPE_TH   = {"ppe": "PPE", "zone": "เขตหวงห้าม", "fall": "การล้ม", "heat": "ความร้อน"}

# ── A4 geometry + type scale (Thai official-document style) ──────────────────
FIG_W, FIG_H = 8.27, 11.69                 # A4 portrait (inches)
_MARGIN = 1.0                               # 1-inch margins on all sides
FS_TITLE, FS_HEAD, FS_BODY, FS_SMALL = 20, 18, 16, 14   # pt

# Per request: ALL letter/text is black. Colour is reserved for graphics only
# (chart bars, severity meters). Hierarchy comes from size + weight, not colour.
INK  = "#000000"
GRID = "#cdd6e3"                            # graphic hairlines (dividers/axes)


def _X(inch: float) -> float:
    return inch / FIG_W


def _Y(inch_from_top: float) -> float:
    return 1 - inch_from_top / FIG_H


_EMOJI_LEAD = re.compile(
    r"^[\U0001F000-\U0001FAFF☀-➿⬀-⯿️←-⇿\s]+")


def _clean_msg(msg: Optional[str]) -> str:
    """Drop a leading emoji/symbol (e.g. '⚠️ ') from an event message."""
    return _EMOJI_LEAD.sub("", msg or "")


def _fmt_date(s: Optional[str]) -> str:
    """'YYYY-MM-DD' → 'DD-MM-YYYY' (วัน-เดือน-ปี)."""
    try:
        y, m, d = (s or "").split("-")
        return f"{d}-{m}-{y}"
    except Exception:
        return s or ""


def _page_number(fig, n: int) -> None:
    fig.text(0.5, _Y(FIG_H - 0.42), f"- {n} -", fontsize=FS_SMALL,
             color="#475569", ha="center", va="center")


def _ensure_thai_font() -> None:
    """Register the report font, preferring TH Sarabun New."""
    global _FONT_READY
    if _FONT_READY:
        return

    def _try_system(family: str) -> Optional[str]:
        try:
            reg = fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
        except Exception:
            return None
        if not reg or not Path(reg).exists():
            return None
        fm.fontManager.addfont(reg)
        # register the bold face too so fontweight="bold" is real, not faux
        try:
            b = fm.findfont(fm.FontProperties(family=family, weight="bold"),
                            fallback_to_default=False)
            if b and Path(b).exists() and b != reg:
                fm.fontManager.addfont(b)
        except Exception:
            pass
        return fm.FontProperties(fname=reg).get_name()

    name = None
    for fam in ("TH Sarabun New", "TH SarabunPSK"):     # Thai national font first
        name = _try_system(fam)
        if name:
            break
    if not name and _BUNDLED_FONT.exists():             # bundled OFL fallback
        try:
            fm.fontManager.addfont(str(_BUNDLED_FONT))
            name = fm.FontProperties(fname=str(_BUNDLED_FONT)).get_name()
        except Exception:
            name = None
    if not name:                                         # last resort
        for fam in ("Sarabun", "Leelawadee UI", "Tahoma", "Angsana New"):
            name = _try_system(fam)
            if name:
                break

    if name:
        matplotlib.rcParams["font.family"] = name
    matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def _safety_index(total: int) -> int:
    """Estimated safety index 0–100 (fewer confirmed incidents → higher)."""
    return max(0, 100 - total * 4)


def build_daily_pdf(day: Optional[str] = None, start: Optional[str] = None,
                    end: Optional[str] = None, org: Optional[dict] = None) -> Path:
    """Render the safety-report PDF and return its local path.
    Single day (`day`) or an inclusive range (`start`,`end`)."""
    _ensure_thai_font()
    org = org or {}
    is_range = bool(start and end)
    if not is_range:
        day = day or date.today().strftime("%Y-%m-%d")
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stats  = store.today_stats(day=day, start=start, end=end)
    events = store.list_events(limit=100000, offset=0, day=day, start=start, end=end)["events"]
    period = (f"ช่วงวันที่ {_fmt_date(start)} ถึง {_fmt_date(end)}"
              if is_range else f"วันที่ {_fmt_date(day)}")

    L = _MARGIN                 # left content edge (inches)
    R = FIG_W - _MARGIN         # right content edge (inches) = 7.27
    W = _X(R) - _X(L)           # content width as a figure fraction

    company  = org.get("company") or "ZENTRA Industrial Safety"
    site     = org.get("site") or "-"
    preparer = org.get("preparer") or "-"

    tag = f"{start}_{end}" if is_range else day
    out = _REPORTS_DIR / f"zentra_report_{tag}.pdf"
    dbg_png = os.getenv("ZENTRA_REPORT_PNG")   # dev-only: also dump PNGs to eyeball

    with PdfPages(str(out)) as pdf:
        # ═══════════════════ PAGE 1 — summary ═══════════════════
        fig = plt.figure(figsize=(FIG_W, FIG_H)); fig.patch.set_facecolor("white")
        yt = L

        fig.text(_X(L), _Y(yt), "รายงานความปลอดภัย", fontsize=FS_TITLE,
                 fontweight="bold", color=INK, va="top"); yt += 0.48
        fig.text(_X(L), _Y(yt), company, fontsize=FS_BODY, color=INK, va="top"); yt += 0.32
        fig.text(_X(L), _Y(yt), f"สถานที่: {site}    ·    {period}",
                 fontsize=FS_BODY, color=INK, va="top"); yt += 0.32
        fig.text(_X(L), _Y(yt),
                 f"ผู้จัดทำ: {preparer}    ·    ออกรายงาน: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
                 fontsize=FS_SMALL, color=INK, va="top"); yt += 0.30
        fig.add_artist(plt.Line2D([_X(L), _X(R)], [_Y(yt), _Y(yt)], color=GRID, lw=1)); yt += 0.34

        # KPI row (value 18 bold, small caption underneath)
        idx = _safety_index(stats["total"])
        kpis = [
            ("ดัชนีความปลอดภัย", f"{idx}",
             "#16a34a" if idx >= 80 else ("#d97706" if idx >= 60 else "#dc2626")),
            ("เหตุการณ์รวม", str(stats["total"]),        "#2563eb"),
            ("PPE", str(stats["ppe_violations"]),        "#ea580c"),
            ("เข้าเขต", str(stats["zone_intrusions"]),    "#2563eb"),
            ("ฉุกเฉิน/ล้ม", str(stats["emergency"]),      "#dc2626"),
        ]
        colw = (R - L) / len(kpis)
        for i, (label, val, _color) in enumerate(kpis):
            cx = L + i * colw
            fig.text(_X(cx), _Y(yt), val, fontsize=FS_HEAD, fontweight="bold", color=INK, va="top")
            fig.text(_X(cx), _Y(yt + 0.36), label, fontsize=12.5, color=INK, va="top")
        yt += 0.92
        fig.add_artist(plt.Line2D([_X(L), _X(R)], [_Y(yt), _Y(yt)], color=GRID, lw=1)); yt += 0.36

        # Severity breakdown — text black; the meter bar carries the colour
        fig.text(_X(L), _Y(yt), "สรุปตามระดับความรุนแรง", fontsize=FS_HEAD,
                 fontweight="bold", color=INK, va="top"); yt += 0.46
        sev = {"warning": 0, "alert": 0, "emergency": 0}
        for e in events:
            if e["level"] in sev:
                sev[e["level"]] += 1
        total_sev = max(1, len(events))
        for lvl in ("emergency", "alert", "warning"):
            c = _LVL_COLOR[lvl]
            fig.text(_X(L + 0.1), _Y(yt), _LVL_TH[lvl], fontsize=FS_BODY, color=INK, va="top")
            bx = _X(L + 2.6); bw = _X(R - 0.7) - bx; by = _Y(yt + 0.13); bh = 0.014
            fig.add_artist(plt.Rectangle((bx, by), bw, bh, color="#eef2f7"))
            fig.add_artist(plt.Rectangle((bx, by), max(0.004, bw * sev[lvl] / total_sev), bh, color=c))
            fig.text(_X(R), _Y(yt), str(sev[lvl]), fontsize=FS_BODY, fontweight="bold",
                     color=INK, ha="right", va="top")
            yt += 0.42
        yt += 0.18

        # Trend chart. Axes: x = time bucket, y = number of events.
        fig.text(_X(L), _Y(yt), ("แนวโน้มเหตุการณ์รายวัน" if is_range else "แนวโน้มเหตุการณ์รายชั่วโมง"),
                 fontsize=FS_HEAD, fontweight="bold", color=INK, va="top"); yt += 0.42
        ch_h = 2.1                                  # chart height (inches)
        ch_left = L + 0.55                          # room on the left for the y-axis label
        ax = fig.add_axes([_X(ch_left), _Y(yt + ch_h), _X(R) - _X(ch_left), ch_h / FIG_H])
        if is_range:
            counts = store.daily_counts(start=start, end=end)
            labels = list(counts.keys()); values = [counts[k] for k in labels]
            ax.bar(range(len(labels)), values, color="#2563eb", width=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([l[5:] for l in labels], fontsize=12, color=INK,
                               rotation=45, ha="right")
            ax.set_xlabel("วันที่ (เดือน-วัน)", fontsize=12, color=INK)
        else:
            hourly = store.hourly(day); hours = [f"{h:02d}" for h in range(24)]
            values = [hourly.get(h, 0) for h in hours]
            ax.bar(range(24), values, color="#2563eb", width=0.7)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=12, color=INK)
            ax.set_xlabel("เวลา (นาฬิกา 00–23)", fontsize=12, color=INK)
        ax.set_ylabel("จำนวนเหตุการณ์ (ครั้ง)", fontsize=12, color=INK)
        ax.tick_params(axis="y", labelsize=12, colors=INK)
        if max(values, default=0) <= 5:
            ax.set_ylim(0, 5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="y", color="#e2e8f0", lw=0.6)

        _page_number(fig, 1)
        if dbg_png:
            fig.savefig(dbg_png + "_p1.png", dpi=130)
        pdf.savefig(fig); plt.close(fig)

        # ═══════════════════ PAGE 2 — detail ═══════════════════
        fig = plt.figure(figsize=(FIG_W, FIG_H)); fig.patch.set_facecolor("white")
        yt = L
        fig.text(_X(L), _Y(yt), f"รายงานความปลอดภัย · {period}",
                 fontsize=FS_SMALL, color=INK, va="top"); yt += 0.30
        fig.add_artist(plt.Line2D([_X(L), _X(R)], [_Y(yt), _Y(yt)], color=GRID, lw=1)); yt += 0.36

        # Evidence gallery (up to 5)
        snaps = [e for e in events if e.get("has_snapshot")][:5]
        if snaps:
            fig.text(_X(L), _Y(yt), "ตัวอย่างภาพหลักฐาน", fontsize=FS_HEAD,
                     fontweight="bold", color=INK, va="top"); yt += 0.42
            th_w, th_h, gap = 1.15, 0.82, 0.12
            for i, e in enumerate(snaps):
                p = store.snapshot_path(e["id"])
                if not p:
                    continue
                try:
                    img = plt.imread(str(p))
                except Exception:
                    continue
                cx = L + i * (th_w + gap)
                axx = fig.add_axes([_X(cx), _Y(yt + th_h), th_w / FIG_W, th_h / FIG_H])
                axx.imshow(img); axx.set_xticks([]); axx.set_yticks([])
                for sp in axx.spines.values():
                    sp.set_color(GRID)
                axx.set_title(e.get("time", ""), fontsize=11, color=INK)
            yt += th_h + 0.5

        # Event log (with a manual corrective-action column)
        fig.text(_X(L), _Y(yt), "บันทึกเหตุการณ์", fontsize=FS_HEAD,
                 fontweight="bold", color=INK, va="top"); yt += 0.44
        c_time, c_type, c_detail, c_action = L, 2.2, 3.3, 5.7
        for cx, head in ((c_time, "เวลา"), (c_type, "ประเภท"),
                         (c_detail, "รายละเอียด"), (c_action, "การแก้ไข/ผู้รับผิดชอบ")):
            fig.text(_X(cx), _Y(yt), head, fontsize=FS_SMALL, fontweight="bold",
                     color=INK, va="top")
        yt += 0.30
        fig.add_artist(plt.Line2D([_X(L), _X(R)], [_Y(yt), _Y(yt)], color=GRID, lw=0.8)); yt += 0.14

        shown = 8    # cap so the signature block always fits under the table
        if not events:
            fig.text(_X(L + 0.1), _Y(yt), "— ไม่มีเหตุการณ์ในช่วงนี้ —",
                     fontsize=FS_BODY, color=INK, va="top"); yt += 0.36
        else:
            for e in events[:shown]:
                fig.text(_X(c_time), _Y(yt), f"{_fmt_date(e.get('date',''))[:5]} {e['time']}",
                         fontsize=FS_BODY, color=INK, va="top")
                fig.text(_X(c_type), _Y(yt), _TYPE_TH.get(e["type"], e["type"]),
                         fontsize=FS_BODY, fontweight="bold", color=INK, va="top")
                fig.text(_X(c_detail), _Y(yt), _clean_msg(e["message"])[:20],
                         fontsize=FS_BODY, color=INK, va="top")
                # blank write-in line for the corrective action
                fig.add_artist(plt.Line2D([_X(c_action), _X(R)], [_Y(yt + 0.24), _Y(yt + 0.24)],
                               color="#e2e8f0", lw=0.8))
                yt += 0.36
            if len(events) > shown:
                fig.text(_X(L), _Y(yt), f"… และอีก {len(events) - shown} เหตุการณ์ (ดูไฟล์ CSV)",
                         fontsize=FS_SMALL, color=INK, va="top"); yt += 0.34

        # Signature block — layout 2 + 1: (ผู้จัดทำรายงาน | ผู้ตรวจสอบ (จป.)) on one
        # row, a blank line, then ผู้อนุมัติ centred. Each box uses the standard Thai
        # form style: a dotted sign line ending in the role, then a name in ().
        def _sign_box(cx_in: float, role: str) -> None:
            fig.text(_X(cx_in), _Y(yt),        f"ลงชื่อ  {'.' * 20}  {role}",
                     fontsize=FS_BODY, color=INK, ha="center", va="top")
            fig.text(_X(cx_in), _Y(yt + 0.36), f"({'.' * 34})",
                     fontsize=FS_BODY, color=INK, ha="center", va="top")
            fig.text(_X(cx_in), _Y(yt + 0.72), "วันที่ ....... /....... /.......",
                     fontsize=FS_BODY, color=INK, ha="center", va="top")

        yt = max(yt + 0.5, 7.6)
        half = (R - L) / 2
        _sign_box(L + half / 2,        "ผู้จัดทำรายงาน")      # row 1 · left
        _sign_box(L + half + half / 2, "ผู้ตรวจสอบ (จป.)")   # row 1 · right
        yt += 1.30                                            # row 1 (3 lines) + blank line
        _sign_box((L + R) / 2,         "ผู้อนุมัติ")          # row 2 · centred
        _page_number(fig, 2)
        if dbg_png:
            fig.savefig(dbg_png + "_p2.png", dpi=130)
        pdf.savefig(fig); plt.close(fig)

    return out


def daily_stats_for_line(day: Optional[str] = None) -> dict:
    """Stats dict shaped for alerts.line_notify.send_daily_report (text only)."""
    s = store.today_stats(day)
    return {
        "ppe_violations":  s["ppe_violations"],
        "zone_intrusions": s["zone_intrusions"],
        "fall_events":     s["falls"],
    }
