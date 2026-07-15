# reports/daily_report.py — ZENTRA Daily Safety Report Generator
# ส่งรายงานอัตโนมัติทุกวันเวลา 20:00 น. ผ่าน LINE OA
# รองรับ: trend graph, heatmap, ISO 45001 log

from __future__ import annotations
import cv2
import json
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional


def _cfg():
    import config as c
    return c


# ──────────────────────────────────────────────────────────────
# Stats Logger
# ──────────────────────────────────────────────────────────────
class StatsLogger:
    """
    บันทึกสถิติรายวันลง JSON file
    data/logs/stats_YYYYMMDD.json
    """

    def __init__(self):
        self.cfg      = _cfg()
        self.log_dir  = Path(self.cfg.LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._today   = self._date_str()
        self._data    = self._load_today()
        self._lock    = threading.Lock()

    def _date_str(self, offset_days: int = 0) -> str:
        return (datetime.now() + timedelta(days=offset_days)).strftime("%Y%m%d")

    def _log_path(self, date_str: str) -> Path:
        return self.log_dir / f"stats_{date_str}.json"

    def _load_today(self) -> dict:
        path = self._log_path(self._today)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {
            "date":            self._today,
            "ppe_violations":  0,
            "zone_intrusions": 0,
            "fall_events":     0,
            "frames":          0,
            "alerts_sent":     0,
            "timestamps":      [],       # [(time_str, event_type, detail)]
        }

    def _save(self):
        path = self._log_path(self._today)
        path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2))

    def log_event(self, event_type: str, detail: str = ""):
        """บันทึก event (ppe | zone | fall)"""
        with self._lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self._data[f"{event_type}"] = self._data.get(event_type, 0) + 1
            self._data["timestamps"].append([ts, event_type, detail])
            if len(self._data["timestamps"]) > 500:
                self._data["timestamps"] = self._data["timestamps"][-500:]
            self._save()

    def update_frames(self, n: int):
        with self._lock:
            self._data["frames"] = n
            self._save()

    def get_today(self) -> dict:
        with self._lock:
            return dict(self._data)

    def get_week_trend(self) -> list[dict]:
        """ดึงข้อมูล 7 วันล่าสุด"""
        result = []
        for i in range(6, -1, -1):
            ds   = self._date_str(-i)
            path = self._log_path(ds)
            if path.exists():
                try:
                    result.append(json.loads(path.read_text()))
                except Exception:
                    pass
            else:
                result.append({
                    "date":            ds,
                    "ppe_violations":  0,
                    "zone_intrusions": 0,
                    "fall_events":     0,
                })
        return result

    def reset_day(self):
        """รีเซ็ตข้อมูลสำหรับวันใหม่"""
        with self._lock:
            self._today = self._date_str()
            self._data  = self._load_today()


# ──────────────────────────────────────────────────────────────
# Report Image Generator
# ──────────────────────────────────────────────────────────────
class ReportImageGenerator:
    """สร้างภาพ report สรุปสถิติ (ส่งผ่าน LINE)"""

    def __init__(self, width: int = 800, height: int = 500):
        self.w = width
        self.h = height

    def generate(self, today: dict, week: list[dict]) -> np.ndarray:
        img = np.ones((self.h, self.w, 3), dtype=np.uint8) * 25   # dark bg

        # Title
        cv2.putText(img, "ZENTRA — Daily Safety Report",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(img, datetime.now().strftime("%d/%m/%Y"),
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)
        cv2.line(img, (20, 80), (self.w - 20, 80), (80, 80, 80), 1)

        # Today stats
        stats_y = 115
        items = [
            ("PPE Violations",  today.get("ppe_violations",  0), (0, 80, 220)),
            ("Zone Intrusions", today.get("zone_intrusions", 0), (220, 80, 0)),
            ("Fall Events",     today.get("fall_events",     0), (0, 0, 220)),
            ("Total Frames",    today.get("frames",          0), (80, 200, 80)),
        ]
        for i, (label, val, color) in enumerate(items):
            x  = 20 + (i % 2) * 390
            y  = stats_y + (i // 2) * 80
            cv2.rectangle(img, (x, y), (x + 360, y + 65), (40, 40, 40), -1)
            cv2.rectangle(img, (x, y), (x + 360, y + 65), color, 2)
            cv2.putText(img, label, (x + 10, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(img, str(val), (x + 10, y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        # 7-day trend bar chart
        chart_y = 295
        chart_h = 160
        chart_x = 20
        chart_w = self.w - 40
        cv2.putText(img, "7-Day Trend",
                    (chart_x, chart_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        series = {
            "PPE": ([d.get("ppe_violations", 0)  for d in week], (0, 80, 220)),
            "Zone": ([d.get("zone_intrusions", 0) for d in week], (220, 80, 0)),
            "Fall": ([d.get("fall_events", 0)     for d in week], (0, 0, 220)),
        }
        max_val = max((max(v) for v, _ in series.values()), default=1)
        max_val = max(max_val, 1)

        bar_group_w = chart_w // len(week)
        n_series    = len(series)

        for gi, d in enumerate(week):
            gx = chart_x + gi * bar_group_w
            for si, (name, (values, color)) in enumerate(series.items()):
                bx  = gx + si * (bar_group_w // n_series) + 2
                bw  = bar_group_w // n_series - 4
                bh  = int((values[gi] / max_val) * (chart_h - 20))
                by  = chart_y + chart_h - bh
                cv2.rectangle(img, (bx, by), (bx + bw, chart_y + chart_h), color, -1)

            # X-axis label (day)
            date_str = d.get("date", "")
            label    = f"{date_str[6:8]}/{date_str[4:6]}" if len(date_str) == 8 else ""
            cv2.putText(img, label, (gx + 4, chart_y + chart_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

        # Legend
        leg_y = chart_y + chart_h + 35
        for i, (name, (_, color)) in enumerate(series.items()):
            lx = chart_x + i * 120
            cv2.rectangle(img, (lx, leg_y - 12), (lx + 18, leg_y), color, -1)
            cv2.putText(img, name, (lx + 22, leg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Footer
        cv2.putText(img, "ZENTRA | ISO 45001 Compliant | PDPA Safe",
                    (20, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        return img


# ──────────────────────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────────────────────
class DailyReportScheduler:
    """รันใน background thread ส่งรายงานตามเวลาที่กำหนด"""

    def __init__(self, logger: StatsLogger):
        self.logger    = logger
        self.generator = ReportImageGenerator()
        self._thread   = None
        self._running  = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="DailyReport")
        self._thread.start()
        print(f"[Report] Scheduler started — ส่งรายงานเวลา {_cfg().DAILY_REPORT_TIME} ทุกวัน")

    def stop(self):
        self._running = False

    def _loop(self):
        sent_today = False
        while self._running:
            now      = datetime.now()
            h, m     = _cfg().DAILY_REPORT_TIME.split(":")
            target   = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
            is_time  = (now.hour == int(h) and now.minute == int(m))

            if is_time and not sent_today:
                self._send_report()
                sent_today = True
            elif not is_time:
                sent_today = False

            time.sleep(30)

    def _send_report(self):
        from alerts.line_notify import send_daily_report
        today = self.logger.get_today()
        week  = self.logger.get_week_trend()
        img   = self.generator.generate(today, week)
        send_daily_report(today, img)

        # บันทึก image ลง disk
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(_cfg().REPORTS_DIR) / f"report_{ts}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"[Report] Report saved → {out_path}")

        # รีเซ็ตสถิติวันใหม่
        self.logger.reset_day()


# ──────────────────────────────────────────────────────────────
# Singleton instances
# ──────────────────────────────────────────────────────────────
_logger_instance:    Optional[StatsLogger]           = None
_scheduler_instance: Optional[DailyReportScheduler] = None


def get_logger() -> StatsLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StatsLogger()
    return _logger_instance


def get_scheduler() -> DailyReportScheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = DailyReportScheduler(get_logger())
    return _scheduler_instance
