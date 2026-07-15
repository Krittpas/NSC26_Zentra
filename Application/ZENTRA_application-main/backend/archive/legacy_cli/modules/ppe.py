# modules/ppe.py — ZENTRA PPE Detection Module
# Slide Module 1: YOLOv8 PPE | LINE alert
# Workflow: frame → Roboflow inference (:9001) → violation check → LINE alert
# Model in use: ppe-vum8g/2
#   worn:      hardhat / vest / gloves / boots   (+ person)
#   violation: no_hardhat / no_vest / no_gloves / no_boots
#   NOTE: this model has NO goggles/glasses class.
# ================================================================

from __future__ import annotations
import cv2
import time
import numpy as np
from typing import Optional

import config as cfg
from alerts.line_notify import send_line_notify as _send_line
from utils.collector    import get_collector

# ── State ───────────────────────────────────────────────────
stats = {
    "frames":      0,
    "violations":  0,
    "alerts_sent": 0,
    "start_time":  time.time(),
}
_last_alert: float = 0.0


# ================================================================
# HELPERS
# ================================================================
def get_fps() -> float:
    elapsed = time.time() - stats["start_time"]
    return stats["frames"] / elapsed if elapsed > 0 else 0.0


def _info(cls: str) -> dict:
    d = cfg.PPE_CLASSES
    if cls in d:                       # exact match — backward compatible
        return d[cls]
    # Normalise so class strings resolve regardless of casing or -/_/space form
    # (models emit "Vest", "no_glasses", "NO-Safety Vest", "no glasses", …).
    # Without this, unrecognised strings fall through to the non-violation
    # default and real violations are silently dropped.
    norm = " ".join(cls.lower().split()).replace("-", "_")
    for key in (norm, norm.replace("_", " "), norm.replace(" ", "_")):
        if key in d:
            return d[key]
    return {"label": cls, "label_th": cls, "color": (160, 160, 160), "violation": False}


def _conf_floor(cls: str) -> float:
    """Real confidence threshold applied in code. The inference server runs at a
    LOW floor (INFERENCE_SERVER_FLOOR) so PPE/Fall don't starve each other; the
    actual PPE cut-off is the global slider, with optional per-class overrides."""
    return cfg.PPE_CLASS_CONF.get(cls, cfg.INFERENCE_CONFIDENCE)


def _keep(pred: dict) -> bool:
    return pred.get("confidence", 0.0) >= _conf_floor(pred.get("class", ""))


# ================================================================
# DRAW PREDICTIONS
# ================================================================
def draw_predictions(frame: np.ndarray, predictions: list[dict]) -> np.ndarray:
    for pred in predictions:
        if not _keep(pred):
            continue
        cls  = pred.get("class", "")
        conf = pred.get("confidence", 0.0)
        x    = int(pred.get("x", 0))
        y    = int(pred.get("y", 0))
        w    = int(pred.get("width",  0))
        h    = int(pred.get("height", 0))

        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        info  = _info(cls)
        color = info["color"]
        label = f"{info['label']} {conf:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      cfg.FONT_SCALE, cfg.FONT_THICKNESS)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE,
                    (255, 255, 255), cfg.FONT_THICKNESS, cv2.LINE_AA)
    return frame


# Kept for compatibility with callers that expect a per-person status overlay.
# The clean per-person status view is not implemented for this model; boxes are
# drawn by draw_predictions above.
def draw_person_status(frame: np.ndarray, predictions, tracks=None) -> np.ndarray:
    return frame


# ================================================================
# ON_FRAME — OSD overlay + imshow (PPE is the LAST module → it shows the window)
# ================================================================
def on_frame(frame: np.ndarray, metadata, window_title: str = ""):
    stats["frames"] += 1
    fps = get_fps()

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), cfg.OSD_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    osd = (
        f"ZENTRA  |  FPS:{fps:.1f}  "
        f"Frames:{stats['frames']:,}  "
        f"Violations:{stats['violations']}  "
        f"Alerts:{stats['alerts_sent']}"
    )
    cv2.putText(frame, osd, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.FONT_SCALE,
                cfg.OSD_COLOR, cfg.FONT_THICKNESS, cv2.LINE_AA)
    if window_title:
        cv2.imshow(window_title, frame)


# ================================================================
# ON_DATA — violation detection + alert + data collection
# ================================================================
def on_data(data: dict, metadata, frame: Optional[np.ndarray] = None):
    global _last_alert

    raw_preds: list[dict] = (
        data.get("predictions") or
        data.get("detection_predictions") or []
    )
    # Apply the real confidence threshold before any logic / collection.
    predictions = [p for p in raw_preds if _keep(p)]

    # No detections → still collect a "normal" frame every N frames so the
    # dataset stays balanced (violation + normal, as requested).
    if not predictions:
        if frame is not None:
            get_collector().collect_normal(frame, [], metadata.frame_id)
        return

    detected   = [p.get("class", "") for p in predictions]
    violations = [c for c in detected if _info(c)["violation"]]

    if detected:
        print(f"[PPE] Frame {metadata.frame_id}: {detected}")

    # Compliant frame (people detected, no violation) → collect as normal.
    if not violations:
        if frame is not None:
            get_collector().collect_normal(frame, predictions, metadata.frame_id)
        return

    # ── Violation found ──────────────────────────────────────
    stats["violations"] += 1
    missing_th = ", ".join(sorted({_info(v)["label_th"] for v in violations}))
    missing_en = ", ".join(sorted({_info(v)["label"]    for v in violations}))
    print(f"[PPE] ⚠️  VIOLATION: {missing_en}")

    # Auto-collect the violation frame (image + YOLO pseudo-labels) for training.
    if frame is not None:
        get_collector().collect(frame, predictions, "ppe_violations")

    # LINE alert (engine handles its own cooldown via cooldown_key).
    now = time.time()
    if now - _last_alert >= cfg.VIOLATION_COOLDOWN_SECONDS:
        _last_alert = now
        stats["alerts_sent"] += 1
        msg = (
            f"🪖 ZENTRA PPE Alert\n"
            f"⚠️ ตรวจพบการไม่สวมอุปกรณ์ PPE:\n"
            f"   {missing_th}\n"
        )
        _send_line(
            msg,
            image        = frame,
            level        = cfg.ALERT_LEVEL_WARNING,
            cooldown_key = "ppe_violation",
            cooldown_sec = cfg.VIOLATION_COOLDOWN_SECONDS,
        )


# Compatibility re-export: some callers import send_line_notify from this module.
def send_line_notify(msg, image=None, level="warning", **kwargs):
    return _send_line(msg, image=image, level=level, **kwargs)
