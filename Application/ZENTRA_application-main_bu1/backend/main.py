#!/usr/bin/env python3.11
# main.py — ZENTRA System Entry Point
# Windows 11 + NVIDIA GPU | Python 3.11
# Slide: IP Camera → RTSP → Edge Unit → 3 Modules → LINE OA
# ================================================================
# Run:   python main.py
# Help:  กด H ในหน้าต่างวิดีโอ
# ================================================================

from __future__ import annotations
import cv2, sys, time, queue, threading, traceback
import numpy as np
from pathlib import Path

if sys.version_info < (3, 10):
    sys.exit("❌ Python 3.10+ required (แนะนำ 3.11)")

# Windows console / redirected output defaults to cp1252, which crashes on the
# emoji used in log messages (UnicodeEncodeError). Force UTF-8 so logs never
# bring the app down regardless of where stdout goes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config as cfg
from alerts.line_notify   import start_sender, stop_sender
from reports.daily_report import get_logger, get_scheduler
from utils.collector      import get_collector
import modules.ppe         as ppe_module
import modules.safety_zone as zone_module
import modules.heat_stroke as fall_module


# ================================================================
# INFERENCE CLIENT
# ================================================================
def _make_client():
    try:
        from inference_sdk import InferenceHTTPClient

        # เลือก model ID (local fine-tuned หรือ Roboflow)
        ppe_id  = cfg.PPE_LOCAL_MODEL  if cfg.USE_LOCAL_MODEL and Path(cfg.PPE_LOCAL_MODEL).exists()  else cfg.PPE_MODEL_ID
        fall_id = cfg.FALL_LOCAL_MODEL if cfg.USE_LOCAL_MODEL and Path(cfg.FALL_LOCAL_MODEL).exists() else cfg.FALL_MODEL_ID

        client = InferenceHTTPClient(api_url=cfg.INFERENCE_SERVER_URL, api_key=cfg.ROBOFLOW_API_KEY)
        print(f"[Main] Inference server : {cfg.INFERENCE_SERVER_URL}")
        print(f"[Main] PPE  model       : {ppe_id}")
        print(f"[Main] Fall model       : {fall_id}")
        return client, ppe_id, fall_id

    except Exception as e:
        print(f"[Main] ⚠️  inference_sdk unavailable: {e}")
        print("[Main]    pip install inference-sdk")
        return None, cfg.PPE_MODEL_ID, cfg.FALL_MODEL_ID


# ================================================================
# CAMERA — Windows 11: DirectShow (USE_DSHOW=true) เร็วกว่า
# ================================================================
def open_camera() -> cv2.VideoCapture:
    src = cfg.CAMERA_SOURCE
    if src == "webcam":
        backend = cv2.CAP_DSHOW if cfg.USE_DSHOW else cv2.CAP_ANY
        cap     = cv2.VideoCapture(cfg.WEBCAM_INDEX, backend)
        print(f"[Main] 📷 Webcam index={cfg.WEBCAM_INDEX}  backend={'DSHOW' if cfg.USE_DSHOW else 'AUTO'}")
    elif src == "rtsp":
        cap = cv2.VideoCapture(cfg.RTSP_URL, cv2.CAP_FFMPEG)
        print(f"[Main] 📡 RTSP: {cfg.RTSP_URL}")
    elif src == "file":
        cap = cv2.VideoCapture(cfg.VIDEO_FILE_PATH)
        print(f"[Main] 🎬 File: {cfg.VIDEO_FILE_PATH}")
    else:
        raise ValueError(f"CAMERA_SOURCE ไม่รู้จัก: {src}")

    if not cap.isOpened():
        raise RuntimeError(f"เปิดกล้องไม่ได้ (source={src})")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.FRAME_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS, cfg.TARGET_FPS)
    return cap


# ================================================================
# ASYNC FRAME READER
# ================================================================
class FrameReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True, name="FrameReader")
        self.cap   = cap
        self.q     = queue.Queue(maxsize=cfg.FRAME_BUFFER_SIZE)
        self._stop = threading.Event()
        self.error_count = 0

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.error_count += 1
                if self.error_count > 30:
                    print("[FrameReader] ⚠️  Too many read errors")
                time.sleep(0.05)
                continue
            self.error_count = 0
            if self.q.full():
                try:
                    self.q.get_nowait()   # drop oldest
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return True, self.q.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def stop(self):
        self._stop.set()


# ================================================================
# ASYNC INFERENCE WORKER
# ================================================================
class InferenceWorker(threading.Thread):
    def __init__(self, client, ppe_id: str, fall_id: str):
        super().__init__(daemon=True, name="InferenceWorker")
        self.client  = client
        self.ppe_id  = ppe_id
        self.fall_id = fall_id
        self.in_q    = queue.Queue(maxsize=2)
        self.out_q   = queue.Queue(maxsize=4)
        self._stop   = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                fid, frame = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                ppe_preds, fall_preds = [], []
                if self.client:
                    r1         = self.client.infer(frame, model_id=self.ppe_id)
                    ppe_preds  = r1.get("predictions", [])
                    r2         = self.client.infer(frame, model_id=self.fall_id)
                    fall_preds = r2.get("predictions", [])
                self.out_q.put((fid, ppe_preds, fall_preds))
            except Exception as e:
                print(f"[Inference] Error: {e}")

    def submit(self, fid: int, frame: np.ndarray):
        if not self.in_q.full():
            self.in_q.put((fid, frame.copy()))

    def get_result(self):
        try:
            return self.out_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()


# ================================================================
# KEYBOARD HANDLER
# ================================================================
def handle_key(key: int) -> bool:
    """return True = ควรหยุด"""
    if key == ord("q"):
        print("\n[Main] ⏹️  Q pressed — stopping...")
        return True
    elif key == ord("z"):
        zone_module.toggle_draw_mode()
    elif key == ord("c"):
        zone_module.clear_all_zones()
    elif key == ord("t"):
        _bg_train()
    elif key == ord("s"):
        _show_stats()
    elif key == ord("h"):
        _show_help()
    return False


def _show_help():
    print("\n[Main] ─── Controls ───────────────────────")
    print("  Q  หยุดระบบ")
    print("  Z  วาด Red Zone ใหม่")
    print("  C  ลบ Zone ทั้งหมด")
    print("  T  เริ่ม Training ใน background")
    print("  S  แสดงสถิติ")
    print("  H  แสดงความช่วยเหลือนี้")
    print("─────────────────────────────────────────\n")


def _show_stats():
    today = get_logger().get_today()
    col   = get_collector()
    print("\n[Main] ─── สถิติ ─────────────────────────")
    print(f"  PPE Violations  : {today.get('ppe_violations', 0)}")
    print(f"  Zone Intrusions : {today.get('zone_intrusions', 0)}")
    print(f"  Fall Events     : {today.get('fall_events', 0)}")
    print(f"  Frames          : {ppe_module.stats['frames']:,}")
    print(f"  FPS เฉลี่ย      : {ppe_module.get_fps():.1f}")
    print(f"  Collected data  : {col.get_stats()}")
    print("─────────────────────────────────────────\n")


def _bg_train():
    def _run():
        print("\n[Training] 🚀 Starting background training...")
        try:
            from training.trainer import run_training_pipeline
            run_training_pipeline(task="ppe", augment=True)
        except Exception as e:
            print(f"[Training] ❌ {e}")
    threading.Thread(target=_run, daemon=True, name="Trainer").start()
    print("[Main] Training started in background (กด S เพื่อดูสถิติ)")


# ================================================================
# METADATA
# ================================================================
class _Meta:
    __slots__ = ("frame_id",)
    def __init__(self, fid: int): self.frame_id = fid


# ================================================================
# MAIN LOOP
# ================================================================
def main():
    print("=" * 60)
    print("  ZENTRA — Zone Environment Network Thermal Risk Analysis")
    print(f"  OS      : Windows 11")
    print(f"  Camera  : {cfg.CAMERA_SOURCE}")
    print(f"  Server  : {cfg.INFERENCE_SERVER_URL}")
    print("=" * 60)

    # ── subsystems ──────────────────────────────────────────
    start_sender()
    scheduler = get_scheduler(); scheduler.start()
    logger    = get_logger()
    collector = get_collector()

    # ── inference ───────────────────────────────────────────
    client, ppe_id, fall_id = _make_client()

    # ── camera ──────────────────────────────────────────────
    cap    = open_camera()
    reader = FrameReader(cap)
    if cfg.ENABLE_THREADING:
        reader.start()

    # ── inference worker ────────────────────────────────────
    inf_worker = InferenceWorker(client, ppe_id, fall_id)
    if cfg.ENABLE_THREADING:
        inf_worker.start()

    # ── window (Windows 11) ─────────────────────────────────
    cv2.namedWindow(cfg.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(cfg.WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(cfg.WINDOW_TITLE, cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT)
    cv2.setMouseCallback(cfg.WINDOW_TITLE, zone_module.mouse_callback)

    # ── loop state ──────────────────────────────────────────
    frame_id        = 0
    last_ppe_preds: list = []
    last_fall_preds: list = []

    _show_help()
    print("✅ ระบบพร้อมทำงาน\n")

    try:
        while True:
            # ── read frame ──────────────────────────────────
            if cfg.ENABLE_THREADING:
                ret, raw = reader.read()
            else:
                ret, raw = cap.read()

            if not ret or raw is None:
                time.sleep(0.04)
                continue

            frame_id += 1
            # Mirror สำหรับ webcam (ถ้า RTSP ปิด flip ได้ใน .env)
            if cfg.CAMERA_SOURCE == "webcam":
                raw = cv2.flip(raw, 1)

            # ── submit inference ─────────────────────────────
            if frame_id % cfg.INFER_EVERY_N_FRAMES == 0:
                if cfg.ENABLE_THREADING:
                    inf_worker.submit(frame_id, raw)
                elif client:
                    try:
                        r1 = client.infer(raw, model_id=ppe_id)
                        last_ppe_preds  = r1.get("predictions", [])
                        r2 = client.infer(raw, model_id=fall_id)
                        last_fall_preds = r2.get("predictions", [])
                    except Exception as e:
                        print(f"[Inference] {e}")

            # ── collect results ──────────────────────────────
            if cfg.ENABLE_THREADING:
                res = inf_worker.get_result()
                if res:
                    _, last_ppe_preds, last_fall_preds = res

            # ── annotate ─────────────────────────────────────
            annotated = raw.copy()
            annotated = ppe_module.draw_predictions(annotated, last_ppe_preds)
            annotated = fall_module.draw_fall_predictions(annotated, last_fall_preds)

            meta = _Meta(frame_id)

            # ── Module 3: Heat Stroke / Pose (MediaPipe) ────
            fall_module.on_frame(annotated, meta, cfg.WINDOW_TITLE)

            # ── Module 2: Safety Zone (tracks overlay) ───────
            zone_module.on_frame(annotated, meta, cfg.WINDOW_TITLE)

            # ── Module 1: PPE (OSD + imshow) ─────────────────
            ppe_module.on_frame(annotated, meta, cfg.WINDOW_TITLE)

            # ── data callbacks ───────────────────────────────
            ppe_module.on_data( {"predictions": last_ppe_preds},  meta, frame=raw)
            zone_module.on_data({"predictions": last_ppe_preds},  meta, frame=raw)
            fall_module.on_data({"predictions": last_fall_preds}, meta, frame=raw)

            # ── daily logger ─────────────────────────────────
            if frame_id % 150 == 0:
                logger.update_frames(ppe_module.stats["frames"])

            # ── keyboard ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if handle_key(key):
                break

    except KeyboardInterrupt:
        print("\n[Main] ⏹️  Ctrl+C")
    except Exception:
        print("\n[Main] ❌ Unhandled error:")
        traceback.print_exc()
    finally:
        if cfg.ENABLE_THREADING:
            reader.stop()
            inf_worker.stop()
        stop_sender()
        scheduler.stop()
        cap.release()
        cv2.destroyAllWindows()
        _show_stats()
        print("=" * 60)
        print("  ZENTRA ปิดระบบเรียบร้อย")
        print("=" * 60)


if __name__ == "__main__":
    main()
