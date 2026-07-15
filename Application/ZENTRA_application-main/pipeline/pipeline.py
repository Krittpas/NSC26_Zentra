"""
pipeline/pipeline.py — ZENTRA Camera Pipeline (passthrough, no detection)
Reads frames from a camera source, annotates nothing, and exposes them
for WebSocket broadcast.  Detection modules will be added one by one.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

_APP_DATA = Path(__file__).parent.parent / "data"
_APP_DATA.mkdir(exist_ok=True)

# Put the AI backend on the path so `config`, `utils.*`, `alerts.*` import here
# (single-repo: backend lives at <repo>/backend).
_BACKEND = Path(__file__).parent.parent / "backend"
if _BACKEND.is_dir() and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# ================================================================
# FRAME READER
# ================================================================
class _FrameReader(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture, loop: bool = False, src_fps: float = 0.0):
        super().__init__(daemon=True, name="FrameReader")
        self.cap   = cap
        # Small queue + drop-oldest (see _push): keep only the freshest 1-2 frames
        # so the detector/display always work on near-live frames instead of a
        # backlog. Larger buffers just add standing latency for a real-time feed.
        self.q     = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._loop = loop     # file sources: rewind at EOF so demo/testing never freezes
        # File sources decode instantly; throttle to the clip's native fps so the
        # frame stream stays time-ordered (a live camera self-paces, so src_fps=0
        # there). Without this, files replay at hundreds of fps and loop backwards,
        # which breaks ByteTrack's temporal continuity → track IDs churn → the
        # 3-of-5 zone/PPE confirm never accumulates.
        self._interval = (1.0 / src_fps) if src_fps and src_fps > 0 else 0.0

    def run(self):
        errors = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            ret, frame = self.cap.read()
            if not ret:
                if self._loop:
                    # EOF on a video file → seek back to the start and keep playing
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret:
                        errors = 0
                        self._push(frame)
                        self._throttle(t0)
                        continue
                errors += 1
                if errors > 30:
                    break
                time.sleep(0.05)
                continue
            errors = 0
            self._push(frame)
            self._throttle(t0)

    def _throttle(self, t0: float):
        if self._interval:
            dt = self._interval - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)

    def _push(self, frame):
        if self.q.full():
            try:
                self.q.get_nowait()
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
# PIPELINE
# ================================================================
class Pipeline:
    """Camera capture pipeline — passthrough (no AI detection yet)."""

    def __init__(self):
        self._lock        = threading.Lock()
        self._frame_lock  = threading.Lock()
        self._stop_evt    = threading.Event()
        self._running     = False

        self._latest_frame: Optional[np.ndarray] = None
        self._cap         = None
        self._reader      = None
        self._proc_thr    = None
        self._start_time: Optional[float] = None
        self._flip_override: Optional[bool] = None

        self.on_alert:  Optional[Callable[[str, str, bool], None]] = None
        self.on_status: Optional[Callable[[dict], None]] = None

        # Per-camera detection roles (None = all on) + per-level alert switches.
        self._roles: Optional[set] = None
        self._ppe_items: Optional[list] = None   # which PPE categories to enforce (None = cfg default)
        self._alert_levels: dict = {"warning": True, "alert": True, "emergency": True}

        self._source_config: dict = {}
        self._engine = None          # PPEEngine (lazy — built on start)
        # Decoupled detection: display loop runs at camera FPS (draws latest
        # boxes); a worker thread runs the slow inference on the newest frame.
        # (frame, frame_id) published as ONE tuple. Two separate fields could not
        # be read atomically: the detect loop read `raw` then `_raw_id`, and a new
        # frame landing between those two reads paired an old frame with a newer
        # id — advancing last_seen past a frame that was never processed.
        self._raw_pair: Optional[tuple] = None
        self._latest_recs: list = []
        self._recs_pair = None      # (frame, recs, frame_id) published together for the fall loop
        self._detect_thr = None
        self._fall_thr = None

        self._engine_error: Optional[str] = None   # why the engine failed to build

        self.status: dict = {
            "running":        False,
            "source":         None,
            "camera":         "disconnected",
            "modules":        {"ppe": "standby", "zone": "standby", "fall": "standby"},
            "alerts":         {"total": 0, "warning": 0, "alert": 0, "emergency": 0},
            "uptime_seconds": 0,
            "last_emergency": None,
            "engine_error":   None,
        }

    @staticmethod
    def _zero_alerts() -> dict:
        return {"total": 0, "warning": 0, "alert": 0, "emergency": 0}

    # ── Public API ────────────────────────────────────────────

    def start(self, source_config: dict) -> bool:
        if self._running:
            self.stop()
        self._stop_evt.clear()
        self._source_config = dict(source_config)
        try:
            self._apply_config(source_config)
            self._cap = self._open_camera(source_config)
        except Exception as e:
            print(f"[Pipeline] ❌ start failed: {e}")
            self._set_camera_state("disconnected")
            return False

        self._start_time = time.time()
        self._running    = True
        self._engine_error = None
        with self._lock:
            self.status["running"] = True
            self.status["source"]  = source_config.get("source", "webcam")
            self.status["modules"] = {"ppe": "standby", "zone": "standby", "fall": "standby"}
            # A new session starts from zero. History (SQLite) is the durable
            # record; these are live per-session counters and used to accumulate
            # across restarts, disagreeing with the History page.
            self.status["alerts"] = self._zero_alerts()
            self.status["engine_error"]   = None
            self.status["last_emergency"] = None
        self._set_camera_state("connected")

        self._proc_thr = threading.Thread(
            target=self._process_loop, daemon=True, name="PipelineLoop"
        )
        self._proc_thr.start()
        print(f"[Pipeline] ✅ Started (passthrough) — {source_config.get('source', 'webcam')}")
        return True

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_evt.set()
        if self._reader:
            try:
                self._reader.stop()
            except Exception:
                pass
        if self._fall_thr and self._fall_thr.is_alive():
            self._fall_thr.join(timeout=3.0)
        if self._engine is not None:
            try:
                self._engine.close_fall()      # release mediapipe/tflite graphs
            except Exception:
                pass
        if self._detect_thr and self._detect_thr.is_alive():
            self._detect_thr.join(timeout=3.0)
        if self._proc_thr and self._proc_thr.is_alive():
            self._proc_thr.join(timeout=3.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        # Uptime is "how long has this session been live". Leaving _start_time set
        # made get_uptime() keep counting after the camera stopped.
        self._start_time = None
        with self._lock:
            self.status["running"] = False
        self._set_camera_state("disconnected")
        print("[Pipeline] ⏹️  Stopped")

    def is_running(self) -> bool:
        return self._running

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_snapshot(self) -> Optional[bytes]:
        frame = self.get_latest_frame()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def get_uptime(self) -> int:
        if self._start_time is None:
            return 0
        return int(time.time() - self._start_time)

    def reload_zones(self):
        if self._engine is not None:
            try:
                self._engine.reload_zones()
                print("[Pipeline] 🗺️  zones reloaded")
            except Exception as e:
                print(f"[Pipeline] reload_zones: {e}")

    def _build_engine(self):
        """Build the PPE/zone engine (ultralytics + ByteTrack). On any failure,
        return None → the loop falls back to passthrough so the app never crashes.

        A passthrough pipeline shows clean video with no boxes, which reads as
        "nobody is violating anything". For a safety system that silent success is
        the worst possible failure, so the reason is recorded in status.engine_error
        and every module is marked "error" for the UI to shout about.
        """
        try:
            from utils.ppe_engine import PPEEngine
            cam_id = self._source_config.get("camera_id", "cam0")
            roles  = self._source_config.get("roles")
            self._roles = set(roles) if roles is not None else None
            self._ppe_items = self._source_config.get("ppe_items")
            eng = PPEEngine(zones_path=str(_APP_DATA / "zones.json"),
                            camera_id=cam_id, roles=self._roles,
                            ppe_items=self._ppe_items)
            import os as _os
            print(f"[Pipeline] 🧠 PPE engine ready ("
                  f"person={_os.path.basename(eng.person_detector.model_path)}, "
                  f"ppe={_os.path.basename(eng.ppe_detector.model_path) if eng.ppe_detector else 'off (no PPE model)'}, "
                  f"device={eng.detector.device}, camera={cam_id}, zones={len(eng.zones)}, "
                  f"roles={sorted(self._roles) if self._roles is not None else 'all'})")
            return eng
        except Exception as e:
            print(f"[Pipeline] ❌ engine unavailable → NO DETECTION. reason: {e}")
            traceback.print_exc()
            self._engine_error = str(e)
            with self._lock:
                self.status["engine_error"] = str(e)
                self.status["modules"] = {"ppe": "error", "zone": "error", "fall": "error"}
                snapshot = dict(self.status)
            if self.on_status:
                try:
                    self.on_status(snapshot)
                except Exception:
                    pass
            return None

    def _detect_loop(self):
        """Worker: run the heavy inference on the NEWEST raw frame, publish boxes
        (`_latest_recs`) for the display loop, and emit events. Always grabs the
        latest frame → auto-skips frames it can't keep up with (smooth video)."""
        last_seen = -1
        while not self._stop_evt.is_set() and self._running:
            pair = self._raw_pair            # one read → frame and id always agree
            if self._engine is None or pair is None or pair[1] == last_seen:
                time.sleep(0.005)
                continue
            raw, rid = pair
            last_seen = rid
            try:
                recs, events = self._engine.detect(raw)
                self._latest_recs = recs
                # Publish frame+boxes ATOMICALLY. The fall loop used to pair the
                # newest raw frame with these (older) boxes; during a fall the person
                # moves fast enough that the crop / keypoint match lands on the wrong
                # place. A slightly older frame that MATCHES its boxes is correct.
                self._recs_pair = (raw, recs, rid)
                for ev in events:
                    self._emit_event(ev)
            except Exception as e:
                print(f"[Pipeline] detect error: {e}")

    def _fall_loop(self):
        """Fall runs on its OWN fixed-cadence loop, not on the detect loop.

        The classifier consumes 30 evenly-spaced frames (~1-2 s of motion); the
        detect loop drops frames whenever inference lags, which would stretch that
        window unpredictably. Ticking at FALL_LOOP_FPS keeps dt uniform. The loop
        stays alive even while the role is off so it can pick the role up live."""
        try:
            import config as cfg
            interval = 1.0 / max(1, int(getattr(cfg, "FALL_LOOP_FPS", 15)))
        except ImportError:
            interval = 1.0 / 15
        overruns = 0
        last_fid = -1
        while not self._stop_evt.is_set() and self._running:
            t0 = time.monotonic()
            eng = self._engine
            if eng is not None and getattr(eng, "fall_ready", False):
                pair = self._recs_pair          # one read → frame and boxes agree
                raw, recs, fid = pair if pair else (None, [], -1)
                # Detect runs slower than FALL_LOOP_FPS, so this loop would otherwise
                # append the SAME frame to the 30-frame window several times: the
                # sequence freezes while `now` keeps advancing, and a quick fall looks
                # like a slow one to a model trained on real motion. Only step on a
                # frame we have not already consumed.
                fresh, last_fid = (fid != last_fid), fid
                if raw is not None and recs and fresh:
                    try:
                        for ev in eng.fall_step(raw, recs):
                            self._emit_event(ev)
                    except Exception as e:
                        print(f"[Pipeline] fall error: {e}")
                spent = time.monotonic() - t0
                if spent > interval:
                    overruns += 1
                    if overruns % 30 == 1:      # never drop a person; just report
                        print(f"[Pipeline] ⚠️ fall tick {spent*1000:.0f}ms > "
                              f"{interval*1000:.0f}ms budget ({len(recs)} people)")
            dt = interval - (time.monotonic() - t0)
            time.sleep(dt if dt > 0 else 0.001)

    def _sync_module_status(self):
        """Reflect engine roles into status['modules'] (ok / standby / off)."""
        if self._engine is None:
            return
        with self._lock:
            self.status["modules"]["ppe"] = "ok" if self._engine.ppe_enabled else "off"
            if not self._engine.zone_enabled:
                self.status["modules"]["zone"] = "off"
            else:
                self.status["modules"]["zone"] = "ok" if self._engine.zones else "standby"
            if not self._engine.fall_enabled:
                self.status["modules"]["fall"] = "off"
            else:                                # enabled but models failed → visible
                self.status["modules"]["fall"] = "ok" if self._engine.fall_ready else "err"
            snapshot = dict(self.status)
        if self.on_status:
            try:
                self.on_status(snapshot)
            except Exception:
                pass

    def _emit_event(self, ev: dict):
        """Route an engine event → alert counters + UI/History callback (on_alert)."""
        level = ev.get("level", "warning")
        # Log every alert the ENGINE raises, before any of the switches below can
        # swallow it. Without this line the console is silent either way, so a quiet
        # dashboard could mean "the AI saw nothing" OR "the AI saw it and the UI
        # dropped it" — two completely different bugs that look identical.
        print(f"[Pipeline] 🚨 {level.upper()} ({ev.get('type')}): {ev.get('msg', '')}")
        # Per-level alert switch (Settings): a disabled level is fully suppressed.
        if not self._alert_levels.get(level, True):
            print(f"[Pipeline] ⏹️  ...ถูกปิดไว้ใน Settings (alerts.{level}_enabled=false) → ไม่ส่ง")
            return
        with self._lock:
            a = self.status["alerts"]
            a["total"] = a.get("total", 0) + 1
            # One bucket per level. A zone intrusion is level "alert" and used to
            # fall into the `else` branch here, so every intrusion incremented the
            # EMERGENCY counter and lit the dashboard's red alarm state.
            if level in ("warning", "alert", "emergency"):
                a[level] = a.get(level, 0) + 1
        if self.on_alert:
            try:
                self.on_alert(ev.get("msg", ""), level, ev.get("type", "ppe"))
            except Exception as e:
                print(f"[Pipeline] on_alert error: {e}")

    def apply_settings(self, settings: dict):
        try:
            cam = settings.get("camera", {})
            if "flip_horizontal" in cam:
                self._flip_override = bool(cam["flip_horizontal"])
            line = settings.get("line", {})
            try:
                import config as cfg
                if "channel_access_token" in line:
                    cfg.LINE_OA_CHANNEL_ACCESS_TOKEN = line["channel_access_token"]
                sup = line.get("group_supervisor", getattr(cfg, "LINE_OA_GROUP_SUPERVISOR", ""))
                saf = line.get("group_safety",     getattr(cfg, "LINE_OA_GROUP_SAFETY", ""))
                emg = line.get("group_emergency",  getattr(cfg, "LINE_OA_GROUP_EMERGENCY", ""))
                if any(k in line for k in ("group_supervisor", "group_safety", "group_emergency")):
                    cfg.LINE_OA_GROUP_SUPERVISOR = sup
                    cfg.LINE_OA_GROUP_SAFETY     = saf
                    cfg.LINE_OA_GROUP_EMERGENCY  = emg
                    # CRITICAL: config.ALERT_RECIPIENTS is built ONCE at import time
                    # from the (then-empty) group ids, and send_line_notify() picks
                    # recipients from it per level. Updating the group vars above does
                    # NOT touch that frozen dict, so live per-event alerts (fall/zone/
                    # PPE) had an EMPTY recipient list → nothing was ever pushed even
                    # though the manual daily-report button worked (it reads the group
                    # vars directly). Rebuild the map here with the live ids so real
                    # detections actually reach LINE.
                    cfg.ALERT_RECIPIENTS = {
                        cfg.ALERT_LEVEL_WARNING:   [sup],
                        cfg.ALERT_LEVEL_ALERT:     [saf, sup],
                        cfg.ALERT_LEVEL_EMERGENCY: [emg, saf, sup],
                    }

                # ── AI thresholds (INFERENCE_CONFIDENCE is read per-frame in
                # detect_track → hot; confirm/cooldown need refresh_tunables) ──
                ai = settings.get("ai", {})
                if "ppe_confidence" in ai:
                    cfg.INFERENCE_CONFIDENCE = float(ai["ppe_confidence"])
                if "fall_bbox_ratio" in ai:
                    # This slider wrote FALL_BBOX_RATIO_THRESH, which fall_detector
                    # never read — moving it did nothing at all. It now drives the
                    # value it was always meant to: the absolute width/height floor
                    # above which a box counts as prone (read per-frame in
                    # _posture, so it takes effect without a model reload).
                    cfg.FALL_AR_ABS_MIN = float(ai["fall_bbox_ratio"])
                    cfg.FALL_BBOX_RATIO_THRESH = float(ai["fall_bbox_ratio"])  # legacy mirror
                if "fall_confirm_frames" in ai:
                    # CLAMP to the window. The confirmer is N-of-M over a deque of
                    # maxlen=FALL_CONFIRM_WINDOW, so N > M can never be satisfied —
                    # sum() of at most M booleans is at most M. A saved value of 6
                    # against a window of 5 made fall alarms MATHEMATICALLY
                    # IMPOSSIBLE in the live app, silently, while the offline
                    # harness (which reads cfg directly and never sees settings.json)
                    # kept reporting that falls were detected. A safety module that
                    # cannot fire must never be reachable from a settings slider.
                    win = int(getattr(cfg, "FALL_CONFIRM_WINDOW", 5))
                    want = int(ai["fall_confirm_frames"])
                    if want > win:
                        print(f"[Pipeline] ⚠️ fall_confirm_frames={want} > window={win} "
                              f"→ ตรึงไว้ที่ {win} (ค่าเดิมทำให้แจ้งเตือนการล้มไม่ได้เลย)")
                    cfg.FALL_CONFIRM_FRAMES = max(1, min(want, win))
                if "fall_mode" in ai:
                    cfg.FALL_MODE = str(ai["fall_mode"])

                alerts = settings.get("alerts", {})
                if "violation_cooldown_seconds" in alerts:
                    cfg.VIOLATION_COOLDOWN_SECONDS = int(alerts["violation_cooldown_seconds"])
                if "zone_cooldown_seconds" in alerts:
                    cfg.ZONE_COOLDOWN_SECONDS = int(alerts["zone_cooldown_seconds"])
                if "fall_cooldown_seconds" in alerts:
                    cfg.FALL_COOLDOWN_SECONDS = int(alerts["fall_cooldown_seconds"])

                # Per-level alert switches (disabled level → fully suppressed)
                for lvl, key in (("warning", "warning_enabled"),
                                 ("alert", "alert_enabled"),
                                 ("emergency", "emergency_enabled")):
                    if key in alerts:
                        self._alert_levels[lvl] = bool(alerts[key])
            except ImportError:
                pass

            # ── Per-camera detection roles + PPE items for the running camera ──
            cams   = settings.get("cameras", {}) or {}
            cam_id = self._source_config.get("camera_id", "cam0")
            camcfg = cams.get(cam_id) if isinstance(cams.get(cam_id), dict) else None
            if camcfg is not None:
                if "roles" in camcfg:
                    self._roles = set(camcfg["roles"] or [])
                    self._source_config["roles"] = list(self._roles)
                if "ppe_items" in camcfg:
                    self._ppe_items = list(camcfg["ppe_items"] or [])
                    self._source_config["ppe_items"] = self._ppe_items

            # ── Hot-apply to a live engine (no model reload) ──
            if self._engine is not None:
                try:
                    self._engine.apply_roles(self._roles)
                    self._engine.apply_ppe_items(self._ppe_items)
                    self._engine.refresh_tunables()
                    self._sync_module_status()
                except Exception as e:
                    print(f"[Pipeline] engine hot-apply: {e}")

            print("[Pipeline] ⚙️  Settings applied")
        except Exception as e:
            print(f"[Pipeline] apply_settings: {e}")

    # ── Private helpers ───────────────────────────────────────

    def _apply_config(self, src_cfg: dict):
        try:
            import config as cfg
            cfg.CAMERA_SOURCE   = src_cfg.get("source", "webcam")
            cfg.WEBCAM_INDEX    = int(src_cfg.get("webcam_index", 0))
            cfg.RTSP_URL        = src_cfg.get("rtsp_url", getattr(cfg, "RTSP_URL", ""))
            cfg.VIDEO_FILE_PATH = src_cfg.get("video_file_path", "")
            cfg.ZONE_POLYGON_FILE = str(_APP_DATA / "zones.json")
        except ImportError:
            pass

    def _open_camera(self, src_cfg: dict) -> cv2.VideoCapture:
        src = src_cfg.get("source", "webcam")
        if src == "webcam":
            # CAP_DSHOW is Windows-only; use the platform default elsewhere
            # (AVFoundation on macOS, V4L2 in a Linux container).
            idx = int(src_cfg.get("webcam_index", 0))
            cap = (cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32"
                   else cv2.VideoCapture(idx))
        elif src == "rtsp":
            cap = cv2.VideoCapture(src_cfg.get("rtsp_url", ""), cv2.CAP_FFMPEG)
        elif src == "file":
            cap = cv2.VideoCapture(src_cfg.get("video_file_path", ""))
        else:
            raise ValueError(f"Unknown source: {src}")
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open camera (source={src})")
        # BUFFERSIZE=1: for a live feed we always want the NEWEST frame, not a
        # queue of stale ones. A larger driver buffer adds standing latency (each
        # buffered frame is ~1/fps behind) with no benefit for real-time safety.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    def _set_camera_state(self, state: str):
        changed = False
        with self._lock:
            if self.status.get("camera") != state:
                self.status["camera"] = state
                changed = True
            snapshot = dict(self.status)
        if changed and self.on_status:
            try:
                self.on_status(snapshot)
            except Exception as e:
                print(f"[Pipeline] on_status callback: {e}")

    def _reconnect_camera(self) -> bool:
        self._set_camera_state("reconnecting")
        if self._reader:
            try:
                self._reader.stop()
            except Exception:
                pass
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        delays  = [1.0, 2.0, 3.0, 5.0]
        attempt = 0
        while not self._stop_evt.is_set() and self._running:
            try:
                self._cap = self._open_camera(self._source_config)
                reader    = _FrameReader(self._cap)
                reader.start()
                self._reader = reader
                self._set_camera_state("connected")
                print("[Pipeline] 🔌 Camera reconnected")
                return True
            except Exception as e:
                wait = delays[min(attempt, len(delays) - 1)]
                attempt += 1
                print(f"[Pipeline] reconnect attempt {attempt} failed ({e}); retry in {wait}s")
                slept = 0.0
                while slept < wait and not self._stop_evt.is_set() and self._running:
                    time.sleep(0.2)
                    slept += 0.2
        return False

    def _process_loop(self):
        try:
            try:
                import config as cfg
                is_file   = (cfg.CAMERA_SOURCE == "file")
                is_webcam = (cfg.CAMERA_SOURCE == "webcam")
            except ImportError:
                is_file   = False
                is_webcam = True

            # File → throttle to the clip's native fps (time-ordered stream);
            # live camera self-paces (src_fps=0 = no throttle).
            src_fps = 0.0
            if is_file:
                try:
                    src_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
                    if src_fps <= 0 or src_fps > 120:
                        src_fps = 25.0
                except Exception:
                    src_fps = 25.0
            reader = _FrameReader(self._cap, loop=is_file, src_fps=src_fps)
            reader.start()
            self._reader = reader

            # Build the detection engine (falls back to passthrough on failure)
            self._engine = self._build_engine()
            if self._engine is not None:
                self._sync_module_status()
                self._latest_recs = []
                self._detect_thr = threading.Thread(
                    target=self._detect_loop, daemon=True, name="DetectLoop")
                self._detect_thr.start()
                # Always started: it idles while the fall role is off, so toggling
                # the role in Settings takes effect without restarting the camera.
                self._fall_thr = threading.Thread(
                    target=self._fall_loop, daemon=True, name="FallLoop")
                self._fall_thr.start()

            frame_id      = 0
            read_failures = 0
            mode = "detection (decoupled)" if self._engine else "passthrough"
            print(f"[Pipeline] ▶️  Process loop running ({mode})")

            while not self._stop_evt.is_set() and self._running:
                ret, raw = (self._reader.read() if self._reader else (False, None))
                if not ret or raw is None:
                    read_failures += 1
                    if read_failures > 40 and not is_file:
                        print("[Pipeline] ⚠️  Camera signal lost — reconnecting")
                        if not self._reconnect_camera():
                            break
                        read_failures = 0
                    continue
                read_failures = 0

                frame_id += 1
                # Horizontal mirror only makes sense for a live webcam (selfie view).
                # NEVER flip files/RTSP: it's meaningless there and can hurt detection.
                flip = False
                if is_webcam:
                    flip = True if self._flip_override is None else self._flip_override
                if flip:
                    raw = cv2.flip(raw, 1)

                # publish newest raw for the detection worker (skips old frames).
                # MUST stay pristine: draw_on() annotates in place, so the display
                # draws on a COPY — otherwise the detector would run on a frame with
                # boxes already painted on it and fail to detect anyone.
                self._raw_pair = (raw, frame_id)

                # DISPLAY = fast: draw the latest known boxes onto a copy of the frame
                out = raw
                if self._engine is not None:
                    try:
                        out = self._engine.draw_on(raw.copy(), self._latest_recs)
                    except Exception as e:
                        print(f"[Pipeline] draw_on error: {e}")
                        out = raw

                with self._frame_lock:
                    self._latest_frame = out

                if frame_id % 150 == 0:
                    with self._lock:
                        self.status["uptime_seconds"] = self.get_uptime()

        except Exception:
            print("[Pipeline] ❌ Process loop crashed:")
            traceback.print_exc()
        finally:
            self._running = False
            with self._lock:
                self.status["running"] = False
            self._set_camera_state("disconnected")
            print("[Pipeline] Process loop ended")
