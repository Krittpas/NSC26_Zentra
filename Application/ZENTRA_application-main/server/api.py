"""
server/api.py — ZENTRA FastAPI Server (Stage B — Real Pipeline)
REST + SSE + WebSocket endpoints backed by the real AI pipeline.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import mimetypes
import os
import threading
import time

# Bundled web fonts: Python's mimetypes doesn't know woff2 → StaticFiles would
# serve it with a generic type. Register the correct type so the browser accepts
# the locally-hosted Kanit font without complaint.
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
UI_DIR   = BASE_DIR / "ui"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Backend (AI) project + its auto-collected data. Single-repo: the backend lives
# at <repo>/backend. This used to be BASE_DIR.parent/"ZENTRA" — a leftover from
# when ZENTRA_application/ sat beside ZENTRA/. It resolved to the repo root by
# coincidence (repo is named ZENTRA), so /api/data/stats and /api/training/metrics
# silently pointed at directories that never existed and always returned empty.
ZENTRA_BACKEND = BASE_DIR / "backend"
COLLECTED_DIR  = ZENTRA_BACKEND / "data" / "collected"
_DATA_CATEGORIES = ["ppe_violations", "zone_intrusions", "fall_events", "normal"]

# ── Optional API auth ───────────────────────────────────────
# The API is unauthenticated by default and is bound to localhost for exactly that
# reason (see README + docker-compose). Set ZENTRA_API_TOKEN to require a shared
# secret on every /api and /ws call, so the app can sit behind a LAN / reverse
# proxy without exposing worker evidence photos, the event log, or the training
# subprocess spawner to anyone who can reach the port. Empty (default) = no auth,
# behaviour unchanged.
_API_TOKEN = os.getenv("ZENTRA_API_TOKEN", "").strip()


def _token_ok(provided: str | None) -> bool:
    """Constant-time token check. Always True when no token is configured."""
    if not _API_TOKEN:
        return True
    return bool(provided) and hmac.compare_digest(provided, _API_TOKEN)


# ── App lifespan (replaces the deprecated @app.on_event startup/shutdown) ────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    try:
        yield
    finally:
        await _shutdown()


# ── FastAPI app ─────────────────────────────────────────────
app = FastAPI(title="ZENTRA API", docs_url=None, redoc_url=None, lifespan=lifespan)

# NO CORS MIDDLEWARE — deliberately.
#
# The UI is served by this same app, so every request it makes is same-origin and
# needs no CORS headers at all. The previous `allow_origins=["*"]` did nothing for
# the app and everything for an attacker: because the API has no authentication,
# `Access-Control-Allow-Origin: *` let ANY web page the operator happened to visit
# read http://localhost:7788/api/settings (which returns the LINE channel access
# token) and POST /api/zones (whose zone name lands in the History page).
#
# If you ever need a browser on another origin to call this API, add authentication
# FIRST, then allow that exact origin — never "*".


@app.middleware("http")
async def _auth_gate(request, call_next):
    """Gate /api/* behind ZENTRA_API_TOKEN when it is set. The static UI shell
    (/, /ui/*) stays open — it holds no secrets — so the app still loads and can
    then attach the token to its calls. No token configured → every request passes
    and behaviour is identical to before."""
    if _API_TOKEN and request.url.path.startswith("/api"):
        auth = request.headers.get("Authorization", "")
        tok = auth[7:] if auth.startswith("Bearer ") else request.query_params.get("token")
        if not _token_ok(tok):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


class _NoCacheStatic(StaticFiles):
    """Serve UI assets with Cache-Control: no-cache so browsers always revalidate
    (ETag → cheap 304 when unchanged). Prevents the SPA from showing stale
    screens/JS after an edit without a hard refresh."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


app.mount("/ui", _NoCacheStatic(directory=str(UI_DIR)), name="ui")

# ── Globals (set on startup) ─────────────────────────────────
_loop:       asyncio.AbstractEventLoop | None = None
_broadcaster = None
pipeline     = None   # Pipeline singleton
_retention_task = None   # periodic PDPA purge task

# LINE pushes are blocking HTTP with retries. They must never run on the
# detect/fall loop (that thread has a frame budget) nor on the event loop.
# ONE worker → pushes are serialised, so a burst can't open N sockets at once.
_line_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="line-push")


def _camera_label() -> str:
    """Which camera produced the event that is being recorded RIGHT NOW.

    This used to be the string literal "Cam 1", so every row in History claimed
    to come from Cam 1 no matter which camera was actually running — the evidence
    could not be traced back to a place, which is most of what History is for.

    The running pipeline knows its own camera_id (set on /api/pipeline/start), so
    take it from there. A human name from settings (`cameras.<id>.name`) wins when
    the operator has set one; otherwise the id itself is used, which is at least
    true. Falls back to "unknown" rather than inventing a camera.
    """
    cam_id = ""
    if pipeline is not None:
        cam_id = (pipeline._source_config or {}).get("camera_id") or ""
    if not cam_id:
        return "unknown"
    try:
        cam = (_load_settings().get("cameras") or {}).get(cam_id)
        if isinstance(cam, dict) and str(cam.get("name", "")).strip():
            return str(cam["name"]).strip()
    except Exception:
        pass
    return cam_id


def _push_line(event_id: int, msg: str, level: str, ev_type: str, frame) -> None:
    """Send one alert to LINE and record the real outcome against the event.

    Returns quietly when LINE is unconfigured (no token / no group): that is the
    normal offline state, not an error, and the event stays in History either way.
    """
    from server import store
    try:
        from alerts.line_notify import send_line_notify
    except Exception as e:
        print(f"[API] LINE unavailable: {e}")
        return
    try:
        # The engine already gates every alert through its own per-track confirm
        # window + CooldownGate, so line_notify must NOT gate a second time — its
        # default key is global and would swallow unrelated alerts.
        ok = send_line_notify(msg, image=frame, level=level,
                              cooldown_key=f"{ev_type}:{msg}", cooldown_sec=0,
                              async_send=False)
        store.mark_line_sent(event_id, bool(ok))
    except Exception as e:
        print(f"[API] LINE push failed for event {event_id}: {e}")


# ================================================================
# WEBSOCKET MANAGER
# ================================================================
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# Events are persisted locally via server/store.py (SQLite + snapshot files);
# nothing is kept only in memory and nothing leaves the device (PDPA).


# ================================================================
# STARTUP / SHUTDOWN
# ================================================================
# PDPA data minimisation runs at startup AND on this timer. The startup purge
# alone is not enough: a factory box runs for weeks, and without a periodic sweep
# it would never drop events/snapshots that age past retention_days until the next
# restart — so old worker evidence would pile up indefinitely. 6h keeps disk and
# retention honest at negligible cost (one indexed DELETE).
_RETENTION_INTERVAL_SEC = 6 * 3600


def _run_retention_purge() -> None:
    """Read retention_days from settings and drop older events. Safe to call
    repeatedly (idempotent — deletes anything past the day cutoff)."""
    rdays = int((_load_settings().get("data") or {}).get("retention_days", 0) or 0)
    if rdays <= 0:
        return
    from server import store
    removed = store.purge_before(rdays)
    if removed:
        print(f"[API] PDPA retention: purged {removed} event(s) older than {rdays} day(s)")


async def _retention_loop():
    """Re-run the retention purge every _RETENTION_INTERVAL_SEC so a long-running
    instance keeps honouring retention_days without needing a restart."""
    while True:
        try:
            await asyncio.sleep(_RETENTION_INTERVAL_SEC)
            await _in_executor(_run_retention_purge)   # off the event loop (SQLite)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[API] retention loop error: {e}")


async def _startup():
    global _loop, _broadcaster, pipeline, _retention_task

    _loop = asyncio.get_running_loop()

    if _API_TOKEN:
        print("[API] 🔒 Token auth ENABLED (ZENTRA_API_TOKEN set).")
    else:
        print("[API] ⚠️  API is UNAUTHENTICATED — keep it bound to 127.0.0.1 only "
              "(set ZENTRA_API_TOKEN to require a token behind a proxy).")

    # PDPA data minimisation: drop events + evidence snapshots older than the
    # configured retention window (local only). Once now, then every 6h via
    # _retention_loop so a long-running instance keeps purging.
    try:
        _run_retention_purge()
    except Exception as e:
        print(f"[API] retention purge skipped: {e}")
    _retention_task = asyncio.create_task(_retention_loop())

    try:
        # Import Pipeline (adds ZENTRA backend to sys.path, imports cv2/numpy)
        from pipeline.pipeline          import Pipeline
        from pipeline.frame_broadcaster import FrameBroadcaster

        pipeline = Pipeline()

        # Wire alert callback → WebSocket broadcast + history + LINE push
        def _on_alert(msg: str, level: str, ev_type: str = "ppe"):
            # Capture an evidence snapshot (local only) of the current frame
            snap = frame = None
            try:
                if pipeline:
                    frame = pipeline.get_latest_frame()
                    snap = pipeline.get_snapshot()
            except Exception:
                snap = frame = None
            from server import store
            camera = _camera_label()
            # Persist FIRST: evidence must survive a LINE/network failure. The
            # line_sent flag is corrected by _push_line once the push resolves.
            event = store.add_event(level=level, message=msg, camera=camera,
                                    frame_jpeg=snap, line_sent=False, type_=ev_type)
            _line_pool.submit(_push_line, event["id"], msg, level, ev_type, frame)
            # Authoritative counts from the pipeline (avoids client drift)
            with pipeline._lock:
                alerts = dict(pipeline.status.get("alerts", {}))
            broadcast_msg = {
                "type":         "event",
                "event":        "alert",
                "id":           event["id"],
                "level":        level,
                "kind":         ev_type,
                "message":      event["message"],
                "timestamp":    event["time"],
                "camera":       camera,
                "alerts":       alerts,
                "has_snapshot": event["has_snapshot"],
            }
            asyncio.run_coroutine_threadsafe(
                manager.broadcast(broadcast_msg), _loop
            )

        pipeline.on_alert = _on_alert

        # Wire status changes (camera connect/reconnect/disconnect) → WebSocket
        def _on_status(status: dict):
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({
                    "type":    "event",
                    "event":   "status",
                    "modules": status.get("modules", {}),
                    "alerts":  status.get("alerts", {}),
                    "camera":  status.get("camera", "disconnected"),
                    "running": status.get("running", False),
                    # Non-null → the AI engine failed to load and NOTHING is being
                    # detected. The UI must say so loudly; clean video with no boxes
                    # otherwise reads as "no violations".
                    "engine_error": status.get("engine_error"),
                }),
                _loop,
            )

        pipeline.on_status = _on_status

        # Apply any saved settings to config before first run
        try:
            pipeline.apply_settings(_load_settings())
        except Exception as e:
            print(f"[API] settings preload skipped: {e}")

        # Start frame broadcaster (frames → WebSocket). STREAM_FPS drives motion
        # smoothness of the raw video; encode is ~0.6ms so 15 is cheap on LAN.
        _stream_fps = int(os.getenv("STREAM_FPS", "15"))
        _broadcaster = FrameBroadcaster(pipeline, manager, _loop, fps=_stream_fps)
        _broadcaster.start()
        print("[API] Startup complete ✅")

    except Exception as e:
        print(f"[API] ⚠️  Startup warning (pipeline not loaded): {e}")
        print("[API] Server running in UI-only mode")


async def _shutdown():
    global _broadcaster, _retention_task
    if _retention_task:
        _retention_task.cancel()
        _retention_task = None
    if _broadcaster:
        _broadcaster.stop()
    if pipeline:
        pipeline.stop()
    # Let any in-flight LINE push finish so an alert isn't lost on shutdown.
    _line_pool.shutdown(wait=True)
    print("[API] Shutdown complete")


# ================================================================
# STATIC / ROOT
# ================================================================
@app.get("/")
async def root():
    return FileResponse(str(UI_DIR / "index.html"))


# ================================================================
# SPLASH — SSE init progress
# ================================================================
@app.get("/api/init")
async def init_stream():
    steps = [
        (15,  "กำลังโหลดการตั้งค่า..."),
        (35,  "เริ่มต้นเครื่องตรวจจับ AI (ประมวลผลในเครื่อง)..."),
        (55,  "เตรียมโมดูล PPE / โซน / การล้ม..."),
        (75,  "โหลดข้อมูลโซนความปลอดภัย..."),
        (90,  "เริ่มต้นระบบ..."),
        (100, "พร้อมใช้งาน"),
    ]

    async def _gen():
        for pct, msg in steps:
            yield f"data: {json.dumps({'percent': pct, 'message': msg})}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ================================================================
# STATUS
# ================================================================
@app.get("/api/status")
async def status():
    if pipeline is None:
        return JSONResponse({
            "running": False, "source": None,
            "modules": {"ppe": "error", "zone": "error", "fall": "error"},
            "alerts":  {"total": 0, "warning": 0, "alert": 0, "emergency": 0},
            "uptime":  0, "last_emergency": None,
            "engine_error": "pipeline not initialised",
        })
    with pipeline._lock:
        s = dict(pipeline.status)
    s["uptime"] = pipeline.get_uptime()
    eng = getattr(pipeline, "_engine", None)
    s["device"] = (getattr(eng.detector, "device", None) if eng is not None
                   else os.getenv("PPE_INFER_DEVICE", "cpu"))
    return JSONResponse(s)


# ================================================================
# PIPELINE  start / stop
# ================================================================
@app.post("/api/pipeline/start")
async def pipeline_start(body: dict[str, Any]):
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "pipeline not initialised"}, status_code=503)

    source    = body.get("source", "webcam")
    camera_id = body.get("camera_id", "cam0")
    # Per-camera detection roles + PPE items to enforce from settings
    # (None = all modules on / cfg default PPE items).
    roles = None
    ppe_items = None
    try:
        cams = (_load_settings().get("cameras") or {})
        c = cams.get(camera_id)
        if isinstance(c, dict):
            if isinstance(c.get("roles"), list):
                roles = c["roles"]
            if isinstance(c.get("ppe_items"), list):
                ppe_items = c["ppe_items"]
    except Exception:
        roles = None
    src_cfg = {
        "source":          source,
        "webcam_index":    int(body.get("webcam_index", 0)),
        "rtsp_url":        body.get("rtsp_url", ""),
        "video_file_path": body.get("video_file_path", ""),
        "camera_id":       camera_id,
        "roles":           roles,
        "ppe_items":       ppe_items,
    }

    # Run blocking start() in thread pool so we don't block the event loop
    loop   = asyncio.get_running_loop()
    ok     = await loop.run_in_executor(None, pipeline.start, src_cfg)
    if not ok:
        return JSONResponse({"ok": False, "error": "ไม่สามารถเปิดกล้อง / แหล่งภาพได้"}, status_code=400)

    return JSONResponse({"ok": True, "source": source})


@app.post("/api/pipeline/stop")
async def pipeline_stop():
    if pipeline:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline.stop)
    return JSONResponse({"ok": True})


# ================================================================
# ZONES
# ================================================================
ZONES_FILE = DATA_DIR / "zones.json"


def _load_zones() -> list:
    if ZONES_FILE.exists():
        return json.loads(ZONES_FILE.read_text(encoding="utf-8"))
    return []


def _save_zones(zones: list) -> None:
    ZONES_FILE.write_text(json.dumps(zones, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/zones")
async def get_zones():
    return JSONResponse(_load_zones())


@app.post("/api/zones")
async def create_zone(body: dict[str, Any]):
    zones = _load_zones()
    zone  = {
        "id":        max((z.get("id", 0) for z in zones), default=0) + 1,
        "name":      body.get("name", f"Zone {len(zones) + 1}"),
        "color":     body.get("color", "#ef4444"),
        "points":    body.get("points", []),
        "type":      body.get("type", "danger"),   # danger (detect) | exclusion (ignore)
        "camera_id": body.get("camera_id", "cam0"),  # which camera this zone belongs to
        "enabled":   True,
    }
    zones.append(zone)
    _save_zones(zones)
    if pipeline:
        pipeline.reload_zones()
    return JSONResponse(zone)


@app.put("/api/zones/{zone_id}")
async def update_zone(zone_id: int, body: dict[str, Any]):
    zones = _load_zones()
    for z in zones:
        if z.get("id") == zone_id:
            z.update({k: v for k, v in body.items() if k != "id"})
            _save_zones(zones)
            if pipeline:
                pipeline.reload_zones()
            return JSONResponse(z)
    return JSONResponse({"error": "not found"}, status_code=404)


@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: int):
    zones = [z for z in _load_zones() if z.get("id") != zone_id]
    _save_zones(zones)
    if pipeline:
        pipeline.reload_zones()
    return JSONResponse({"ok": True})


# ================================================================
# SETTINGS
# ================================================================
SETTINGS_FILE = DATA_DIR / "settings.json"

SETTINGS_DEFAULTS: dict[str, Any] = {
    "line": {
        "channel_access_token": "",
        "group_supervisor": "",
        "group_safety": "",
        "group_emergency": "",
    },
    "ai": {
        # Person-centric engine needs decent person recall; 0.30 matches the
        # deployed default. Higher values (0.70) filtered detections out entirely.
        "ppe_confidence": 0.30,
        "fall_bbox_ratio": 0.72,
        "fall_confirm_frames": 6,
        "fall_mode": "hybrid",          # hybrid | yolo | pose
    },
    "alerts": {
        "violation_cooldown_seconds": 30,
        "zone_cooldown_seconds": 20,
        "fall_cooldown_seconds": 15,
        "warning_enabled": True,
        "alert_enabled":   True,
        "emergency_enabled": True,
        # PDPA: OFF by default — LINE alerts are text-only, no image leaves the
        # device. Opt-in sends evidence photos via an external public host.
        "upload_images":   False,
    },
    "camera": {
        "source": "webcam",
        "webcam_index": 0,
        "rtsp_url": "",
        "video_file_path": "",
        "flip_horizontal": True,
    },
    "display": {
        "stream_fps": 10,
        "stream_jpeg_quality": 70,
    },
    "data": {
        # PDPA data minimisation: auto-delete events + snapshots older than N days
        # on startup. 0 = keep forever.
        "retention_days": 90,
    },
    # Per-camera detection roles, e.g. {"cam0": {"roles": ["ppe","zone"]}}.
    # Absent/empty → all modules on for that camera.
    "cameras": {},
    # Organization identity printed on exported safety reports.
    "report": {
        "site": "",
        "company": "",
        "preparer": "",
    },
}


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        saved  = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        merged = {**SETTINGS_DEFAULTS}
        for k, v in saved.items():
            if isinstance(v, dict) and k in merged:
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        return merged
    return dict(SETTINGS_DEFAULTS)


def _save_settings(data: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into `base` (nested dicts merged, not replaced),
    so a POST that only touches one section can't clobber another screen's keys."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_TOKEN_KEY = "channel_access_token"


def _redacted(settings: dict) -> dict:
    """Settings safe to hand to a client: the LINE channel access token never
    leaves the server. It is a bearer credential — anyone holding it can push
    messages to the customer's LINE groups — and the UI has no reason to read it
    back. `token_set` tells the UI whether to show "saved" or "not configured".
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in settings.items()}
    line = out.get("line") or {}
    line["token_set"] = bool(line.get(_TOKEN_KEY))
    line[_TOKEN_KEY] = ""
    out["line"] = line
    return out


@app.get("/api/settings")
async def get_settings():
    return JSONResponse(_redacted(_load_settings()))


@app.get("/api/ai/ppe-models")
async def get_ppe_models():
    """List selectable PPE models for the Settings dropdown (from config)."""
    try:
        import config as cfg
        return JSONResponse({"models": list(getattr(cfg, "PPE_MODELS", [])),
                             "local_sentinel": getattr(cfg, "PPE_LOCAL_SENTINEL", "__local__")})
    except Exception as e:
        return JSONResponse({"models": [], "local_sentinel": "__local__", "error": str(e)})


@app.post("/api/settings")
async def save_settings(body: dict[str, Any]):
    # Deep-merge over what's on disk so concurrent screens don't clobber keys.
    on_disk = {}
    if SETTINGS_FILE.exists():
        try:
            on_disk = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            on_disk = {}
    body = {k: (dict(v) if isinstance(v, dict) else v) for k, v in body.items()}
    line = body.get("line")
    if isinstance(line, dict):
        line.pop("token_set", None)          # read-only marker from _redacted()
        # GET never returns the token, so the Settings form posts it back empty
        # unless the user typed a new one. Empty means "leave unchanged" — writing
        # it through would erase the token every time any setting was saved.
        if not line.get(_TOKEN_KEY):
            line.pop(_TOKEN_KEY, None)
    merged = _deep_merge(on_disk, body)
    _save_settings(merged)
    if pipeline:
        pipeline.apply_settings(merged)
    return JSONResponse({"ok": True})


# ================================================================
# SNAPSHOT  (for Zone Editor canvas background)
# ================================================================

# 1×1 dark pixel PNG fallback (no camera)
_DARK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@app.get("/api/frame/snapshot")
async def frame_snapshot():
    if pipeline and pipeline.is_running():
        snap = pipeline.get_snapshot()
        if snap:
            return StreamingResponse(iter([snap]), media_type="image/jpeg")
    return StreamingResponse(iter([_DARK_PNG]), media_type="image/png")


# ================================================================
# HISTORY  (backed by local SQLite store — PDPA: on-device)
# ================================================================
async def _in_executor(fn, *args, **kwargs):
    """Run a blocking (SQLite) call off the event loop so it can't stall the
    WebSocket frame broadcast or any other request while it runs."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


@app.get("/api/history/today")
async def history_today(day: str | None = None):
    from server import store
    s = await _in_executor(store.today_stats, day)
    s["uptime_seconds"] = pipeline.get_uptime() if pipeline else 0
    return JSONResponse(s)


@app.get("/api/history/hourly")
async def history_hourly(day: str | None = None):
    from server import store
    return JSONResponse(await _in_executor(store.hourly, day))


@app.get("/api/history/events")
async def history_events(limit: int = 20, offset: int = 0, day: str | None = None,
                         start: str | None = None, end: str | None = None):
    from server import store
    return JSONResponse(await _in_executor(
        store.list_events, limit=limit, offset=offset, day=day, start=start, end=end))


@app.get("/api/history/days")
async def history_days():
    from server import store
    return JSONResponse({"days": await _in_executor(store.available_days)})


@app.get("/api/history/snapshot/{event_id}")
async def history_snapshot(event_id: int):
    from server import store
    p = await _in_executor(store.snapshot_path, event_id)
    if p:
        return FileResponse(str(p), media_type="image/jpeg")
    return StreamingResponse(iter([_DARK_PNG]), media_type="image/png")


@app.get("/api/history/export.csv")
async def history_export(day: str | None = None, start: str | None = None,
                         end: str | None = None):
    from server import store
    tag = (start + "_" + end) if (start and end) else (day or "all")
    csv = await _in_executor(store.export_csv, day, start=start, end=end)
    return StreamingResponse(
        iter([csv]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=zentra_history_{tag}.csv"},
    )


@app.post("/api/history/clear")
async def history_clear():
    from server import store
    removed = await _in_executor(store.purge_all)
    # The dashboard KPIs read the pipeline's LIVE session counters, not the DB, so
    # clearing History alone leaves stale counts on the dashboard. Zero those too
    # and broadcast the reset so open dashboards update without a reload.
    if pipeline is not None:
        with pipeline._lock:
            pipeline.status["alerts"] = pipeline._zero_alerts()
            pipeline.status["last_emergency"] = None
            snapshot = dict(pipeline.status)
        if pipeline.on_status:
            try:
                pipeline.on_status(snapshot)
            except Exception:
                pass
    return JSONResponse({"ok": True, "removed": removed})


# ================================================================
# DAILY REPORT  (local PDF + LINE text summary)
# ================================================================
@app.get("/api/report/daily.pdf")
async def report_daily_pdf(day: str | None = None, start: str | None = None,
                           end: str | None = None):
    from server.report import build_daily_pdf
    report_cfg = (_load_settings().get("report") or {})
    loop = asyncio.get_running_loop()
    try:
        path = await loop.run_in_executor(
            None, lambda: build_daily_pdf(day=day, start=start, end=end, org=report_cfg))
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return FileResponse(str(path), media_type="application/pdf", filename=path.name)


@app.post("/api/report/send-line")
async def report_send_line(body: dict[str, Any] | None = None):
    """Push the daily summary to LINE and report the REAL outcome.

    This used to return ok:True unconditionally, so with LINE unconfigured (no
    token / no group id) the UI cheerfully said "sent" while nothing left the
    machine. A safety tool must never claim an alert was delivered when it wasn't.
    """
    body = body or {}
    day  = body.get("day")
    try:
        from server.report import daily_stats_for_line
        from alerts.line_notify import send_daily_report
        import config as cfg

        if not getattr(cfg, "LINE_OA_CHANNEL_ACCESS_TOKEN", ""):
            return JSONResponse(
                {"ok": False, "error": "ยังไม่ได้ตั้งค่า LINE Token — ไปที่ ตั้งค่า → การแจ้งเตือน LINE"},
                status_code=400)
        # The daily report goes to the supervisor + safety groups only.
        if not any(getattr(cfg, g, "") for g in
                   ("LINE_OA_GROUP_SUPERVISOR", "LINE_OA_GROUP_SAFETY")):
            return JSONResponse(
                {"ok": False, "error": "ยังไม่ได้ตั้งค่า Group ID — ต้องมีกลุ่มหัวหน้างาน หรือ Safety อย่างน้อย 1 กลุ่ม"},
                status_code=400)

        stats = daily_stats_for_line(day)
        loop  = asyncio.get_running_loop()
        sent  = await loop.run_in_executor(None, send_daily_report, stats)
        if not sent:
            return JSONResponse(
                {"ok": False, "error": "LINE ปฏิเสธคำขอ — ตรวจสอบว่า Token ถูกต้องและบอทอยู่ในกลุ่มนั้น"},
                status_code=502)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ================================================================
# DATA COLLECTION (training dataset) + JOBS (train / upload)
# ================================================================
def _dir_size_mb(path: Path) -> float:
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    return round(total / (1024 * 1024), 1)


@app.get("/api/data/stats")
async def data_stats():
    cats = {}
    for cat in _DATA_CATEGORIES:
        d = COLLECTED_DIR / cat
        cats[cat] = len(list(d.glob("*.jpg"))) if d.exists() else 0
    return JSONResponse({
        "categories":   cats,
        "total_images": sum(cats.values()),
        "size_mb":      _dir_size_mb(COLLECTED_DIR),
        "labeled":      sum(
            1 for cat in _DATA_CATEGORIES
            for j in (COLLECTED_DIR / cat).glob("*.jpg")
            if j.with_suffix(".txt").exists()
        ) if COLLECTED_DIR.exists() else 0,
    })


@app.post("/api/data/clear")
async def data_clear(body: dict[str, Any] | None = None):
    body = body or {}
    cats = [body["category"]] if body.get("category") in _DATA_CATEGORIES else _DATA_CATEGORIES
    removed = 0
    for cat in cats:
        d = COLLECTED_DIR / cat
        if not d.exists():
            continue
        for f in list(d.glob("*.jpg")) + list(d.glob("*.txt")):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return JSONResponse({"ok": True, "removed_files": removed})


@app.post("/api/jobs/train")
async def jobs_train(body: dict[str, Any]):
    from server.jobs import manager as jobs
    task    = body.get("task", "ppe")
    if task not in ("ppe", "fall"):
        return JSONResponse({"ok": False, "error": "task ต้องเป็น ppe หรือ fall"}, status_code=400)
    args = ["training.trainer", "--task", task, "--export"]
    project = body.get("project")
    if project:
        args += ["--project", str(project)]
    ok, msg = jobs.start(args, label=f"เทรน {task.upper()}")
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.post("/api/jobs/upload")
async def jobs_upload(body: dict[str, Any]):
    from server.jobs import manager as jobs
    task = body.get("task", "ppe")
    if task not in ("ppe", "fall", "zone"):
        return JSONResponse({"ok": False, "error": "task ไม่ถูกต้อง"}, status_code=400)
    args = ["training.upload", "--task", task]
    project = body.get("project")
    if project:
        args += ["--project", str(project)]
    ok, msg = jobs.start(args, label=f"อัปโหลด {task.upper()} → Roboflow")
    return JSONResponse({"ok": ok, "message": msg}, status_code=200 if ok else 409)


@app.get("/api/jobs/status")
async def jobs_status():
    from server.jobs import manager as jobs
    return JSONResponse(jobs.status())


@app.get("/api/training/metrics")
async def training_metrics():
    """Latest persisted training metrics (mAP/precision/recall) per task."""
    logs_dir = ZENTRA_BACKEND / "logs"
    out: dict[str, Any] = {}
    if logs_dir.exists():
        for task in ("ppe", "fall"):
            files = sorted(logs_dir.glob(f"metrics_{task}_*.json"))
            if files:
                try:
                    out[task] = json.loads(files[-1].read_text(encoding="utf-8"))
                except Exception:
                    pass
    return JSONResponse(out)


@app.post("/api/jobs/stop")
async def jobs_stop():
    from server.jobs import manager as jobs
    return JSONResponse({"ok": jobs.stop()})


# ================================================================
# WEBSOCKET
# ================================================================
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    # Auth (when ZENTRA_API_TOKEN is set): the HTTP middleware can't see the WS
    # handshake, so gate here on the ?token= query param before accepting.
    if not _token_ok(websocket.query_params.get("token")):
        await websocket.close(code=1008)
        return
    await manager.connect(websocket)
    try:
        # Send initial status on connect (real values, not placeholders)
        modules = {"ppe": "error", "zone": "error", "fall": "error"}
        alerts  = {"total": 0, "warning": 0, "alert": 0, "emergency": 0}
        camera  = "disconnected"
        engine_error = "pipeline not initialised"
        if pipeline:
            with pipeline._lock:
                modules = dict(pipeline.status.get("modules", modules))
                alerts  = dict(pipeline.status.get("alerts", alerts))
                camera  = pipeline.status.get("camera", camera)
                engine_error = pipeline.status.get("engine_error")
        await websocket.send_json({
            "type":    "event",
            "event":   "status",
            "modules": modules,
            "alerts":  alerts,
            "camera":  camera,
            "engine_error": engine_error,
        })
        # Keep connection alive; frames arrive via FrameBroadcaster
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
