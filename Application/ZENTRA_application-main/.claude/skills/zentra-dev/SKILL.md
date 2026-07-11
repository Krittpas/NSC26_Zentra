---
name: zentra-dev
description: >
  Use when running, debugging, or extending the ZENTRA Safety AI system
  (the PyWebView + FastAPI desktop app in c:\ZENTRA\ZENTRA_application that
  wraps the AI backend in c:\ZENTRA\ZENTRA). Covers launching it, the Docker
  inference server that PPE/Zone need, the mediapipe/protobuf version trap,
  the Edge WebView2 gotchas that cause blank/stuck screens, the SPA router
  contract, the Pipeline integration, the in-app data-collection + training
  workflow, and zone-detection behaviour. Read this before touching app.py,
  server/*, pipeline/*, ui/*, or the backend modules/training so you don't
  re-discover the same traps.
---

# ZENTRA Desktop Application — Developer Skill

A native Windows desktop app that wraps the **existing, unchanged** ZENTRA
AI backend (`c:\ZENTRA\ZENTRA`) in a 6-screen dark-themed GUI.

- **App layer** (this repo): `c:\ZENTRA\ZENTRA_application` — repo `ZENTRA_application`
- **AI backend** (separate): `c:\ZENTRA\ZENTRA` — PPE / Safety Zone / Fall modules

## Run it

```
cd c:\ZENTRA\ZENTRA_application
.\run_zentra.ps1          # or double-click run_zentra.bat
```

`run_zentra.ps1` ensures the Docker inference server is up on :9001 (starts
Docker Desktop + the container if needed) **then** runs `python app.py`.
Plain `python app.py` also works if the inference server is already running.

`app.py` forces UTF-8 stdout (backend prints emoji/Thai → would crash a
cp1252 console), starts **uvicorn in a daemon thread** (`127.0.0.1:7788`),
waits ~2s, then opens a **PyWebView** window (Edge WebView2). Flow:
Splash → Source Selection → Live Dashboard → (Zone Editor / History / Settings).

It runs **without** a LINE token (events still show in-app). It runs without
the inference server too, but then **PPE + Zone produce nothing** (only
MediaPipe pose works) — see next section.

## AI runtime — what each module needs

| Module | Tech | Needs |
|--------|------|-------|
| PPE Detection | YOLO via Roboflow | inference server on `localhost:9001` |
| Safety Zone | ByteTrack + polygon | inference server (uses the `person` class from PPE preds) |
| Heat-Stroke / Fall | MediaPipe Pose (+ Roboflow fall fallback) | `mediapipe` (local, no server) |

**Inference server (Docker):**
```
docker run -d --name zentra-inference --restart unless-stopped -p 9001:9001 \
  roboflow/roboflow-inference-server-cpu:latest
```
Verify: `curl http://localhost:9001/` → HTML 200. The client + model IDs +
API key all default correctly in `config.py` (`ppe-cpxsz/2`,
`fall-detection-ovjqo/5`, server `http://localhost:9001`). GPU image
(`...-gpu`, `--gpus all`) is faster but needs WSL2 GPU set up; CPU image is
the reliable default.

**⚠️ mediapipe / protobuf version trap (cost real time):** MediaPipe pose
needs `mediapipe==0.10.14` **and** `protobuf>=4.25.3,<5`. Symptoms:
- `module 'mediapipe' has no attribute 'solutions'` → broken/incomplete
  mediapipe wheel → `pip install --force-reinstall --no-deps mediapipe==0.10.14`
- `'FieldDescriptor' object has no attribute 'label'` → protobuf too new →
  `pip install "protobuf>=4.25.3,<5"`
These pins are in the backend `requirements.txt`. When MediaPipe is
unavailable the code falls back to the Roboflow fall model (still works).

## Architecture

```
app.py
├─ daemon thread: uvicorn → server/api.py (FastAPI)
└─ main thread:   PyWebView window → http://127.0.0.1:7788/

server/api.py (FastAPI)
├─ @startup: build Pipeline(), wire on_alert → WS broadcast + event history,
│            start FrameBroadcaster (frames → WS @10fps)
├─ REST: /api/status /api/pipeline/{start,stop} /api/zones(CRUD)
│        /api/settings /api/frame/snapshot /api/history/*
└─ WS  : /ws/stream  (frames + events, multiplexed by msg.type)

pipeline/pipeline.py — Pipeline class (wraps the backend's main.py loop)
├─ start(src_cfg) / stop() / is_running()
├─ _process_loop (daemon): read → infer(worker) → annotate → store frame
│   passes window_title="" to modules so cv2.imshow() is skipped
├─ get_latest_frame() / get_snapshot() / get_uptime()
├─ reload_zones() / apply_settings()
└─ monkey-patches send_line_notify in ALL module namespaces → on_alert cb

pipeline/frame_broadcaster.py — daemon: latest frame → resize → JPEG →
                                base64 → manager.broadcast() via the loop
```

## ⚠️ Edge WebView2 traps (these cost real debugging time — don't repeat)

1. **`element.innerHTML = html` does NOT execute `<script>` tags.**
   The SPA router (`ZENTRA.navigate`) injects screen HTML, so each screen's
   `init_<screen>()` would never run. Fix already in `app.js`: after setting
   innerHTML, **re-create** every `<script>` as a fresh element and append it
   so the browser executes it.

2. **Use script-element injection, NOT `eval()`, for inline scripts.**
   Direct `eval(text)` runs in the *local* scope of `navigate()`, so
   `function init_x(){}` is defined locally and `window['init_x']` stays
   undefined → screen looks stuck. A re-created `<script>` element executes in
   **global** scope, which is what the router needs.

3. **External CDN scripts must be awaited.** Chart.js (history page) is loaded
   by appending a `<script src>` to `<head>`; `navigate()` awaits its `onload`
   before calling `init_history()`, else `Chart` is undefined. Loaded once,
   tracked via `data-cdn` attribute.

4. **Top-level `let`/`const` in screen scripts → use `var`.** Classic scripts
   share one global lexical environment. Re-navigating to a screen re-runs its
   script; a second top-level `let X`/`const X` throws
   `SyntaxError: already declared`, killing that screen. `var` redeclaration is
   allowed. (Variables *inside* `init_*()` functions are fine.)

5. **SSE (`EventSource`) is unreliable inside WebView2** — it buffers and may
   never fire `onmessage`. The splash progress bar was switched from an SSE
   stream to a local `setTimeout` animation. Avoid SSE for anything the UI
   depends on; prefer WebSocket or polling.

## SPA router contract (`ui/assets/app.js`)

- `ZENTRA.navigate(screenId)` fetches `/ui/screens/<id>.html`, injects it,
  re-executes scripts, then calls `window['init_<id>']()`.
- Every screen file is an HTML fragment + a `<script>` that defines
  `function init_<id>() { ... }`. Main screens call
  `renderNavbar('<id>')` into `#navbar-mount`.
- Global state lives on `ZENTRA.state` (`pipeline`, `modules`, `alerts`,
  `uptime`, `last_emergency`, `camera_label`).
- WebSocket messages: `{type:'frame', data:<base64 jpeg>}` updates
  `#video-feed`; `{type:'event', ...}` updates module dots / alert counters /
  emergency banner.

## Pipeline integration rules

- **Never edit the AI backend's logic.** The only allowed change is the
  `if window_title:` guard before `cv2.imshow()` in `modules/ppe.py` and
  `modules/safety_zone.py` (so the headless pipeline doesn't pop OpenCV
  windows from a background thread).
- The pipeline mirrors `main.py`'s threading model (`FrameReader`,
  `InferenceWorker`, process loop) but stores the annotated frame in
  `self._latest_frame` instead of `cv2.imshow()`.
- **asyncio/threading bridge:** the FrameBroadcaster and the alert callback
  run on threads but must talk to async WebSockets — they use
  `asyncio.run_coroutine_threadsafe(manager.broadcast(...), _loop)` where
  `_loop` is captured in the FastAPI `@startup` handler.
- **Alerts:** `send_line_notify` is monkey-patched in every module namespace
  (`ppe`, `safety_zone`, `heat_stroke`, and `alerts.line_notify`) because each
  module did `from alerts.line_notify import send_line_notify` (binds a local
  name). Patch must hit all of them or some alerts won't reach the UI.

## Data contracts

- `data/zones.json`: list of `{id, name, color, points:[[x,y],...], enabled}`.
  Points are **raw pixel coords** at camera resolution. The backend's
  `safety_zone._load_zones()` reads `points`/`name` and ignores the rest, so
  the format is compatible. After any zone CRUD, the API calls
  `pipeline.reload_zones()` → `safety_zone._load_zones()` (no restart).
- `data/settings.json`: merged over `SETTINGS_DEFAULTS` in `api.py`. Saving
  calls `pipeline.apply_settings()` to push values into `config` at runtime.

## Zone detection behaviour (common confusion)

- Zone intrusion fires on a detected **`person`** whose **bbox centre** is
  inside the polygon — NOT on motion. Waving a hand ≠ a person; the whole
  body/torso must be visible and its centre inside the zone.
- The Zone Editor draws on a **live camera snapshot** (`/api/frame/snapshot`).
  If the pipeline isn't running it returns a 1×1 dark pixel → the editor now
  checks `/api/status`, shows a "connect a camera first" overlay, and retries
  the snapshot up to 10×. **You must connect a camera (see video on the
  Dashboard) before the editor shows an image.**
- Coordinates are consistent because the snapshot, inference frame, and zone
  points are all in the same raw camera resolution, and the webcam flip is
  applied before both inference and annotation.
- Verified on-device: snapshot returns a real JPEG and intrusion counts rise
  as a person enters — so a "blank editor" or "no effect" almost always means
  the pipeline isn't running / no person detected, not a code bug.

## Training workflow (accuracy) + in-app jobs

Goal: fine-tune on **your own footage with corrected labels** — that, not
epoch count, is what raises accuracy. Default models are public Roboflow
models, untuned for the site.

Pipeline (backend `c:\ZENTRA\ZENTRA`, see `TRAINING.md`):
1. **Collect** — running the app auto-saves frames+YOLO labels to
   `data/collected/{ppe_violations,zone_intrusions,fall_events,normal}/`
   via `utils/collector.py`.
2. **Upload** — `python -m training.upload --task ppe` → Roboflow.
3. **Label** ⭐ — correct boxes in Roboflow Annotate (the real accuracy driver).
4. **Train** — `python -m training.trainer --task ppe --project <slug> --export`
   → `models/ppe_finetuned.pt`. GPU verified working (RTX 3050, CUDA).
5. **Deploy** — set `USE_LOCAL_MODEL=true` in `.env`.

**In-app (Settings → "เกี่ยวกับข้อมูล"):** live collected counts, upload,
train (task/source/project), live progress (epoch x/y, mAP50, log tail),
clear-data. Backed by:
- `server/jobs.py` — `JobManager` runs train/upload as a **separate Python
  subprocess** (`sys.executable -m ...`, cwd = backend, `PYTHONUTF8=1`), one
  at a time, streams stdout to a ring buffer, parses epoch/mAP. Never run
  heavy training inside the web loop.
- API: `/api/data/stats`, `/api/data/clear`, `/api/jobs/{train,upload,status,stop}`.
- 4 GB GPU: `.env` pins `YOLO_BASE_MODEL=yolov8s.pt`, `TRAIN_BATCH_SIZE=4`
  (yolov8m OOMs). `config.TRAIN_AUG` must exist (used by `trainer.train`).

## Backend layout & config (`c:\ZENTRA\ZENTRA`)

- Runtime code at root: `config.py`, `main.py`, `modules/`, `utils/`,
  `alerts/`, `reports/`, `training/`. Non-code moved to `docs/` (proposals,
  slides, notebooks) and `archive/` (old snapshots, dead code).
- `config.py` loads its own `.env` explicitly (`load_dotenv(Path(__file__).parent/".env")`)
  so it works regardless of the app's cwd. `.env` is git-ignored;
  `.env.example` has placeholders (no real API key — credential-leak guard).
- The backend is a **separate git repo** (`Krittpas/Zentra`); the app repo is
  `ThePpoon/ZENTRA_application`. Commit backend changes locally, don't push
  them to the app remote. The `if window_title:` guards live in the backend.

## Where to look when X breaks

| Symptom | Likely cause / file |
|---|---|
| Screen blank / stuck on splash | script not executing — `app.js navigate()` script re-injection |
| Screen breaks on 2nd visit | top-level `let`/`const` in that screen → `var` |
| No video on dashboard | pipeline not running, or FrameBroadcaster/WS down — check `/api/status`, server console |
| Chart missing on history | Chart.js `onload` not awaited, or CDN blocked |
| Alerts not in UI | monkey-patch missed a module namespace |
| RTSP/file won't connect | body field names must be `rtsp_url`/`video_file_path`/`webcam_index` (match `api.py`) |
| OpenCV window pops up | a module's `cv2.imshow()` missing the `if window_title:` guard |
| PPE/Zone detect nothing | inference server down → `docker start zentra-inference` / run launcher; check `curl localhost:9001` |
| Zone Editor shows no image | pipeline not running → connect a camera first (snapshot needs a live frame) |
| Zone "no effect" on movement | detection is person-based; a full body's centre must be inside the polygon, not a hand |
| `mediapipe has no attribute 'solutions'` | reinstall `mediapipe==0.10.14 --no-deps` |
| `FieldDescriptor ... no attribute 'label'` | `pip install "protobuf>=4.25.3,<5"` |
| Training crashes immediately | `cfg.TRAIN_AUG` missing, or CUDA OOM → lower `TRAIN_BATCH_SIZE`, use `yolov8n/s.pt` |
| Emoji/Thai console crash | run via launcher; `app.py` forces UTF-8 stdout |

## Conventions

- Commit per phase, push to `origin/main` (`https://github.com/ThePpoon/ZENTRA_application.git`).
- Thai UI copy; font is Sarabun (Google Fonts CDN). Online CDN is allowed.
- Design tokens in `ui/assets/style.css`: `--bg:#0d1b2a`, `--accent:#7ecfff`,
  `--green:#4ade80`, `--red:#f87171`, `--emergency:#ef4444`.
