---
name: zentra-dev
description: >
  Use when running, debugging, or extending the ZENTRA Safety AI system — the
  PyWebView + FastAPI desktop app that wraps the AI engine, all in ONE repo
  (Application/ZENTRA_application-main). Covers launching it, the in-process
  ultralytics engine (PPE / Safety Zone / Fall — NO Docker inference server),
  the missing-PPE-model behaviour, the mediapipe/protobuf trap, the Edge WebView2
  gotchas that cause blank/stuck screens, the SPA router contract, the Pipeline
  integration, per-camera roles, and the in-app training workflow. Read this
  before touching app.py, server/*, pipeline/*, ui/*, or backend/utils · training.
---

# ZENTRA Desktop Application — Developer Skill

A native desktop app (PyWebView + FastAPI) with a 6-screen dark web UI, wrapping
the AI engine. **Single repo:** `app.py`, `server/`, `pipeline/`, `ui/`, `data/`,
and `backend/` (the AI engine) all live under `Application/ZENTRA_application-main/`.
There is **no separate backend repo** and **no Docker inference server** — every
model runs in-process via ultralytics (this replaced an older Roboflow `:9001`
architecture; nothing listens on 9001).

## Run it

```
cd Application/ZENTRA_application-main
python app.py                 # → http://127.0.0.1:7788/  + PyWebView window
```
`app.py` forces UTF-8 stdout (the engine prints emoji/Thai → would crash a cp1252
console), starts **uvicorn in a daemon thread** (`127.0.0.1:7788`), waits ~2s,
then opens a **fullscreen PyWebView** window. Flow: Splash → Source → Dashboard →
(Cameras / Zone Editor / History / Settings). `run_zentra.ps1` just launches the
app (no Docker). It runs **without** a LINE token (events still show in-app).

## The AI engine (all in-process via ultralytics)

| Module | How | Model | Alert level |
|--------|-----|-------|-------------|
| **PPE** | person → match PPE items by box containment | `ppe_finetuned.pt` (items) | `warning` |
| **Safety Zone** | foot-point in polygon, 3-of-5 confirm, rising-edge | (geometry — no model) | `alert` |
| **Fall** | pose → 30-frame skeleton → TFLite Transformer + rule layer | `yolo11n-pose.pt` + `.tflite` | `emergency` |

- **Person detection + tracking is decoupled from PPE:** a COCO `yolo11s.pt` finds
  + ByteTrack-tracks people (high recall, stable ids); the PPE fine-tune only
  classifies items. Everything (PPE/Zone/Fall) stands on the person box.
- **Fall pose backend defaults to `yolo`** (`yolo11n-pose`, multi-person, better on
  distant workers). MediaPipe is an *optional* backend (`FALL_POSE_BACKEND=mediapipe`).
- Models live in `backend/models/` (**git-ignored**). `yolo11s`/`yolo11n-pose`
  auto-download from ultralytics; `ppe_finetuned.pt` is yours and cannot.
- **Missing `ppe_finetuned.pt` → PPE pauses (status `off`), it does NOT crash the
  engine** — person/zone/fall keep running (utils/ppe_engine `__init__`). A totally
  failed engine falls back to passthrough and reports `engine_error` loudly, because
  clean video with no boxes must never read as "everyone is compliant".

## Per-camera roles (common gotcha)

`data/settings.json` → `cameras.<id>.roles` gates which modules run/draw for that
camera, e.g. `{"cam0": {"roles": ["zone","fall"], "ppe_items": ["helmet"]}}`.
- Roles read at pipeline **start** (and hot-applied on Settings save).
- **If roles don't include a module that draws person boxes (zone/fall/ppe-with-
  model), no boxes appear** — a camera set to `["ppe"]` with no PPE model shows
  nothing. `roles` absent/`null` = all modules on.

## ⚠️ mediapipe / protobuf trap (only if FALL_POSE_BACKEND=mediapipe)

Needs `mediapipe==0.10.14` **and** `protobuf>=4.25.3,<5`.
- `module 'mediapipe' has no attribute 'solutions'` → `pip install --force-reinstall --no-deps mediapipe==0.10.14`
- `'FieldDescriptor' object has no attribute 'label'` → `pip install "protobuf>=4.25.3,<5"`

The default `yolo` backend needs none of this.

## ⚠️ Edge WebView2 traps (these cost real debugging time — don't repeat)

1. **`innerHTML = html` does NOT execute `<script>` tags.** The SPA router injects
   screen HTML; `app.js navigate()` re-creates each `<script>` as a fresh element so
   it runs in **global** scope (so `window['init_<screen>']` is found). `eval()`
   would define it locally → screen looks stuck.
2. **Top-level `let`/`const` in screen scripts → use `var`.** Re-navigating re-runs
   the script; a second top-level `let X` throws `already declared`. `var` is fine.
3. **External CDN scripts must be awaited.** `navigate()` awaits a `<script src>`
   `onload` (tracked via `data-cdn`) before calling `init_*` (e.g. Chart.js).
4. **SSE (`EventSource`) is unreliable in WebView2** — buffers, may never fire.
   Use WebSocket or polling. (Splash uses a local timer, not SSE.)

## SPA router contract (`ui/assets/app.js`)

- `ZENTRA.navigate(id)` fetches `/ui/screens/<id>.html`, injects it, re-executes
  scripts, then calls `window['init_<id>']()`. Main screens get the left sidebar.
- Global state on `ZENTRA.state` (`pipeline`, `modules`, `alerts`, `camera`,
  `recentAlarms`, …). WS messages: `{type:'frame', data:<b64 jpeg>}` → `#video-feed`;
  `{type:'event', event:'status'|'alert', ...}` → dots / counters / banners.
- If a token is set (`ZENTRA_API_TOKEN`), app.js attaches it to every fetch + WS.

## Architecture

```
app.py ── daemon thread: uvicorn → server/api.py (FastAPI)
       └─ main thread:  PyWebView fullscreen window → 127.0.0.1:7788

server/api.py ── @lifespan: build Pipeline(), wire on_alert → store + WS + LINE push,
  │             start FrameBroadcaster (frames → WS), periodic PDPA retention purge
  ├─ REST: /api/status · /api/pipeline/{start,stop} · /api/zones · /api/settings
  │        /api/frame/snapshot · /api/history/* · /api/report/* · /api/jobs/*
  └─ WS:   /ws/stream (frames + events, multiplexed by msg.type)

pipeline/pipeline.py — decoupled loops: process loop (display @ camera fps, draws
  latest boxes) · detect worker (heavy inference on newest frame) · fall loop
  (fixed cadence FALL_LOOP_FPS). Builds PPEEngine; falls back to passthrough on
  failure. FrameBroadcaster: latest frame → JPEG → base64 → WS.
```

## Data contracts

- **`data/zones.json`**: `[{id, name, color, points:[[x,y]…], type:"danger"|"exclusion",
  camera_id, enabled}]`. **Points are NORMALIZED 0–1** (scaled to the frame at
  runtime) — this fixes the editor-canvas ↔ camera-frame size mismatch. After any
  zone CRUD the API calls `pipeline.reload_zones()` (no restart).
- **`data/settings.json`**: merged over `SETTINGS_DEFAULTS` in `api.py`; saving calls
  `pipeline.apply_settings()` to push values into `config` + engine at runtime. The
  LINE channel token is redacted on GET and only overwritten when non-empty on POST.

## Zone detection behaviour (common confusion)

- Fires on a detected **person** whose **foot point** (bottom-center of bbox) is in
  the polygon — NOT on motion; a hand/object isn't a person.
- 3-of-5 temporal confirm + **rising-edge** alert: a fresh entry (outside→inside)
  alerts immediately, a continuous stay re-alerts every `ZONE_COOLDOWN_SECONDS`.
- The Zone Editor draws on a live snapshot (`/api/frame/snapshot`) — **connect a
  camera first**, or it shows a dark pixel.

## Thai overlay text

On-frame labels (zone names, PPE status, "ล้ม!") are shaped with **harfbuzz +
freetype** (`utils/ppe_engine._shape_text_img`) — Pillow alone mis-stacks Thai
vowel+tone marks. Optional deps (`uharfbuzz`, `freetype-py`); absent → plain Pillow
fallback (ASCII fine, Thai unshaped). UI text (HTML) is shaped by the browser.

## In-app training (Settings → data)

`server/jobs.py` runs train/upload as a **separate subprocess** (`python -m
training.*`, cwd = `backend/`), one at a time, streaming stdout. Never train inside
the web loop. Real training needs a CUDA GPU (this dev box may be CPU-only → use
Colab). See `docs/TRAINING_PIPELINE.md` and `notebooks/`.

## Where to look when X breaks

| Symptom | Likely cause / file |
|---|---|
| Toast "ระบบ AI ไม่ทำงาน" | engine failed to build (missing person model?) — `pipeline._build_engine` |
| Video, but NO boxes | camera `roles` exclude every drawing module → `data/settings.json` cameras.<id>.roles |
| Screen blank / stuck | script not re-injected — `app.js navigate()` |
| Screen breaks on 2nd visit | top-level `let`/`const` in that screen → `var` |
| Thai overlay garbled | `uharfbuzz`/`freetype-py` missing → falls back to unshaped Pillow |
| Zone won't re-alert | rising-edge state — `_zone_inside` in `ppe_engine.detect` |
| Emoji/Thai console crash | run via `app.py` (forces UTF-8 stdout) |

## Conventions

- Thai UI copy; font Sarabun (loaded via Google Fonts CDN — allowed).
- Design tokens in `ui/assets/style.css`.
- API is unauthenticated by default → bound to 127.0.0.1. Add auth before exposing.
