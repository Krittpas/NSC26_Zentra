# ZENTRA — Safety AI

Factory safety monitoring: **PPE compliance**, **danger-zone intrusion**, and
**fall detection** from a webcam, RTSP camera, or video file. All inference runs
on-device; no image leaves the machine.

The app is a FastAPI server plus a 6-screen web UI, wrapped in a native desktop
window (PyWebView). It also runs headless as a plain web app.

## Layout (single repo)

```
ZENTRA/
├─ app.py              desktop entry point (PyWebView window + uvicorn)
├─ server/             FastAPI: api.py · store.py (SQLite) · jobs.py · report.py
├─ pipeline/           Pipeline (camera → detect → draw) + FrameBroadcaster
├─ ui/                 SPA: index.html + screens/ + assets/
├─ data/               zones.json · settings.json · zentra.db · snapshots/
└─ backend/            the AI engine
    ├─ config.py       all tunables (reads backend/.env)
    ├─ utils/          ppe_engine · detect_track · fall_detector · zone_geometry
    ├─ models/         *.pt weights (git-ignored — see "Models" below)
    ├─ training/       trainer · autolabel · dedupe · upload
    └─ assets/         bytetrack_zentra.yaml · fall Transformer (.tflite) · fonts
```

## The 3 modules

| Module | How it works | Alert level |
|--------|--------------|-------------|
| **PPE** | COCO YOLO finds + tracks people; a fine-tuned YOLO finds PPE items; items are matched to people by box containment | `warning` |
| **Safety Zone** | point-in-polygon on each person's foot point, 3-of-5 temporal confirm | `alert` |
| **Fall** | YOLO-pose → 30-frame skeleton sequence → TFLite Transformer, plus a rule layer for distant workers | `emergency` |

All three run in-process via **ultralytics**. There is **no Roboflow inference
server** and nothing listens on `:9001` — that was an older architecture.

## Models (required — not in git)

Weights are git-ignored, so a fresh clone has **no PPE model**. The app now
refuses to pretend: every module reports `error`, and the UI shows
"ระบบ AI ไม่ทำงาน — ไม่มีการตรวจจับใด ๆ". Clean video with no boxes must never be
mistaken for "nobody is violating anything".

Put the weights in `backend/models/`:

| File | Purpose |
|------|---------|
| `ppe_finetuned.pt` | PPE items (`PPE_LOCAL_MODEL`) |
| `yolo11s.pt` | person detection + tracking (`PERSON_MODEL`) |
| `yolo11n-pose.pt` | pose keypoints for fall (`FALL_POSE_MODEL`) |

The person and pose models auto-download from ultralytics if missing; the PPE
fine-tune cannot — it is yours. Vendor all three for an offline factory box.

## Run it

**macOS (native, uses the Apple GPU via MPS — ~30 fps):**
```bash
./run_native.sh                 # http://127.0.0.1:7788/
STREAM_FPS=60 ./run_native.sh
```

**Desktop window:**
```bash
python app.py
```

**Docker (Linux factory host):**
```bash
docker compose up -d            # http://127.0.0.1:7788/
```
> Docker Desktop on macOS is CPU-only (~7 fps) and cannot reach a USB webcam.
> On a Mac, use `run_native.sh`. RTSP works fine in Docker anywhere.

## First-time setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # app: fastapi, uvicorn, pywebview
pip install -r backend/requirements.txt  # AI: ultralytics, torch, opencv, tflite
cp backend/.env.example backend/.env     # optional: LINE tokens, RTSP URL
```

> **MediaPipe pin:** only needed if you set `FALL_POSE_BACKEND=mediapipe` (the
> default is `yolo`). It requires `mediapipe==0.10.14` + `protobuf>=4.25.3,<5`;
> newer protobuf breaks pose with `FieldDescriptor ... has no attribute 'label'`.

## Security — read before exposing this

**The API has no authentication.** Anything that can reach port 7788 can read
worker evidence photos, wipe the event log, and start training subprocesses.

So it binds to **localhost only** — both `app.py` and `docker-compose.yml`. To
reach it from another machine, put an authenticating reverse proxy in front.
Do not widen the binding. There is deliberately **no CORS middleware**: the UI is
same-origin, and a wildcard `Access-Control-Allow-Origin` would let any website
the operator visits read `/api/settings`.

The LINE channel access token is a bearer credential and is **never returned** by
`GET /api/settings`. Leaving the token field blank in Settings keeps the stored one.

## LINE alerts (optional)

Set the channel token + group IDs in Settings → **การแจ้งเตือน Line**, or in
`backend/.env`. Alerts go out per level (`warning` → supervisor, `alert` → safety,
`emergency` → all three). With LINE unconfigured the system still detects, stores,
and displays everything — only the push is skipped, and History records
`line_sent = false` honestly.

**PDPA:** `upload_images` is **off** by default, so alerts are text-only and no
person image leaves the device. Turning it on uploads evidence photos to an
external public host (LINE requires a public HTTPS URL). Events and snapshots are
stored locally and auto-purged after `data.retention_days` (default 90).

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| Toast: "ระบบ AI ไม่ทำงาน" | weights missing from `backend/models/` — see **Models** |
| Video plays, no boxes, no toast | pipeline stopped; press Start on the Source screen |
| Nothing detected in Docker on Mac | Docker is CPU-only + no webcam → use `./run_native.sh` |
| Train/Upload button fails | jobs run `python -m training.*` with cwd `backend/` |
| Emoji/Thai console crash | `app.py` forces UTF-8 stdout |
| App stuck on splash | hard-refresh; see `.claude/skills/zentra-dev` (WebView2 notes) |

## Training

See [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md) — a staged plan
(person → zone → helmet → vest → gloves → glasses → boots → fall) where each
stage is gated on a human quality review before the next begins.
