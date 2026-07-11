#!/usr/bin/env python3
# tools/live_stream.py — Live RTSP → engine (ByteTrack + PPE association + zone)
# → MJPEG stream in the browser (real-time, no GUI-window/permission issues).
#
# USAGE:
#   python -m tools.live_stream --src "rtsp://user:pass@ip:554/stream" --zones zones.json
#   then open http://localhost:8008/  in a browser
# ================================================================
from __future__ import annotations
import argparse
import threading
import time

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

import config as cfg
from utils.detect_track import Detector
from utils.ppe_association import associate, violations_of, CATEGORIES
from utils.temporal import TrackWindowConfirmer, CooldownGate
from utils import zone_geometry

GREEN, RED, CYAN = (0, 210, 0), (0, 0, 220), (255, 190, 0)

_latest_jpeg: bytes | None = None
_stats = {"frames": 0, "persons": 0, "events": 0, "fps": 0.0}
_lock = threading.Lock()


def _annotate(frame, recs, confirmer, zones, w, h):
    for z in zones:
        poly = z.polygon_px(w, h)
        col = RED if z.type == "danger" else (140, 140, 140)
        cv2.polylines(frame, [poly], True, col, 2)
        cv2.putText(frame, z.name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for rec in recs:
        p = rec["person"]; tid = p["track_id"]
        x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        cviol = [c for c in violations_of(rec) if tid is not None and confirmer.is_confirmed((tid, c))]
        color = RED if cviol else CYAN
        status = ("NO " + ",".join(cviol)) if cviol else "OK"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"#{tid} {status}", (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (int(p['foot'][0]), int(p['foot'][1])), 4, color, -1)
    return frame


def _worker(src, zones_path, conf, imgsz):
    global _latest_jpeg
    det = Detector(); det.reset()
    confirmer = TrackWindowConfirmer(cfg.PPE_CONFIRM_FRAMES, cfg.PPE_CONFIRM_WINDOW)
    cooldown = CooldownGate(cfg.VIOLATION_COOLDOWN_SECONDS)
    zones = zone_geometry.load_zones(zones_path)
    zconf = TrackWindowConfirmer(cfg.ZONE_CONFIRM_FRAMES, cfg.ZONE_CONFIRM_WINDOW)
    zcool = CooldownGate(cfg.ZONE_COOLDOWN_SECONDS)
    print(f"[live] opening {src} … zones={len(zones)}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[live] ❌ cannot open source: {src}"); return
    t0, fcount = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[live] read failed, retrying in 1s"); time.sleep(1)
            cap.release(); cap = cv2.VideoCapture(src); continue
        h, w = frame.shape[:2]
        dets = det.track(frame, conf=conf, imgsz=imgsz)
        recs = associate(dets)
        live, zl = set(), set()
        for rec in recs:
            tid = rec["person"]["track_id"]
            if tid is None:
                continue
            if zone_geometry.in_any_exclusion(rec["person"], zones, w, h):
                continue
            viols = violations_of(rec)
            for cat in CATEGORIES:
                k = (tid, cat); live.add(k)
                if confirmer.update(k, cat in viols) and cat in viols and cooldown.ready(k):
                    _stats["events"] += 1
                    print(f"[live] VIOLATION person #{tid} no_{cat}")
            for z in zone_geometry.danger_hits(rec["person"], zones, w, h):
                zk = (tid, z.id); zl.add(zk)
                if zconf.update(zk, True) and zcool.ready(zk):
                    _stats["events"] += 1
                    print(f"[live] ZONE INTRUSION person #{tid} → {z.name}")
        confirmer.gc(live); zconf.gc(zl)
        frame = _annotate(frame, recs, confirmer, zones, w, h)
        fcount += 1
        with _lock:
            _stats["frames"] = fcount
            _stats["persons"] = sum(1 for d in dets if d["is_person"])
            if fcount % 10 == 0:
                _stats["fps"] = round(fcount / (time.time() - t0), 1)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            globals()["_latest_jpeg"] = buf.tobytes()


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    return """<body style='margin:0;background:#0d1b2a;color:#7ecfff;font-family:sans-serif'>
    <h3 style='padding:8px'>ZENTRA live — ByteTrack + PPE + Zone</h3>
    <img src='/stream' style='max-width:100%'></body>"""


def _mjpeg():
    while True:
        j = _latest_jpeg
        if j:
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
        time.sleep(0.04)


@app.get("/stream")
def stream():
    return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="rtsp url / file / webcam index")
    ap.add_argument("--zones", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()
    src = int(args.src) if args.src.isdigit() else args.src
    threading.Thread(target=_worker, args=(src, args.zones, args.conf, args.imgsz),
                     daemon=True).start()
    print(f"[live] open http://localhost:{args.port}/  in your browser")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
