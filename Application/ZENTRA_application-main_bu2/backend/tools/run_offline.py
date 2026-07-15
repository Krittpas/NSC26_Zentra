#!/usr/bin/env python3
# tools/run_offline.py — Offline video harness for the tracking/association/zone
# engine. Video in → annotated video out + printed event log + tracking stats.
# The per-phase verification driver (test everything on a clip before the app).
#
# USAGE:
#   python -m tools.run_offline --video clip.mp4 --out annotated.mp4
#   python -m tools.run_offline --video clip.mp4 --max-frames 200
# ================================================================
from __future__ import annotations
import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2

import config as cfg
from utils.detect_track import Detector
from utils.ppe_association import associate, violations_of, CATEGORIES
from utils.temporal import TrackWindowConfirmer, CooldownGate
from utils import zone_geometry

# BGR colors
GREEN = (0, 210, 0)
RED = (0, 0, 220)
CYAN = (255, 190, 0)
GRAY = (160, 160, 160)


def _draw(frame, recs, confirmer):
    """Draw one clean box + PPE-status label per person (draw_person_status style)."""
    for rec in recs:
        p = rec["person"]
        tid = p["track_id"]
        x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        # confirmed violations only (per track+category)
        conf_viol = [c for c in violations_of(rec)
                     if tid is not None and confirmer.is_confirmed((tid, c))]
        color = RED if conf_viol else CYAN
        head = f"#{tid}" if tid is not None else "#?"
        status = ("NO " + ",".join(conf_viol)) if conf_viol else "OK"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{head} {status}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(p['foot'][0]), int(p['foot'][1])), 4, color, -1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="annotated output mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--zones", default=None, help="zones.json (normalized polygons)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15

    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    det = Detector()
    det.reset()
    confirmer = TrackWindowConfirmer(cfg.PPE_CONFIRM_FRAMES, cfg.PPE_CONFIRM_WINDOW)
    cooldown = CooldownGate(cfg.VIOLATION_COOLDOWN_SECONDS)
    zones = zone_geometry.load_zones(args.zones)
    zconfirm = TrackWindowConfirmer(cfg.ZONE_CONFIRM_FRAMES, cfg.ZONE_CONFIRM_WINDOW)
    zcooldown = CooldownGate(cfg.ZONE_COOLDOWN_SECONDS)
    print(f"loaded {len(zones)} zone(s): {[(z.name, z.type) for z in zones]}")

    frames = 0
    id_frame_count: dict[int, int] = defaultdict(int)   # track_id -> #frames seen
    per_frame_persons = []
    events = []
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        dets = det.track(frame, conf=args.conf)
        persons = [d for d in dets if d["is_person"]]
        per_frame_persons.append(len(persons))
        for p in persons:
            if p["track_id"] is not None:
                id_frame_count[p["track_id"]] += 1

        recs = associate(dets)
        live_keys = set()
        zlive_keys = set()
        for rec in recs:
            person = rec["person"]
            tid = person["track_id"]
            if tid is None:
                continue
            # exclusion zones mask a person out of BOTH ppe + zone logic
            if zone_geometry.in_any_exclusion(person, zones, w, h):
                continue
            # ── PPE violations (per track+category) ──
            viols = violations_of(rec)
            for cat in CATEGORIES:
                key = (tid, cat)
                live_keys.add(key)
                now_confirmed = confirmer.update(key, cat in viols)
                if now_confirmed and cat in viols and cooldown.ready(key):
                    events.append((frames, tid, "ppe", f"no_{cat}"))
                    print(f"  [frame {frames}] CONFIRMED VIOLATION: person #{tid} no_{cat}")
            # ── Zone intrusion (per track+zone) ──
            for z in zone_geometry.danger_hits(person, zones, w, h):
                zkey = (tid, z.id)
                zlive_keys.add(zkey)
                zc = zconfirm.update(zkey, True)
                if zc and zcooldown.ready(zkey):
                    events.append((frames, tid, "zone", z.name))
                    print(f"  [frame {frames}] ZONE INTRUSION: person #{tid} entered '{z.name}'")
        confirmer.gc(live_keys)
        zconfirm.gc(zlive_keys)

        # draw zone polygons
        for z in zones:
            poly = z.polygon_px(w, h)
            zc_col = (0, 0, 220) if z.type == "danger" else (120, 120, 120)
            cv2.polylines(frame, [poly], True, zc_col, 2)
            cv2.putText(frame, z.name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, zc_col, 1)
        frame = _draw(frame, recs, confirmer)
        cv2.putText(frame, f"frame {frames} | persons {len(persons)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer:
            writer.write(frame)
        if args.max_frames and frames >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()

    dt = time.time() - t0
    print(f"\n=== tracking stats ({frames} frames, {dt:.1f}s, {frames/max(dt,1e-9):.1f} fps) ===")
    print(f"unique track IDs: {len(id_frame_count)}")
    print(f"avg persons/frame: {sum(per_frame_persons)/max(frames,1):.2f}")
    print(f"frames-per-ID (want few IDs each lasting many frames — low=ID churn):")
    for tid, c in sorted(id_frame_count.items()):
        print(f"   #{tid}: {c} frames")
    if args.out:
        print(f"annotated video → {args.out}")


if __name__ == "__main__":
    main()
