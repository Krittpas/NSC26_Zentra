#!/usr/bin/env python3
"""Make URFD clips usable as a LIVE DEMO in the app.

The problem
-----------
URFD clips are 3-4 seconds long and are cut the moment the person hits the floor.
In `fall_eval.py` that is handled with `--hold-last`, which extends the final frame
so the "stayed down" timer can finish. The app has no such thing: it LOOPS a video
file, so ~1 second after landing the person teleports back to standing.

FALL_MOTIONLESS_SEC is 2.0 s. A person who is only on the floor for 1 second before
the video rewinds can never satisfy it — the app will never raise a fall alert on a
raw URFD clip, no matter how well the detector works. That is a property of the
CLIP, not of the system.

A real factory camera does not rewind. The person stays on the floor. So this
freezes the final frame — where URFD's own annotation says the subject is still on
the ground (label 1) — for `--hold` seconds, and writes a clip that behaves the way
a real camera would.

This is a DEMO/TEST aid. It asserts nothing that URFD's annotation does not already
say, but it IS a synthetic extension of real footage: never quote a recall number
measured on these. Score with `fall_eval.py` on the originals.

    python scripts/make_demo_clips.py
    → backend/data/demo_clips/fall-02-cam0-demo.mp4   (3.7s + 8s on the floor)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

REPO = Path(__file__).resolve().parent.parent
# The seven clips fall_eval proved the system alerts on. Demo with these, not with a
# random clip — a miss on a clip the system was never going to catch tells you
# nothing except that you picked the wrong clip.
PROVEN = ["fall-02", "fall-06", "fall-07", "fall-10", "fall-11", "fall-12", "fall-13"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPO / "backend/data/fall_clips")
    ap.add_argument("--out", type=Path, default=REPO / "backend/data/demo_clips")
    ap.add_argument("--hold", type=float, default=10.0,
                    help="วินาทีที่ให้คน 'นอนอยู่กับพื้น' ต่อ (เหมือนกล้องจริง)")
    ap.add_argument("--lead", type=float, default=20.0,
                    help="วินาทีที่ให้คน 'ยืนรอ' ก่อนล้ม — ต้องมี เพราะแอปใช้เวลา "
                         "~10 วินาทีโหลดโมเดล ถ้าไม่มี lead ระบบจะตื่นมาตอนคนนอนไปแล้ว")
    ap.add_argument("--clips", nargs="*", default=PROVEN)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for stem in args.clips:
        src = args.src / f"{stem}-cam0.mp4"
        if not src.exists():
            print(f"  {stem}: ไม่พบไฟล์ — ข้าม")
            continue
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            print(f"  {stem}: อ่านเฟรมไม่ได้ — ข้าม")
            continue

        h, w = frames[0].shape[:2]
        dst = args.out / f"{stem}-cam0-demo.mp4"
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # LEAD-IN: the person standing, held. Two things need it.
        #  1. The app spends ~10 s loading models. Without a lead the clip is over
        #     before detection even starts, and the system wakes up to a person who
        #     is already on the floor — with no history of them ever standing, which
        #     is the one fact the whole rule layer is built on.
        #  2. The app LOOPS a video file. On rewind the subject teleports from lying
        #     over there to standing over here; IoU collapses and ByteTrack — quite
        #     correctly — calls them two different people. Observed: a permanent
        #     track #1 (standing, ar 0.44) and a permanent track #2 (lying, ar 2.41,
        #     "prone" for 24 s), and NEITHER ever witnessed the standing→falling
        #     transition. A long lead lets the fall be seen inside one loop.
        n_lead = int(args.lead * fps)
        for _ in range(n_lead):
            vw.write(frames[0])
        for f in frames:
            vw.write(f)
        n_hold = int(args.hold * fps)
        for _ in range(n_hold):          # the person stays on the floor, as they would
            vw.write(frames[-1])
        vw.release()
        total = (n_lead + len(frames) + n_hold) / fps
        print(f"  {stem}: ยืนรอ {args.lead:g}s + คลิป {len(frames)/fps:.1f}s + "
              f"นอนต่อ {args.hold:g}s = {total:.1f}s  →  {dst.name}")

    print(f"\nคลิปเดโม → {args.out}")
    print("\nวิธีใช้ในแอป: หน้า Source → ไฟล์วิดีโอ → ชี้ไปที่ไฟล์ข้างบน → Start")
    print("ควรเห็นกรอบแดง + 'ล้ม!' + แจ้งเตือน EMERGENCY ภายใน ~5 วินาที")
    print("\n⚠️  คลิปพวกนี้ใช้ 'สาธิต' เท่านั้น — ห้ามเอาไปอ้างเป็นตัวเลข recall")
    print("    วัดผลจริงด้วย fall_eval.py บนคลิปต้นฉบับเสมอ")


if __name__ == "__main__":
    main()
