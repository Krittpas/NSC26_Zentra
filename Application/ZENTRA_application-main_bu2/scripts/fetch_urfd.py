#!/usr/bin/env python3
"""Fetch the UR Fall Detection dataset and turn it into clips + ground truth.

Why this dataset
----------------
Stage 0 cannot start without labelled falls, and filming 30 real falls is not
something anyone should do. URFD is public, free, and — unlike most fall sets —
ships a PER-FRAME label, so the fall interval is read from their annotation
instead of guessed from a stopwatch.

  falls (30 seq)  a person falls. label 0 = going down, 1 = on the ground.
  adl   (40 seq)  activities of daily living: walking, bending, crouching,
                  sitting, AND LYING DOWN DELIBERATELY.

That last one is the point. 1,470 ADL frames carry label 1 — "on the ground" —
without a fall having happened. Those clips are the hard negative that decides
whether this system is deployable at all: same picture as a fall, no fall. Every
alert raised on an ADL clip is a false positive, and Gate 8 allows zero.

  labels.json for adl-* is written as {"falls": []} — a pure negative.

What it does NOT give you
-------------------------
Nobody sits on the floor for five minutes in URFD. Gate 8 asks for exactly that,
and no public dataset has it. Film it yourself: one static shot, five minutes,
nobody falls, nobody gets hurt. Drop it in the same folder, add it to labels.json
with "falls": [], and it is scored alongside these.

Usage
-----
    python scripts/fetch_urfd.py                     # 10 falls + 10 adl (~1.3 GB)
    python scripts/fetch_urfd.py --falls 30 --adl 40 # everything (~4.5 GB)
    python scripts/fetch_urfd.py --cam 1             # ceiling camera instead
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

# A Windows console is cp1252 and dies on the arrows/emoji below, exactly as it
# does on the engine's Thai output — app.py forces UTF-8 for the same reason.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

BASE = "http://fenix.ur.edu.pl/~mkepski/ds/data/"
SRC_FPS = 30.0                     # URFD RGB is recorded at 30 fps
REPO = Path(__file__).resolve().parent.parent


def fetch(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def load_labels(cam: int) -> dict[str, dict[int, int]]:
    """seq -> {frame_no: label}. label: -1 upright · 0 going down · 1 on the ground."""
    out: dict[str, dict[int, int]] = {}
    for kind in ("falls", "adls"):
        raw = fetch(f"{BASE}urfall-cam{cam}-{kind}.csv").decode()
        for row in csv.reader(io.StringIO(raw)):
            if len(row) < 3:
                continue
            try:
                out.setdefault(row[0].strip(), {})[int(row[1])] = int(row[2])
            except ValueError:
                continue                      # header or a malformed line
    return out


def fall_interval(frames: dict[int, int]) -> list[list[float]] | None:
    """The fall, in seconds, from URFD's own per-frame annotation.

    Starts at the first frame that is no longer upright (label 0 — the descent has
    begun) and ends at the last frame on the ground (label 1). Timing the fall from
    the moment they hit the floor instead would make every detector look faster than
    it is: `fall_eval` measures latency from the START of the fall, deliberately.
    """
    going = sorted(f for f, v in frames.items() if v == 0)
    down = sorted(f for f, v in frames.items() if v == 1)
    if not down:
        return None                          # no fall in this sequence
    start = going[0] if going else down[0]
    return [[round((start - 1) / SRC_FPS, 2), round((down[-1] - 1) / SRC_FPS, 2)]]


def build_clip(seq: str, cam: int, dest: Path) -> bool:
    """Download one sequence's RGB zip and assemble its PNGs into an mp4."""
    if dest.exists():
        print(f"  {seq:<8} already there — skipped")
        return True
    url = f"{BASE}{seq}-cam{cam}-rgb.zip"
    try:
        blob = fetch(url)
    except Exception as e:
        print(f"  {seq:<8} download failed: {e}")
        return False

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".png"))
        if not names:
            print(f"  {seq:<8} zip has no PNG frames")
            return False
        first = cv2.imdecode(np.frombuffer(z.read(names[0]), np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            print(f"  {seq:<8} cannot decode frames")
            return False
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"),
                             SRC_FPS, (w, h))
        if not vw.isOpened():
            print(f"  {seq:<8} cannot open VideoWriter (missing codec?)")
            return False
        for n in names:
            img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                vw.write(img)
        vw.release()
    print(f"  {seq:<8} {len(names):4d} frames  {len(names)/SRC_FPS:5.1f}s  {w}x{h}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "backend" / "data" / "fall_clips")
    ap.add_argument("--cam", type=int, default=0, choices=(0, 1),
                    help="0 = side view (CCTV-like) · 1 = ceiling")
    ap.add_argument("--falls", type=int, default=10, help="how many fall sequences (max 30)")
    ap.add_argument("--adl", type=int, default=10, help="how many ADL sequences (max 40)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"labels  : urfall-cam{args.cam}-*.csv")
    labels = load_labels(args.cam)

    seqs = ([f"fall-{i:02d}" for i in range(1, min(args.falls, 30) + 1)]
            + [f"adl-{i:02d}" for i in range(1, min(args.adl, 40) + 1)])
    print(f"clips   : {len(seqs)}  →  {args.out}\n")

    # Merge, never overwrite: any clip the user filmed and labelled themselves must
    # survive a re-run of this script.
    lp = args.out / "labels.json"
    doc = {"_readme": "falls = [[start_sec, end_sec], ...]  ·  [] = negative clip",
           "clips": {}}
    if lp.exists():
        try:
            doc = json.loads(lp.read_text(encoding="utf-8"))
            doc.setdefault("clips", {})
        except Exception:
            print("⚠️  existing labels.json is unreadable — starting a fresh one")

    ok = 0
    for seq in seqs:
        name = f"{seq}-cam{args.cam}.mp4"
        if not build_clip(seq, args.cam, args.out / name):
            continue
        ok += 1
        frames = labels.get(seq, {})
        iv = fall_interval(frames) if seq.startswith("fall") else None
        # An ADL clip is a PURE NEGATIVE even where the person is on the ground:
        # they lay down, they did not fall. Any alert here is a false positive, and
        # that is precisely what these clips are for.
        doc["clips"][name] = {"falls": iv or []}

    lp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    n_fall = sum(1 for v in doc["clips"].values() if v["falls"])
    n_neg = len(doc["clips"]) - n_fall
    print(f"\n{ok}/{len(seqs)} clips ready")
    print(f"ground truth → {lp}")
    print(f"  {n_fall} clip(s) with a labelled fall · {n_neg} negative clip(s)")
    print("\n⚠️  Still missing, and no public dataset has it: someone SITTING ON THE")
    print("    FLOOR for 5 minutes. Gate 8 is decided by that clip. Film it, drop it")
    print("    in this folder, add it to labels.json with \"falls\": [].")
    print("\nnext:  python scripts/fall_eval.py --clips "
          f"{args.out.relative_to(REPO)} --dump-frames")


if __name__ == "__main__":
    sys.exit(main())
