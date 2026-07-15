#!/usr/bin/env python3
"""Stage 0 — measure the fall detector we ALREADY have, before replacing it.

Without this number, "the new model is better" is a feeling, not a fact.

What it measures
----------------
The DEPLOYED system, not a component. Clips are replayed through the real
`PPEEngine` with roles={"fall"} — the same person detector, the same tracker, the
same 30-frame window, the same N-of-M confirm, the same cooldown and the same
spatial de-duplication that the live app uses. What comes out is what the factory
would actually see: alerts, not probabilities.

Two things this has to get right, or the numbers are fiction
------------------------------------------------------------
1. VIDEO TIME, NOT WALL TIME. The fall logic is full of real-time gates: "prone
   for >= FALL_MOTIONLESS_SEC", CooldownGate, the FALL_DEDUPE_SEC window. This
   box runs person+pose at ~5.8 fps but FALL_LOOP_FPS is 10, so a real-time
   replay would fall behind and those gates would fire against the wrong clock —
   a 2-second "motionless" test would silently become 4 seconds of video. So we
   install a virtual clock (time.time is patched for the duration of the replay)
   that advances with the VIDEO, and every timer in the engine follows it.

2. THE DEPLOY CADENCE. Frames are resampled to FALL_LOOP_FPS, exactly like
   pipeline._fall_loop feeds the detector. Evaluating at the clip's native 30 fps
   would measure a system nobody runs.

Ground truth
------------
    labels.json  {"clips": {"<file>": {"falls": [[start_sec, end_sec], ...]}}}
A clip with "falls": [] is a pure negative (walking, sitting on the floor, …) —
those are the clips that decide whether this thing is deployable at all.

Generate a template to fill in:
    python scripts/fall_eval.py --clips data/fall_clips --init-labels

Run:
    python scripts/fall_eval.py --clips data/fall_clips
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time as _time_mod
from pathlib import Path

import cv2
import numpy as np

# This script reports in Thai. A Windows console is cp1252 and raises
# UnicodeEncodeError on the first word — the same trap app.py forces UTF-8 to avoid.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backend"))

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


# ── Virtual clock ────────────────────────────────────────────────────────────
class _Clock:
    """Video time, shared by every gate in the engine.

    time.time is monkey-patched to read this for the length of the replay. That
    is heavy-handed, but it is the only way to make CooldownGate, the motionless
    timer and the dedupe window (which all call time.time() with no injection
    point) agree with the clip instead of with the wall.

    IT MUST START AT A REALISTIC EPOCH, NOT AT ZERO.
    ------------------------------------------------
    CooldownGate.ready() is:

        last = self._last.get(key, 0.0)     # never fired → 0.0
        if now - last >= self.seconds:      # FALL_COOLDOWN_SECONDS = 15

    In production `now` is time.time() ≈ 1.7e9, so `now - 0.0` is astronomically
    larger than 15 and a first-ever alert always passes. With a clock that starts
    at 0.0, `now - 0.0` is merely the elapsed VIDEO time — and these clips are 3-8
    seconds long. `4.0 >= 15` is False, for every alert, in every clip.

    So the harness silently made it IMPOSSIBLE for any clip shorter than the
    cooldown to raise an alert, and scored recall 0.000 no matter how well the
    detector actually worked. Measured before this fix: 7 of 30 clips held
    `fallen=True` for 30+ CONSECUTIVE ticks — sailing through the 3-of-5 confirm —
    and still produced zero alerts.

    Every "recall 0.000" reported before 2026-07-14 evening is an artifact of this,
    not a property of the detector.
    """

    EPOCH = 1_700_000_000.0        # a plausible time.time(), as production sees it

    def __init__(self):
        self.t = self.EPOCH

    def __call__(self) -> float:
        return self.t


@contextlib.contextmanager
def virtual_time(clock: _Clock):
    real = _time_mod.time
    _time_mod.time = clock
    try:
        yield
    finally:
        _time_mod.time = real


# ── Engine ───────────────────────────────────────────────────────────────────
def build_engine():
    from utils.ppe_engine import PPEEngine
    eng = PPEEngine(zones_path=None, camera_id=None, roles={"fall"})
    if not eng.fall_ready:
        raise RuntimeError(
            "fall detector did not load — check FALL_TFLITE_PATH and that "
            "yolo11n-pose.pt is in backend/models/")
    return eng


def reset_engine(eng) -> None:
    """Every clip is an independent trial. Track ids, the 30-frame windows, the
    confirm history, the cooldowns and the fired-incident list must all start
    empty, or clip N inherits clip N-1's state and the scores are meaningless."""
    eng.reset()                 # person tracker + zones
    eng.refresh_tunables()      # rebuilds fconf / fcool from config
    eng._fall._states.clear()
    eng._fall._lost.clear()
    eng._fallen.clear()
    eng._fall_incidents.clear()


def spy_on_fall(eng) -> dict:
    """Capture the per-track FallResult that fall_step() consumes internally.

    We want to know WHY a fall was missed — p_fall was 0.85 against a 0.90
    threshold is a different bug from "pose never found the person" — but
    fall_step() only returns events. Wrapping the inner step() reads the
    diagnosis without running pose a second time.

    The caller MUST clear `box["last"]` before every tick (see `replay`).
    fall_step() returns early — without calling step() at all — whenever the frame
    holds no tracked person, so a spy that is only ever written stays STALE: the
    trace then reports the previous tick's p_fall / torso / hip_v as if they were
    current, and at a clip boundary it reports the previous CLIP's. Observed: the
    first tick of adl-02 faithfully echoed adl-01's final p_fall of 0.644.
    """
    box: dict = {"last": {}}
    orig = eng._fall.step

    def spy(frame, persons, now=None):
        res = orig(frame, persons, now=now)
        box["last"] = res
        return res

    eng._fall.step = spy
    return box


# ── Replay ───────────────────────────────────────────────────────────────────
def replay(eng, spy: dict, clip: Path, clock: _Clock, target_fps: float,
           frame_dump: Path | None, hold_last: float = 0.0) -> dict:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (0 < src_fps <= 240):
        src_fps = 25.0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_src / src_fps if n_src else 0.0

    reset_engine(eng)
    step = src_fps / target_fps          # source frames per sampled frame

    detections: list[dict] = []          # alerts the SYSTEM raised
    trace: list[dict] = []               # per-tick diagnostics
    state = {"k": 0}

    def tick(frame, held: bool) -> None:
        k = state["k"]
        # The ENGINE sees an epoch-like clock (so CooldownGate behaves as it does in
        # production); everything we REPORT is video time, measured from 0.
        vt = k / target_fps
        clock.t = _Clock.EPOCH + vt

        # fall_step() returns early, without calling step(), whenever no tracked
        # person is in the frame — so a spy left over from the previous tick would
        # be read as if it were current. Clear it: an empty spy is the truthful
        # answer ("nothing was evaluated"), a stale one is a fabricated observation.
        spy["last"] = {}
        recs, _ = eng.detect(frame)
        events = eng.fall_step(frame, recs)

        res = spy.get("last") or {}
        if res:
            best = max(res.values(), key=lambda r: r.p_fall)
            trace.append({
                "t": round(vt, 2), "persons": len(recs), "held": held,
                "p_fall": round(best.p_fall, 3),
                "coverage": round(best.pose_coverage, 2),
                "prone": bool(best.prone), "prone_for": round(best.prone_for, 2),
                "body_ok": bool(best.body_ok), "transition": bool(best.transition),
                # null (not 0.0) when pose could not resolve a torso — 0° means
                # "resolved, and perfectly upright", which is the opposite finding.
                "torso_deg": (None if best.torso_deg is None
                              else round(best.torso_deg, 1)),
                "hip_v": round(best.hip_v, 3),
                # THE decisive observable. `fallen` is what fall_step() feeds the
                # confirm window. If it is never True, the block is upstream
                # (corroboration / the rule terms). If it IS True and no alert came
                # out, the block is the confirm window or the cooldown. Without this
                # the diagnosis cannot tell those two apart — and guessing between
                # them has already cost two wrong fixes.
                "fallen": bool(best.fallen),
                # The three terms `transition` is an AND of (with prone_for, above).
                # Recorded separately because the AND alone cannot say WHICH one
                # rejected the fall, and the two candidates — "never seen upright"
                # and "did not stay down long enough" — demand opposite fixes.
                "was_upright": bool(best.was_upright),
                "went_down_fast": bool(best.went_down_fast),
                "still_down": bool(best.still_down),
                # The LATCH itself. was_upright/went_down_fast say "was it ever
                # true"; the latch says "were they true TOGETHER, while prone".
                "transition_ok": bool(best.transition_ok),
                "tracks": len(res),
            })
        else:
            trace.append({"t": round(vt, 2), "persons": len(recs), "held": held,
                          "p_fall": 0.0, "coverage": 0.0, "prone": False,
                          "prone_for": 0.0, "body_ok": False, "transition": False,
                          "torso_deg": None, "hip_v": 0.0,
                          "fallen": False, "was_upright": False,
                          "went_down_fast": False, "still_down": False,
                          "transition_ok": False, "tracks": 0})

        for ev in events:
            detections.append({"t": round(vt, 2), "track_id": ev.get("track_id")})
            if frame_dump is not None:
                frame_dump.mkdir(parents=True, exist_ok=True)
                out = eng.draw_on(frame.copy(), recs)
                cv2.imwrite(str(frame_dump / f"{clip.stem}_alert_{vt:07.2f}s.jpg"), out)
        state["k"] = k + 1

    last = None
    while True:
        idx = int(round(state["k"] * step))
        if n_src and idx >= n_src:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
        tick(frame, held=False)
    cap.release()

    # ── Hold the final frame ─────────────────────────────────────────────────
    # The system only alarms once a person has STAYED down for FALL_MOTIONLESS_SEC,
    # and URFD clips are trimmed to end about a second after impact. fall-01 goes
    # prone at 4.3 s and the video stops at 5.3 s, so the 2 s timer can never
    # finish: the clip scores a miss for a reason that has nothing to do with the
    # detector. That is a measurement artifact, and holding the last frame removes
    # it — the subject is still on the ground in that frame (URFD labels it 1), so
    # this asserts nothing that the annotation does not already say.
    #
    # It is still a synthetic extension of real footage, so it is opt-in, it is
    # marked `held` in the trace, and the report says how much of each clip it was.
    n_hold = int(round(hold_last * target_fps))
    for _ in range(n_hold):
        if last is None:
            break
        tick(last, held=True)

    k = state["k"]
    return {"detections": detections, "trace": trace,
            "duration": (duration or (k / target_fps)) + (n_hold / target_fps),
            "src_fps": src_fps, "ticks": k, "held_sec": n_hold / target_fps}


# ── Scoring ──────────────────────────────────────────────────────────────────
def score_clip(falls: list[list[float]], det: list[dict], max_latency: float) -> dict:
    """A fall is CAUGHT if an alert lands between its start and start+max_latency.

    Latency is measured from the START of the fall, not the end: an alert that
    arrives 30 seconds after someone hits the floor is not a detection, it is an
    obituary. Alerts outside every fall's window are false positives.

    Each alert can only be credited to ONE fall, so two people falling together
    need two alerts — the engine emits one per track, and this must not let a
    single alert cover both.
    """
    caught: dict[int, float] = {}          # fall index -> latency
    used: set[int] = set()
    for i, (fs, _fe) in enumerate(falls):
        for j, d in enumerate(det):
            if j not in used and fs <= d["t"] <= fs + max_latency:
                used.add(j)
                caught[i] = round(d["t"] - fs, 2)
                break
    false_alerts = [d for j, d in enumerate(det) if j not in used]
    return {"tp": len(caught), "fn": len(falls) - len(caught),
            "fp": len(false_alerts), "caught": caught,
            "latencies": list(caught.values()), "false_alerts": false_alerts}


def diagnose_miss(trace: list[dict], fs: float, fe: float, cfg) -> str:
    """Say WHY a fall was missed, from the recorded trace — a bare FN count tells
    you nothing about what to fix."""
    win = [t for t in trace if fs - 1.0 <= t["t"] <= fe + float(
        getattr(cfg, "FALL_MOTIONLESS_SEC", 2.0)) + 3.0]
    if not win:
        return "ไม่มีเฟรมในช่วงนี้เลย"
    if not any(t["persons"] for t in win):
        return "person detector ไม่เจอคนเลย (ปัญหาอยู่ที่ Stage 1 ไม่ใช่ fall)"
    peak = max(t["p_fall"] for t in win)
    cov = max(t["coverage"] for t in win)
    thr = float(getattr(cfg, "FALL_PROB_THRESHOLD", 0.90))

    # The rule layer is now a first-class path, not just a fallback, so a miss has to
    # say which of its four terms failed — otherwise "tune the thresholds" is a guess.
    angles = [t["torso_deg"] for t in win if t.get("torso_deg") is not None]
    peak_v = max((t.get("hip_v") or 0.0) for t in win)
    prone_deg = float(getattr(cfg, "FALL_PRONE_ANGLE", 60.0))
    need_v = float(getattr(cfg, "FALL_HIP_VELOCITY", 0.35))

    if not any(t["body_ok"] for t in win):
        return "body span ไม่ผ่าน (ใกล้/ไกลเกิน) → ถูกปฏิเสธก่อนถึงโมเดล"
    if peak >= thr:
        return f"โมเดลยิงแล้ว (p={peak:.3f}) แต่ถูก corroboration/confirm/cooldown บล็อก"

    if not angles:
        if cov < float(getattr(cfg, "FALL_MIN_POSE_COVERAGE", 0.6)):
            if any(t["transition"] for t in win):
                return (f"pose อ่านไม่ได้ (coverage {cov:.2f}) rule layer เข้าเงื่อนไขแล้ว "
                        f"แต่ไม่ยิง — ตรวจ confirm/cooldown")
            return (f"pose อ่านคนไม่ได้เลย (coverage {cov:.2f}, ไม่มี torso สักเฟรม) "
                    f"→ เหลือแต่ rule แบบ bbox ซึ่งไม่เข้าเงื่อนไข transition")
        return f"ไม่มี torso ที่อ่านได้ ทั้งที่ coverage {cov:.2f} — ผิดปกติ ตรวจ anatomy()"

    max_deg = max(angles)

    # `prone` is (wide box) OR (torso past FALL_PRONE_ANGLE) — either witness will
    # do, because pose goes blind once the person is on the floor and only the box
    # survives there. So a torso that never reached the angle is NOT on its own a
    # reason for a miss, and reporting it as one sends you off to lower a threshold
    # that was not blocking anything. Ask what actually blocked: did the system ever
    # consider this person prone at all?
    ever_prone = any(t["prone"] for t in win)
    if not ever_prone:
        return (f"ไม่เคยเข้าสถานะ prone เลย: กล่องไม่เคยกว้างพอ (FALL_AR_ABS_MIN) "
                f"และ torso สูงสุดแค่ {max_deg:.0f}° ≤ {prone_deg:.0f}° → "
                f"ทั้งกล่องและ pose ต่างมองไม่เห็นว่าคนนี้ล้มลง")

    # NOTE: there used to be a `if peak_v < need_v: return "hip too slow"` here, and
    # it lied. The engine accepts EITHER witness — hip velocity OR the box centre's
    # fall rate (`went_down_fast`) — so a slow hip is not on its own a reason for a
    # miss, and blaming it sent us off to lower FALL_HIP_VELOCITY on clips the box
    # rate had already vouched for. The term-by-term block below answers this
    # correctly, from what the engine actually evaluated.
    if not any(t["transition"] for t in win):
        # `transition` = body_ok AND still_down AND (was_upright AND went_down_fast)
        #                AND prone_for >= FALL_MOTIONLESS_SEC.
        # This used to report the failure as "was_upright OR motionless — pick one",
        # which is not a diagnosis, it is a coin toss between two fixes that pull in
        # opposite directions. Each term is now recorded per tick, so say which one.
        need_still = float(getattr(cfg, "FALL_MOTIONLESS_SEC", 2.0))
        max_prone_for = max((t.get("prone_for") or 0.0) for t in win)
        ever_latched = any(t.get("transition_ok") for t in win)
        ever_upright = any(t.get("was_upright") for t in win)
        ever_fast = any(t.get("went_down_fast") for t in win)

        # Ask the LATCH first. An earlier version asked the two ingredients
        # ("was_upright ever?", "went_down_fast ever?") and, seeing both, concluded
        # the only thing left must be the motionless timer — and then printed
        # "prone_for 4.10s < 2s", which is simply false. The ingredients being true
        # at SOME point does not mean they were true TOGETHER while the person was
        # prone, which is what st.transition_ok actually requires.
        if not ever_latched:
            if not ever_upright:
                return ("prone ✓ แต่ latch ไม่ติด: was_upright = False ตลอด → ระบบไม่เคย"
                        "เห็นคนนี้ในท่ายืนภายใน FALL_BASELINE_SEC "
                        f"({getattr(cfg, 'FALL_BASELINE_SEC', 10.0):g}s) — คลิปเริ่มตอน"
                        "กำลังล้มอยู่แล้ว หรือ person detector จับไม่ได้ตอนยังยืน")
            if not ever_fast:
                return (f"prone ✓ was_upright ✓ แต่ latch ไม่ติด: went_down_fast = False "
                        f"→ ทั้ง hip_v ({peak_v:.2f}) และอัตราตกของกล่องต่ำกว่าเกณฑ์ "
                        f"(FALL_HIP_VELOCITY {need_v:.2f} / FALL_BOX_DROP_RATE "
                        f"{getattr(cfg, 'FALL_BOX_DROP_RATE', 0.15):.2f})")
            return (f"was_upright ✓ และ went_down_fast ✓ เคยจริงทั้งคู่ แต่ "
                    f"transition_ok ไม่เคยติด → มันไม่เคยจริง 'พร้อมกัน ณ tick ที่ prone' "
                    f"(prone_for สูงสุด {max_prone_for:.2f}s) — น่าจะเป็น track id "
                    f"เปลี่ยนกลางการล้ม (state ใหม่ = ประวัติหาย) หรือ prone ขาดช่วง"
                    f"เกิน FALL_PRONE_GRACE_SEC จน latch ถูกล้าง")

        # The latch DID close. So what is left is the "stayed down" timer, or body_ok.
        if max_prone_for < need_still:
            return (f"latch ติดแล้ว (was_upright ✓ went_down_fast ✓ hip_v {peak_v:.2f}) "
                    f"แต่ 'นอนค้าง' ไม่ครบ: prone_for สูงสุด {max_prone_for:.2f}s < "
                    f"FALL_MOTIONLESS_SEC {need_still:g}s → คลิปจบก่อนนับครบ "
                    f"(ใช้ --hold-last)")
        return (f"latch ติด + prone_for {max_prone_for:.2f}s ≥ {need_still:g}s ครบแล้ว "
                f"แต่ transition ยังเป็น False → เหลือ body_ok / still_down ที่ไม่ผ่าน "
                f"ณ tick เดียวกัน (ตรวจ FALL_MIN/MAX_BODY_SPAN)")

    return (f"rule layer ครบเงื่อนไขทุกข้อ (torso {max_deg:.0f}°, hip_v {peak_v:.2f}, "
            f"transition=True) แต่ยังไม่เตือน → ติดที่ confirm "
            f"({getattr(cfg, 'FALL_CONFIRM_FRAMES', 3)}-of-"
            f"{getattr(cfg, 'FALL_CONFIRM_WINDOW', 5)}) หรือ cooldown")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, type=Path)
    ap.add_argument("--labels", type=Path, default=None,
                    help="default: <clips>/labels.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <clips>/eval_out")
    ap.add_argument("--max-latency", type=float, default=10.0,
                    help="วินาทีหลังเริ่มล้ม ที่ยังนับว่าตรวจเจอทัน")
    ap.add_argument("--fps", type=float, default=None,
                    help="override FALL_LOOP_FPS (default: ค่าที่ deploy จริง)")
    ap.add_argument("--hold-last", type=float, default=0.0,
                    help="ยืดเฟรมสุดท้ายต่ออีก N วินาที เพื่อให้ตัวนับ 'นอนนิ่ง' เดินจนครบ "
                         "(คลิป URFD ถูกตัดจบทันทีหลังกระแทกพื้น)")
    ap.add_argument("--dump-frames", action="store_true",
                    help="เซฟภาพทุกครั้งที่ระบบเตือน (ดู false positive ด้วยตา)")
    ap.add_argument("--init-labels", action="store_true",
                    help="สร้าง labels.json เปล่าจากคลิปที่มี แล้วออก")
    args = ap.parse_args()

    labels_p = args.labels or (args.clips / "labels.json")
    clips = sorted(p for p in args.clips.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not clips:
        sys.exit(f"ไม่พบไฟล์วิดีโอใน {args.clips}")

    if args.init_labels:
        tmpl = {"_readme": "falls = [[วินาทีที่เริ่มล้ม, วินาทีที่จบ], ...]  ·  [] = คลิป negative",
                "clips": {c.name: {"falls": []} for c in clips}}
        labels_p.write_text(json.dumps(tmpl, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"เขียน template → {labels_p}\nกรอกเวลาที่ล้มลงไป แล้วรันใหม่โดยไม่ต้องใส่ --init-labels")
        return

    if not labels_p.exists():
        sys.exit(f"ไม่พบ {labels_p} — รันด้วย --init-labels ก่อนเพื่อสร้าง template")
    labels = json.loads(labels_p.read_text(encoding="utf-8")).get("clips", {})

    missing = [c.name for c in clips if c.name not in labels]
    if missing:
        # Guessing that an unlabelled clip is a negative is how a fall clip ends
        # up scored as a false positive. Refuse instead.
        sys.exit("คลิปเหล่านี้ยังไม่มีใน labels.json — เพิ่มก่อน:\n  "
                 + "\n  ".join(missing))

    import config as cfg
    target_fps = args.fps or float(getattr(cfg, "FALL_LOOP_FPS", 10))
    out = args.out or (args.clips / "eval_out")
    out.mkdir(parents=True, exist_ok=True)

    print(f"โมเดล fall  : {Path(getattr(cfg, 'FALL_TFLITE_PATH', '?')).name}")
    print(f"pose backend: {getattr(cfg, 'FALL_POSE_BACKEND', 'yolo')}")
    print(f"mode        : {getattr(cfg, 'FALL_MODE', 'hybrid')}  "
          f"· threshold {getattr(cfg, 'FALL_PROB_THRESHOLD', 0.9)}")
    print(f"cadence     : {target_fps:g} fps → หน้าต่าง 30 เฟรม = "
          f"{30/target_fps:.1f} วินาที")
    print(f"คลิป        : {len(clips)}\n")

    eng = build_engine()
    spy = spy_on_fall(eng)
    clock = _Clock()

    rows, tot = [], {"tp": 0, "fn": 0, "fp": 0}
    all_lat: list[float] = []
    neg_seconds = 0.0
    misses: list[str] = []

    with virtual_time(clock):
        for c in clips:
            falls = [list(map(float, f)) for f in labels[c.name].get("falls", [])]
            t0 = _time_mod.perf_counter()
            r = replay(eng, spy, c, clock, target_fps,
                       out / "alerts" if args.dump_frames else None,
                       hold_last=args.hold_last)
            s = score_clip(falls, r["detections"], args.max_latency)

            for k in tot:
                tot[k] += s[k]
            all_lat += s["latencies"]
            if not falls:
                neg_seconds += r["duration"]

            # A bare FN count says nothing about what to fix, so explain every
            # fall the system did NOT alert on, using the recorded trace.
            for i, (fs, fe) in enumerate(falls):
                if i not in s["caught"]:
                    misses.append(f"  {c.name} @ {fs:.1f}s → "
                                  f"{diagnose_miss(r['trace'], fs, fe, cfg)}")

            (out / f"trace_{c.stem}.json").write_text(
                json.dumps(r["trace"], ensure_ascii=False), encoding="utf-8")

            kind = "FALL" if falls else " neg"
            rows.append((c.name, kind, len(falls), s["tp"], s["fn"], s["fp"],
                         r["duration"], _time_mod.perf_counter() - t0))
            print(f"  [{kind}] {c.name:<34} TP {s['tp']}  FN {s['fn']}  "
                  f"FP {s['fp']}   ({r['duration']:.0f}s วิดีโอ)")

    # ── report ──────────────────────────────────────────────────────────────
    n_falls = tot["tp"] + tot["fn"]
    recall = tot["tp"] / n_falls if n_falls else float("nan")
    prec = tot["tp"] / (tot["tp"] + tot["fp"]) if (tot["tp"] + tot["fp"]) else float("nan")
    fp_per_min = (tot["fp"] / (neg_seconds / 60)) if neg_seconds else float("nan")
    lat = np.array(all_lat) if all_lat else np.array([])

    print("\n" + "=" * 62)
    print("BASELINE ของโมเดลที่ใช้อยู่ตอนนี้ (นี่คือเลขที่ต้องเอาชนะ)")
    print("=" * 62)
    print(f"  ล้มทั้งหมด    : {n_falls}   ตรวจเจอ {tot['tp']}   พลาด {tot['fn']}")
    print(f"  recall        : {recall:.3f}   (Gate 8 ต้อง ≥ 0.90)")
    print(f"  precision     : {prec:.3f}   (Gate 8 ต้อง ≥ 0.95)")
    print(f"  false alert   : {tot['fp']} ครั้ง บน negative {neg_seconds/60:.1f} นาที "
          f"= {fp_per_min:.2f} ครั้ง/นาที   (Gate 8 ต้อง = 0)")
    if lat.size:
        print(f"  latency       : mean {lat.mean():.1f}s · median "
              f"{np.median(lat):.1f}s · worst {lat.max():.1f}s   (Gate 8 ต้อง ≤ 4s)")
    else:
        print("  latency       : n/a (ไม่มีการล้มที่ตรวจเจอเลย)")

    if misses:
        print("\nทำไมถึงพลาด (นี่คือส่วนที่บอกว่าต้องแก้อะไร):")
        print("\n".join(misses))

    gates = {
        "recall ≥ 0.90":     bool(n_falls) and recall >= 0.90,
        "precision ≥ 0.95":  bool(tot["tp"] + tot["fp"]) and prec >= 0.95,
        "false alert = 0":   tot["fp"] == 0,
        "latency ≤ 4s":      bool(lat.size) and float(lat.max()) <= 4.0,
    }
    print("\nGate 8:")
    for g, ok in gates.items():
        print(f"  {'✅' if ok else '❌'} {g}")

    (out / "summary.json").write_text(json.dumps({
        "config": {"tflite": str(getattr(cfg, "FALL_TFLITE_PATH", "")),
                   "mode": getattr(cfg, "FALL_MODE", "hybrid"),
                   "threshold": getattr(cfg, "FALL_PROB_THRESHOLD", 0.9),
                   "loop_fps": target_fps,
                   "window_seconds": round(30 / target_fps, 2)},
        "falls": n_falls, "tp": tot["tp"], "fn": tot["fn"], "fp": tot["fp"],
        "recall": None if n_falls == 0 else round(recall, 4),
        "precision": None if (tot["tp"] + tot["fp"]) == 0 else round(prec, 4),
        "negative_minutes": round(neg_seconds / 60, 2),
        "false_alerts_per_min": None if not neg_seconds else round(fp_per_min, 3),
        "latency_sec": {"mean": None if not lat.size else round(float(lat.mean()), 2),
                        "median": None if not lat.size else round(float(np.median(lat)), 2),
                        "max": None if not lat.size else round(float(lat.max()), 2)},
        "gates": gates,
        "misses": misses,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nสรุป → {out / 'summary.json'}")
    if args.dump_frames:
        print(f"ภาพตอนเตือน → {out / 'alerts'}")


if __name__ == "__main__":
    main()
