#!/usr/bin/env python3
# utils/fall_detector.py — per-track fall detection.
# ================================================================
# Pose sequence → pretrained Transformer (TFLite), plus a resolution-independent
# rule layer. Ported from punpayut/Fall-Detection (MIT):
#   https://github.com/punpayut/Fall-Detection
# The feature contract is copied VERBATIM from their deployment/raspberry_pi/
# fall-detector.py so the shipped weights see exactly what they were trained on:
#   17 keypoints, names SORTED ALPHABETICALLY → 51 features (x, y, conf) laid out
#   at [3i, 3i+1, 3i+2]; per-frame normalisation subtracts the hip midpoint and
#   divides x,y (never conf) by |mid_shoulder_y − mid_hip_y|; a frame with no pose
#   contributes zeros; a deque of 30 consecutive frames is the model input
#   (1, 30, 51) float32 → one fall probability.
#
# Two things upstream never had to solve, and we do:
#  * MULTI-PERSON. A real emergency can put two people on the floor. Every tracked
#    person gets their own deque / confirmation / cooldown, and we never cap, evict
#    or skip a live track.
#  * DISTANCE. Measured on this project's footage, MediaPipe finds landmarks for
#    ~88% of people ≥150px wide but only ~25% of people below that; YOLO-pose gets
#    ~75% of the distant ones. So pose alone silently blinds us to far-away
#    workers, and the rule layer (prone + motionless/sudden-drop) is mandatory,
#    not a nicety.
# ================================================================
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config as cfg

# ── Upstream feature contract (do not "clean up" — the weights depend on it) ──
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]
SORTED_NAMES = sorted(KEYPOINT_NAMES)
KP_INDEX = {n: i for i, n in enumerate(SORTED_NAMES)}
NUM_FEATURES = len(SORTED_NAMES) * 3          # 51
INPUT_TIMESTEPS = 30
MIN_KP_CONF = 0.3                              # for the normalisation reference only

# COCO-17 (ultralytics pose) happens to be exactly KEYPOINT_NAMES, in order.
_COCO_ORDER = KEYPOINT_NAMES


def _cols(name: str) -> tuple[int, int, int]:
    i = KP_INDEX[name]
    return 3 * i, 3 * i + 1, 3 * i + 2


def normalize_skeleton_frame(f: np.ndarray, min_confidence: float = MIN_KP_CONF) -> np.ndarray:
    """Verbatim port of upstream `normalize_skeleton_frame`.

    Hip-centred, torso-scaled → translation/scale invariant, which is also why a
    crop and a full frame produce the same vector, and why YOLO's pixel keypoints
    and MediaPipe's 0-1 keypoints both work."""
    out = np.copy(f)
    lsx, lsy, lsc = _cols("Left Shoulder")
    rsx, rsy, rsc = _cols("Right Shoulder")
    lhx, lhy, lhc = _cols("Left Hip")
    rhx, rhy, rhc = _cols("Right Hip")

    mid_sh_x = mid_sh_y = np.nan
    v_ls, v_rs = f[lsc] > min_confidence, f[rsc] > min_confidence
    if v_ls and v_rs:
        mid_sh_x, mid_sh_y = (f[lsx] + f[rsx]) / 2, (f[lsy] + f[rsy]) / 2
    elif v_ls:
        mid_sh_x, mid_sh_y = f[lsx], f[lsy]
    elif v_rs:
        mid_sh_x, mid_sh_y = f[rsx], f[rsy]

    mid_hip_x = mid_hip_y = np.nan
    v_lh, v_rh = f[lhc] > min_confidence, f[rhc] > min_confidence
    if v_lh and v_rh:
        mid_hip_x, mid_hip_y = (f[lhx] + f[rhx]) / 2, (f[lhy] + f[rhy]) / 2
    elif v_lh:
        mid_hip_x, mid_hip_y = f[lhx], f[lhy]
    elif v_rh:
        mid_hip_x, mid_hip_y = f[rhx], f[rhy]

    if np.isnan(mid_hip_x) or np.isnan(mid_hip_y):
        return f                                    # upstream returns the RAW vector

    ref_h = np.nan
    if not np.isnan(mid_sh_y) and not np.isnan(mid_hip_y):
        ref_h = np.abs(mid_sh_y - mid_hip_y)
    scale = not (np.isnan(ref_h) or ref_h < 1e-5)

    for n in SORTED_NAMES:
        xc, yc, _ = _cols(n)
        out[xc] -= mid_hip_x
        out[yc] -= mid_hip_y
        if scale:
            out[xc] /= ref_h
            out[yc] /= ref_h
    return out


def _has_pose(f: np.ndarray) -> bool:
    """A frame carried real landmarks (upstream signals 'no pose' with all-zeros)."""
    return bool(np.any(f))


def pose_is_normalizable(f: np.ndarray, min_confidence: float = MIN_KP_CONF) -> bool:
    """Can this skeleton actually be hip-centred and torso-scaled?

    Without a hip AND a shoulder, `normalize_skeleton_frame` falls through and
    returns the vector untouched — a different distribution from every frame the
    model was trained on. A close-up of someone's head produces exactly that, and
    it must count as "no pose", not as a usable observation."""
    _, _, lhc = _cols("Left Hip");      _, _, rhc = _cols("Right Hip")
    _, _, lsc = _cols("Left Shoulder"); _, _, rsc = _cols("Right Shoulder")
    hip = f[lhc] > min_confidence or f[rhc] > min_confidence
    sho = f[lsc] > min_confidence or f[rsc] > min_confidence
    return bool(hip and sho)


# ────────────────────────── pose extractors ──────────────────────────
class _MediaPipeExtractor:
    """Faithful-to-upstream backend. MediaPipe Pose is single-person, so we run it
    on each tracked person's crop. Instances are NOT thread-safe → one per worker
    thread; sized so we still fit the loop budget with several people on screen
    (measured: 4 workers → 8 people in ~44 ms)."""

    name = "mediapipe"

    def __init__(self, workers: int = 4, complexity: int = 1):
        import mediapipe as mp
        from concurrent.futures import ThreadPoolExecutor
        self._mp = mp
        self._complexity = complexity
        self._local = threading.local()
        self._workers = max(1, workers)
        # ONE long-lived pool. A fresh executor per call would spawn fresh threads,
        # and the thread-local below would then build a brand-new mediapipe graph
        # every frame and never close it — that leaks native resources until the
        # process aborts (observed).
        self._ex = ThreadPoolExecutor(max_workers=self._workers,
                                      thread_name_prefix="fall-pose")
        self._lm_index = {
            n: getattr(mp.solutions.pose.PoseLandmark, n.upper().replace(" ", "_")).value
            for n in KEYPOINT_NAMES
        }

    def _pose(self):
        p = getattr(self._local, "pose", None)
        if p is None:
            p = self._mp.solutions.pose.Pose(
                static_image_mode=True, model_complexity=self._complexity,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self._local.pose = p
        return p

    def set_complexity(self, c: int):
        """Degrade under load (18.3ms → 13.7ms/person) rather than drop a person."""
        if c != self._complexity:
            self._complexity = c
            self._rebuild()

    def _rebuild(self):
        """Tear the pool down cleanly so the old graphs are released, then let the
        new worker threads build fresh Pose instances on first use."""
        from concurrent.futures import ThreadPoolExecutor
        self.close()
        self._local = threading.local()
        self._ex = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="fall-pose")

    def close(self):
        ex, self._ex = getattr(self, "_ex", None), None
        if ex is not None:
            ex.shutdown(wait=True)

    def _one(self, crop_bgr) -> np.ndarray:
        f = np.zeros(NUM_FEATURES, dtype=np.float32)
        if crop_bgr is None or crop_bgr.size == 0:
            return f
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose().process(rgb)
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            for n in KEYPOINT_NAMES:
                lm = lms[self._lm_index[n]]
                xc, yc, cc = _cols(n)
                f[xc], f[yc], f[cc] = lm.x, lm.y, lm.visibility
        return f

    def extract(self, frame_bgr, persons: list[dict]) -> dict[int, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        crops, tids = [], []
        for p in persons:
            x1, y1 = max(0, int(p["x1"])), max(0, int(p["y1"]))
            x2, y2 = min(w, int(p["x2"])), min(h, int(p["y2"]))
            pw, ph = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)   # small context pad
            crops.append(frame_bgr[max(0, y1 - ph):min(h, y2 + ph),
                                   max(0, x1 - pw):min(w, x2 + pw)])
            tids.append(p["track_id"])
        if not crops:
            return {}
        if len(crops) == 1:                       # avoid pool hand-off for the common case
            return {tids[0]: self._one(crops[0])}
        return dict(zip(tids, self._ex.map(self._one, crops)))


class _YoloPoseExtractor:
    """One forward pass yields keypoints for EVERY person — multi-person for free,
    and measurably more robust on distant workers (75% vs MediaPipe's 25%).
    Ultralytics' COCO-17 order is identical to upstream's 17 names."""

    name = "yolo"

    def __init__(self, model_path: str, device: Optional[str] = None, imgsz: int = 960):
        from ultralytics import YOLO
        from utils.detect_track import _device
        p = Path(model_path)
        self.model_path = str(p) if p.exists() else (p.name or "yolo11n-pose.pt")
        self.device = device or _device()
        self.model = YOLO(self.model_path)
        self.imgsz = imgsz

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def extract(self, frame_bgr, persons: list[dict]) -> dict[int, np.ndarray]:
        out = {p["track_id"]: np.zeros(NUM_FEATURES, dtype=np.float32) for p in persons}
        if not persons:
            return out
        H, W = frame_bgr.shape[:2]
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=0.25,
                               device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return out
        # Ultralytics returns PIXELS; MediaPipe returns 0-1. normalize_skeleton_frame
        # has fallback branches (no valid hip, or shoulder_y == hip_y) that return the
        # vector UNSCALED — for pixels that is hundreds, far outside the ~[-3,3] the
        # Transformer was trained on. Convert here so both backends are identical in
        # every branch, not just the happy path.
        kxy = r.keypoints.xy.cpu().numpy().astype(np.float32)   # (N,17,2)
        kxy[..., 0] /= max(1, W)
        kxy[..., 1] /= max(1, H)
        kconf = (r.keypoints.conf.cpu().numpy()
                 if r.keypoints.conf is not None else np.ones(kxy.shape[:2], np.float32))
        pboxes = r.boxes.xyxy.cpu().numpy()
        used = set()
        for p in persons:                                   # assign each track its own skeleton
            box = (p["x1"], p["y1"], p["x2"], p["y2"])
            best, best_iou = -1, 0.3
            for j in range(len(pboxes)):
                if j in used:
                    continue
                s = self._iou(box, pboxes[j])
                if s > best_iou:
                    best, best_iou = j, s
            if best < 0:
                continue
            used.add(best)
            f = np.zeros(NUM_FEATURES, dtype=np.float32)
            for i, n in enumerate(_COCO_ORDER):
                xc, yc, cc = _cols(n)
                f[xc], f[yc], f[cc] = kxy[best, i, 0], kxy[best, i, 1], kconf[best, i]
            out[p["track_id"]] = f
        return out


# ────────────────────────── per-track state ──────────────────────────
class _TrackState:
    __slots__ = ("seq", "pose_hits", "hist", "prone_since", "transition_ok", "p_fall",
                 "last_box")

    def __init__(self):
        self.seq: deque = deque(maxlen=INPUT_TIMESTEPS)
        self.pose_hits: deque = deque(maxlen=INPUT_TIMESTEPS)
        # (t, aspect_ratio, cx, cy) — cx/cy normalized by frame WIDTH/HEIGHT.
        # Long enough to hold the upright baseline window.
        self.hist: deque = deque(maxlen=200)
        self.prone_since: Optional[float] = None
        self.transition_ok: bool = False
        self.p_fall: float = 0.0
        self.last_box: Optional[tuple] = None      # for rebinding across an id switch


def body_span(p: dict, W: int, H: int) -> float:
    """How big is this person, as a fraction of the frame diagonal.

    Deliberately uses the box DIAGONAL: it is the same whether the person is
    standing or lying down. Box height is not — a fallen person's box is short by
    definition, which is why gating on height vetoed the very falls we exist to
    catch, and why gating on "height while upright" needs an upright history that a
    mid-fall id switch destroys. Measured: real falls span 0.04–0.51 of the
    diagonal; a face filling the lens spans 0.72."""
    w = float(p["x2"] - p["x1"])
    h = float(p["y2"] - p["y1"])
    diag = float(np.hypot(W, H)) or 1.0
    return float(np.hypot(w, h)) / diag


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ar_now(st: "_TrackState") -> float:
    """This person's current width/height. History-free by design."""
    return st.hist[-1][1] if st.hist else 0.0


def ar_grew(st: "_TrackState", baseline: float) -> bool:
    """Has this person's box widened meaningfully versus their own upright shape?
    Corroboration for the model — a fall changes the silhouette; sitting at a desk
    with the camera in your face does not (there is no upright baseline at all)."""
    if not st.hist or not baseline:
        return False
    ar = st.hist[-1][1]
    return ar >= float(getattr(cfg, "FALL_AR_CORROBORATE", 1.25)) * baseline


class FallResult:
    __slots__ = ("p_fall", "prone", "prone_for", "pose_coverage", "drop", "body_ok",
                 "baseline", "transition", "fallen")

    def __init__(self, p_fall, prone, prone_for, pose_coverage, drop, body_ok,
                 baseline, transition, fallen):
        self.p_fall, self.prone, self.prone_for = p_fall, prone, prone_for
        self.pose_coverage, self.drop, self.body_ok = pose_coverage, drop, body_ok
        self.baseline, self.transition, self.fallen = baseline, transition, fallen


class FallDetector:
    """Per-track fall detection. `step()` is called from the pipeline's fixed-cadence
    fall loop so the 30-frame window always spans the duration the model was trained
    on — the PPE detect loop skips frames and would stretch a fall to ~2.3 s."""

    def __init__(self, backend: Optional[str] = None, tflite_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.threshold = float(getattr(cfg, "FALL_PROB_THRESHOLD", 0.90))
        self.min_coverage = float(getattr(cfg, "FALL_MIN_POSE_COVERAGE", 0.6))
        self.mode = getattr(cfg, "FALL_MODE", "hybrid")
        self._states: dict[int, _TrackState] = {}
        self._lost: dict[int, _TrackState] = {}    # recently-vanished tracks, for rebinding

        backend = (backend or getattr(cfg, "FALL_POSE_BACKEND", "yolo")).lower()
        if backend == "mediapipe":
            self.extractor = _MediaPipeExtractor(
                workers=int(getattr(cfg, "FALL_POSE_WORKERS", 4)),
                complexity=int(getattr(cfg, "MEDIAPIPE_MODEL_COMPLEXITY", 1)))
        else:
            self.extractor = _YoloPoseExtractor(
                getattr(cfg, "FALL_POSE_MODEL", "yolo11n-pose.pt"), device=device)

        path = tflite_path or getattr(cfg, "FALL_TFLITE_PATH", "")
        from ai_edge_litert.interpreter import Interpreter
        self._it = Interpreter(model_path=str(path))
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]
        self._out = self._it.get_output_details()[0]
        got = tuple(int(v) for v in self._in["shape"])
        if got != (1, INPUT_TIMESTEPS, NUM_FEATURES):
            raise RuntimeError(f"fall model expects {got}, code assumes "
                               f"(1,{INPUT_TIMESTEPS},{NUM_FEATURES})")
        self._lock = threading.Lock()

    def close(self):
        """Release the pose backend's native graphs (mediapipe leaks without this)."""
        c = getattr(self.extractor, "close", None)
        if callable(c):
            c()

    # ── ID churn is worst exactly when it hurts most: during the fall ─────────
    def _state_for(self, tid: int, p: dict, now: float) -> _TrackState:
        """A falling body flips its box from tall to wide in a few frames; the IoU
        between consecutive boxes collapses and ByteTrack happily issues a fresh id
        (the live camera churned #515→#933 in minutes). A fresh id used to mean a
        fresh, empty history — losing the upright baseline, the pose sequence, and
        with them the transition that *is* the fall. So when an id we have never
        seen appears where an id we just lost disappeared, it is the same person:
        adopt the state instead of starting from nothing."""
        st = self._states.get(tid)
        if st is not None:
            return st
        box = (p["x1"], p["y1"], p["x2"], p["y2"])
        ttl = float(getattr(cfg, "FALL_REBIND_SEC", 3.0))
        for old in [k for k, s in self._lost.items()
                    if s.last_box is None or now - s.last_box[4] > ttl]:
            self._lost.pop(old, None)
        best, best_iou = None, float(getattr(cfg, "FALL_REBIND_IOU", 0.2))
        for old_tid, s in self._lost.items():
            score = iou(box, s.last_box[:4])
            if score > best_iou:
                best, best_iou = old_tid, score
        st = self._lost.pop(best) if best is not None else _TrackState()
        self._states[tid] = st
        return st

    # ── rule layer: a fall is a TRANSITION, not a posture ──────────────────────
    def _posture(self, st: _TrackState, p: dict, W: int, H: int, now: float):
        """Returns (body_ok, prone, prone_for, drop, baseline, transition).

        Two dead ends are baked into this comment so they are not walked again.
        v1 tested a posture in isolation — `w/h > 0.72 and still` — so a seated
        person under a close-up lens alarmed forever (28 false emergencies).
        v2 over-corrected: EVERY gate hung off "this person's own upright
        baseline", which a fall destroys. The instant ByteTrack reassigns the id
        mid-fall (it does: the box flips from tall to wide, IoU collapses), the new
        track has never been seen upright, so `box_ok` was False for the rest of its
        life and the alarm could not fire at any probability. Measured: a track that
        starts already-prone reaches p_fall 0.988 and stays silent.

        So body plausibility is now judged by the box DIAGONAL, which does not care
        which way up the person is and needs no history. The upright baseline is
        still computed — but only the *rule* layer, which detects a transition and
        therefore genuinely needs a "before", depends on it."""
        w_px = max(1.0, p["x2"] - p["x1"])
        h_px = max(1.0, p["y2"] - p["y1"])
        ar = w_px / h_px
        cx = (p["x1"] + p["x2"]) / 2.0 / max(1, W)       # was divided by H (bug)
        cy = (p["y1"] + p["y2"]) / 2.0 / max(1, H)

        st.hist.append((now, ar, cx, cy))
        base_sec = float(getattr(cfg, "FALL_BASELINE_SEC", 10.0))
        while st.hist and now - st.hist[0][0] > base_sec:
            st.hist.popleft()

        upright_ar = float(getattr(cfg, "FALL_UPRIGHT_AR", 0.8))
        upright = [e for e in st.hist if e[1] <= upright_ar]
        baseline = float(np.median([e[1] for e in upright])) if upright else None

        # Is this a plausible human body, right now, in this frame? Orientation-free.
        span = body_span(p, W, H)
        body_ok = (float(getattr(cfg, "FALL_MIN_BODY_SPAN", 0.05)) <= span
                   <= float(getattr(cfg, "FALL_MAX_BODY_SPAN", 0.60)))

        # "Prone" = much wider than THIS person normally is, with an absolute floor.
        spike = float(getattr(cfg, "FALL_AR_SPIKE", 1.8))
        abs_min = float(getattr(cfg, "FALL_AR_ABS_MIN", 0.9))
        thr = max(abs_min, spike * baseline) if baseline else abs_min
        prone = ar > thr

        if prone:
            if st.prone_since is None:
                st.prone_since = now
                # Look back over the transition window: were they upright, and did
                # the box centre actually drop? Both must hold at the moment they
                # first went prone — that is the fall itself.
                tw = float(getattr(cfg, "FALL_TRANSITION_SEC", 1.5))
                past = [e for e in st.hist if now - e[0] <= tw]
                was_upright = any(e[1] <= upright_ar for e in past)
                cy0 = min((e[3] for e in past), default=cy)
                st.transition_ok = bool(
                    baseline is not None and was_upright and
                    (cy - cy0) >= float(getattr(cfg, "FALL_DROP_MIN", 0.02)))
        else:
            st.prone_since = None
            st.transition_ok = False

        prone_for = (now - st.prone_since) if st.prone_since else 0.0
        drop = 0.0
        if len(st.hist) >= 2:
            drop = cy - min(e[3] for e in st.hist)

        # A fall the rule layer will vouch for: plausible body, seen upright, went
        # prone with a real drop, and STAYED down.
        transition = bool(body_ok and prone and st.transition_ok and
                          prone_for >= float(getattr(cfg, "FALL_MOTIONLESS_SEC", 2.0)))
        return body_ok, prone, prone_for, drop, baseline, transition

    def _predict(self, st: _TrackState) -> float:
        if len(st.seq) < INPUT_TIMESTEPS:
            return 0.0
        x = np.expand_dims(np.array(st.seq, dtype=np.float32), 0)
        with self._lock:                                # one interpreter, many tracks
            self._it.set_tensor(self._in["index"], x)
            self._it.invoke()
            return float(self._it.get_tensor(self._out["index"])[0][0])

    def step(self, frame_bgr, persons: list[dict], now: Optional[float] = None) -> dict[int, FallResult]:
        """persons: dicts with track_id + x1,y1,x2,y2. Every live track is processed —
        no cap, no eviction. Returns per-track state; the engine decides on events."""
        now = time.time() if now is None else now
        persons = [p for p in persons if p.get("track_id") is not None]
        H, W = frame_bgr.shape[:2]
        feats = self.extractor.extract(frame_bgr, persons) if persons else {}

        alive, results = set(), {}
        for p in persons:
            tid = p["track_id"]
            alive.add(tid)
            st = self._state_for(tid, p, now)

            f = feats.get(tid, np.zeros(NUM_FEATURES, np.float32))
            # A skeleton without a hip+shoulder cannot be normalised, so it is NOT
            # an observation — it is a gap, exactly like "no pose" upstream.
            ok = _has_pose(f) and pose_is_normalizable(f)
            st.seq.append(normalize_skeleton_frame(f) if ok else np.zeros(NUM_FEATURES, np.float32))
            st.pose_hits.append(1.0 if ok else 0.0)

            coverage = float(np.mean(st.pose_hits)) if st.pose_hits else 0.0
            # A mostly-zero window makes p_fall meaningless — judge those tracks by rules.
            st.p_fall = self._predict(st) if coverage >= self.min_coverage else 0.0

            body_ok, prone, prone_for, drop, baseline, transition = self._posture(st, p, W, H, now)
            st.last_box = (p["x1"], p["y1"], p["x2"], p["y2"], now)

            pose_hit = st.p_fall >= self.threshold
            # The model is the evidence; posture only corroborates. Every term here
            # is answerable from the CURRENT frame, so a person who is already on the
            # floor when their track begins is still detectable — that is the whole
            # point. `ar_grew` remains as the fallback for someone lying towards the
            # lens, whose box is foreshortened and never gets wide.
            # The close-up view that produced the old false alarms is rejected twice
            # over: its span is 0.72 (> max) and its coverage is 0.00, so `pose_hit`
            # is False regardless.
            lying_now = ar_now(st) > float(getattr(cfg, "FALL_AR_ABS_MIN", 0.9))
            corroborated = body_ok and (lying_now or ar_grew(st, baseline))
            pose_fall = pose_hit and corroborated
            # Rule-only is the safety net for people the pose model genuinely cannot
            # read (too far / occluded). It may never fire on its own otherwise.
            rule_fall = (coverage < self.min_coverage) and transition

            if self.mode == "pose":
                fallen = pose_fall
            elif self.mode == "yolo":
                fallen = transition
            else:                                        # hybrid (default)
                fallen = pose_fall or rule_fall

            results[tid] = FallResult(st.p_fall, prone, prone_for, coverage, drop,
                                      body_ok, baseline, transition, fallen)

        # A track that left is not forgotten immediately — it is parked for
        # FALL_REBIND_SEC so the same person returning under a new id keeps their
        # history. Anything older is dropped by `_state_for`.
        for tid in list(self._states):
            if tid not in alive:
                st = self._states.pop(tid)
                if st.last_box is not None:
                    self._lost[tid] = st
        ttl = float(getattr(cfg, "FALL_REBIND_SEC", 3.0))
        for tid, st in list(self._lost.items()):         # bound the parked set
            if now - st.last_box[4] > ttl:
                self._lost.pop(tid, None)
        return results
