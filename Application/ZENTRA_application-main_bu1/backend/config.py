# config.py — ZENTRA System Configuration
# Zone Environment Network Thermal Risk Analysis
# Windows 11 + NVIDIA GPU | Python 3.11
# อ้างอิง Slide: CEDT Innovation Summit 2026 (อันดับ 1)
# ================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load this project's .env explicitly so it works no matter what the
# current working directory is (the desktop app imports config from a
# different cwd). override=False keeps real environment vars authoritative.
load_dotenv(Path(__file__).parent / ".env", override=False)

# ================================================================
# BASE PATHS
# ================================================================
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
MODELS_DIR    = BASE_DIR / "models"
REPORTS_DIR   = BASE_DIR / "reports"
LOGS_DIR      = BASE_DIR / "logs"
COLLECTED_DIR = DATA_DIR / "collected"

for _d in [
    DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
    COLLECTED_DIR / "ppe_violations",
    COLLECTED_DIR / "zone_intrusions",
    COLLECTED_DIR / "fall_events",
    COLLECTED_DIR / "normal",
]:
    _d.mkdir(parents=True, exist_ok=True)

# ================================================================
# ROBOFLOW / INFERENCE SERVER
# ================================================================
# SECURITY: no hardcoded default — the previous committed key is leaked in git
# history and MUST be rotated. Set ROBOFLOW_API_KEY in .env/env. Empty is fine:
# the deployed pipeline runs the local .pt (ultralytics), not the Roboflow server.
ROBOFLOW_API_KEY     = os.getenv("ROBOFLOW_API_KEY",  "")
ROBOFLOW_WORKSPACE   = os.getenv("ROBOFLOW_WORKSPACE", "pholawats-workspace")
INFERENCE_SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:9001")

PPE_MODEL_ID  = os.getenv("PPE_MODEL_ID",  "")   # empty = no PPE model (cleared for restructure)
FALL_MODEL_ID = os.getenv("FALL_MODEL_ID", "fall-detection-ovjqo/5")

USE_LOCAL_MODEL  = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
PPE_LOCAL_MODEL  = str(MODELS_DIR / "ppe_finetuned.pt")
FALL_LOCAL_MODEL = str(MODELS_DIR / "fall_finetuned.pt")

# Sentinel used by the UI model selector to mean "use the locally fine-tuned
# .pt (USE_LOCAL_MODEL path)" instead of a Roboflow model served by :9001.
PPE_LOCAL_SENTINEL = "__local__"

# Selectable PPE detection models shown in the Settings dropdown. Each "server"
# entry is a Roboflow model_id pulled + run by the LOCAL inference server
# (INFERENCE_SERVER_URL, :9001) — weights download once then inference runs on
# the local GPU (images never leave the device per-frame). The "local" entry
# uses the user's own fine-tuned ppe_finetuned.pt (fully offline, no Docker).
#
# Registry intentionally EMPTY — models were cleared for a clean restructure.
# To add one back, append e.g.:
#   {"id": "ppe-tqmos/1", "kind": "server", "label": "...", "note": "..."}
#   {"id": PPE_LOCAL_SENTINEL, "kind": "local", "label": "โมเดลที่ฉันเทรนเอง (local .pt)", "note": "..."}
# While empty, PPE detection is paused (the pipeline skips inference gracefully).
PPE_MODELS: list[dict] = [
    {"id": "ppe-vum8g/2", "kind": "server",
     "label": "PPE (Roboflow ppe-vum8g/2)",
     "note": "hardhat/vest/gloves/boots + no_* + person — ไม่มี class แว่นตา"},
    {"id": PPE_LOCAL_SENTINEL, "kind": "local",
     "label": "โมเดลที่ฉันเทรนเอง (local .pt)",
     "note": "ใช้ models/ppe_finetuned.pt — fine-tune จากข้อมูลที่เก็บเอง"},
]

ROBOFLOW_PPE_PROJECT  = os.getenv("ROBOFLOW_PPE_PROJECT",  "zentra-ppe")
ROBOFLOW_FALL_PROJECT = os.getenv("ROBOFLOW_FALL_PROJECT", "zentra-fall")

# ================================================================
# CAMERA  (Windows 11: USE_DSHOW=true เร็วกว่า CAP_FFMPEG)
# ================================================================
CAMERA_SOURCE   = os.getenv("CAMERA_SOURCE", "webcam")
WEBCAM_INDEX    = int(os.getenv("WEBCAM_INDEX", "0"))
RTSP_URL        = os.getenv("RTSP_URL", "")   # set in .env or Settings; no baked-in credentials
VIDEO_FILE_PATH = os.getenv("VIDEO_FILE_PATH", "")
USE_DSHOW       = os.getenv("USE_DSHOW", "true").lower() == "true"

# ================================================================
# INFERENCE
# ================================================================
INFERENCE_CONFIDENCE = float(os.getenv("INFERENCE_CONFIDENCE", "0.45"))
INFERENCE_IOU        = float(os.getenv("INFERENCE_IOU",        "0.45"))
# Decouple TRACKING confidence from PPE confidence. The ultralytics Detector runs
# detect+track at THIS low floor so ByteTrack receives the low-score boxes it
# needs for second-association (→ stable ids through a confidence dip). PPE items
# are then filtered separately by INFERENCE_CONFIDENCE in ppe_association, so the
# low track floor does NOT flood the violation logic with junk PPE boxes.
# MUST be <= track_low_thresh in the tracker yaml or those boxes never arrive.
PPE_TRACK_CONF       = float(os.getenv("PPE_TRACK_CONF", "0.10"))

# ── PERSON DETECTOR (decoupled from the PPE model) ──────────────────────────
# "Who is a person" needs high recall + stable tracking; "what PPE item is this"
# needs precision. Those were conflated into one PPE fine-tune whose `person`
# class is only a byproduct (recall ~0.63) — the real reason crowds went
# undetected. A COCO-pretrained YOLO detects people far better, and it now owns
# the tracker so track IDs follow the STRONG detector. The PPE model no longer
# emits persons and no longer tracks (see utils/detect_track.py).
PERSON_MODEL  = os.getenv("PERSON_MODEL", str(MODELS_DIR / "yolo11s.pt"))
# Model-call floor for the person detector. Low on purpose: ByteTrack needs the
# low-score boxes for second-association. Gating happens in the tracker yaml
# (new_track_thresh). MUST be <= track_low_thresh.
PERSON_CONF   = float(os.getenv("PERSON_CONF", "0.10"))
# Higher than the PPE imgsz: small/distant people in a crowd need the resolution.
PERSON_IMGSZ  = int(os.getenv("PERSON_IMGSZ", "960"))
# Inference device for the PPE model. Empty → detect_track auto-picks
# MPS → CUDA → CPU. Set explicitly (e.g. "cpu") to FORCE a device — without this
# the env var docker-compose sets was silently ignored (config never read it).
PPE_INFER_DEVICE     = os.getenv("PPE_INFER_DEVICE", "") or None
INFER_EVERY_N_FRAMES = int(os.getenv("INFER_EVERY_N_FRAMES",   "2"))   # PPE is ~11ms on GPU → infer more often (less flicker)
# Anti-flicker: keep the last PPE boxes this long to bridge single-frame misses
# (detector occasionally drops a frame → boxes blink). 0 = no hold.
PPE_HOLD_SEC         = float(os.getenv("PPE_HOLD_SEC", "0.5"))
# Display smoothing: EMA on box coordinates so boxes glide instead of jittering
# frame-to-frame ("ดิ้น"). Lower ALPHA = smoother but a touch laggier.
# Clean display: draw ONE box per person + a PPE status label instead of many
# overlapping class boxes (Protex-style, far less cluttered).
PPE_CLEAN_DISPLAY    = os.getenv("PPE_CLEAN_DISPLAY", "true").lower() == "true"
PPE_SMOOTH           = os.getenv("PPE_SMOOTH", "true").lower() == "true"
PPE_SMOOTH_ALPHA     = float(os.getenv("PPE_SMOOTH_ALPHA", "0.25"))
PPE_SMOOTH_IOU       = float(os.getenv("PPE_SMOOTH_IOU", "0.30"))
# Deadband (px): if a box moved less than this, freeze it (rock-steady when a
# person stands still; still follows real movement above the threshold).
PPE_SMOOTH_DEADBAND  = float(os.getenv("PPE_SMOOTH_DEADBAND", "3.0"))
# De-duplicate overlapping boxes of the SAME class (one object → one box).
# Predicting at a low floor returns extra overlapping candidates; suppress any
# same-class box overlapping an already-kept one above this IoU. 0.70 only
# removes near-identical duplicates (keeps two distinct-but-close people).
PPE_NMS_IOU          = float(os.getenv("PPE_NMS_IOU", "0.45"))
# One shared inference client serves both PPE and Fall. Keep the server-side
# confidence floor LOW so neither model starves the other; the real PPE
# threshold (INFERENCE_CONFIDENCE) is applied in code and Fall uses
# FALL_YOLO_CONFIDENCE — so the PPE slider can't throttle fall detection.
INFERENCE_SERVER_FLOOR = float(os.getenv("INFERENCE_SERVER_FLOOR", "0.20"))
# Inference resolution for the LOCAL PPE model. Higher = better small-object
# recall (gloves/glasses on hands) at a little more GPU time. 640 = train size.
PPE_IMGSZ = int(os.getenv("PPE_IMGSZ", "960"))
# Per-CATEGORY confidence floor, applied in utils/ppe_association.associate().
# Keyed by ppe_association CATEGORY: helmet / vest / gloves / glasses / boots.
# A class not listed uses the global INFERENCE_CONFIDENCE slider.
#
# Tuned from a real model.val() sweep (2026-07-08): at the deployed 0.30, glasses
# already gets R=0.70/P=0.73 and gloves R=0.82/P=0.69 — they were NOT starved by
# the filter; the reason they weren't *surfaced* was the per-camera required set
# (fixed via ppe_items). glasses/gloves are set a touch lower (0.25) purely as a
# recall margin for the live camera's domain gap — precision still ≥0.63 on val.
# Going below ~0.20 tanks precision (glasses P→0.48) → false "worn" that mask real
# violations, so DON'T lower further without retraining.
PPE_CLASS_CONF: dict = {"glasses": 0.25, "gloves": 0.25}

# ================================================================
# PPE CLASSES
# Slide Module 1: Helmet / Vest / Goggles / Gloves / Safety Boots
# ================================================================
PPE_CLASSES: dict[str, dict] = {
    "helmet":          {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    "vest":            {"label": "Vest",      "label_th": "สวมเสื้อกั๊ก",      "color": (0, 210, 0),   "violation": False},
    # The fine-tuned local model (zentra-ppe v3) emits "Vest" capitalised — index
    # 0 in the deployed taxonomy. Without this key it fell through to the gray
    # default. _info() also normalises case, but keep it explicit for clarity.
    "Vest":            {"label": "Vest",      "label_th": "สวมเสื้อกั๊ก",      "color": (0, 210, 0),   "violation": False},
    "goggles":         {"label": "Goggles",   "label_th": "สวมแว่นตานิรภัย",   "color": (0, 210, 0),   "violation": False},
    "gloves":          {"label": "Gloves",    "label_th": "สวมถุงมือ",          "color": (0, 210, 0),   "violation": False},
    "safety_boots":    {"label": "Boots",     "label_th": "สวมรองเท้าบูท",     "color": (0, 210, 0),   "violation": False},
    "glasses":         {"label": "Glasses",   "label_th": "สวมแว่นตา",         "color": (0, 210, 0),   "violation": False},
    "boots":           {"label": "Boots",     "label_th": "สวมรองเท้าบูท",     "color": (0, 210, 0),   "violation": False},
    "no_helmet":       {"label": "No Helmet", "label_th": "ไม่สวมหมวก",        "color": (0, 0, 220),   "violation": True},
    "no_vest":         {"label": "No Vest",   "label_th": "ไม่สวมเสื้อกั๊ก",   "color": (0, 0, 220),   "violation": True},
    "no_goggles":      {"label": "No Goggles","label_th": "ไม่สวมแว่นตา",      "color": (0, 0, 220),   "violation": True},
    # zentra-ppe v3 emits "no_glasses" (underscore, index 6). Previously only the
    # spaced "no glasses" and "no_goggles" existed, so eye-protection violations
    # from the local model were SILENTLY dropped (fell through to violation:False).
    "no_glasses":      {"label": "No Glasses","label_th": "ไม่สวมแว่นตา",      "color": (0, 0, 220),   "violation": True},
    "no_gloves":       {"label": "No Gloves", "label_th": "ไม่สวมถุงมือ",      "color": (0, 0, 220),   "violation": True},
    "no_safety_boots": {"label": "No Boots",  "label_th": "ไม่สวมรองเท้าบูท", "color": (0, 0, 220),   "violation": True},
    "no helmet":       {"label": "No Helmet", "label_th": "ไม่สวมหมวก",        "color": (0, 0, 220),   "violation": True},
    "no vest":         {"label": "No Vest",   "label_th": "ไม่สวมเสื้อกั๊ก",   "color": (0, 0, 220),   "violation": True},
    "no glasses":      {"label": "No Glasses","label_th": "ไม่สวมแว่นตา",      "color": (0, 0, 220),   "violation": True},
    "no gloves":       {"label": "No Gloves", "label_th": "ไม่สวมถุงมือ",      "color": (0, 0, 220),   "violation": True},
    "no boots":        {"label": "No Boots",  "label_th": "ไม่สวมรองเท้าบูท", "color": (0, 0, 220),   "violation": True},
    # ppe-tqmos/1 emits these extra class names — map them so violations/worn
    # states register correctly (singular "no glove" + redundant "helmet on").
    "no glove":        {"label": "No Gloves", "label_th": "ไม่สวมถุงมือ",      "color": (0, 0, 220),   "violation": True},
    "helmet on":       {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    # ppe-vum8g/2 emits these class names — "hardhat" == helmet, "no_boots" not
    # covered by the safety_boots mapping above. Map them so worn/violation
    # states register with the same labels as the rest of the system.
    "hardhat":         {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    "no_hardhat":      {"label": "No Helmet", "label_th": "ไม่สวมหมวก",        "color": (0, 0, 220),   "violation": True},
    "no_boots":        {"label": "No Boots",  "label_th": "ไม่สวมรองเท้าบูท", "color": (0, 0, 220),   "violation": True},
    "person":          {"label": "Person",    "label_th": "บุคคล",              "color": (255, 190, 0), "violation": False},
    # ppe-vum8g/2 quirk: the runtime class strings differ from the Roboflow UI
    # labels. WORN items come back prefixed with "a " (a person / a hardhat /
    # a vest / a gloves / a boots) while VIOLATIONS use underscores (no_hardhat
    # …). Map the "a " variants so worn/person boxes draw correctly AND get
    # written into the collected YOLO training labels.
    "a person":        {"label": "Person",    "label_th": "บุคคล",              "color": (255, 190, 0), "violation": False},
    "a hardhat":       {"label": "Helmet",    "label_th": "สวมหมวกนิรภัย",    "color": (0, 210, 0),   "violation": False},
    "a vest":          {"label": "Vest",      "label_th": "สวมเสื้อกั๊ก",      "color": (0, 210, 0),   "violation": False},
    "a gloves":        {"label": "Gloves",    "label_th": "สวมถุงมือ",          "color": (0, 210, 0),   "violation": False},
    "a boots":         {"label": "Boots",     "label_th": "สวมรองเท้าบูท",     "color": (0, 210, 0),   "violation": False},
}
# NOTE: PPE violation detection does NOT compute "person without helmet" — it
# relies on the model emitting explicit negative classes (e.g. "no helmet",
# "no vest") which modules/ppe.py flags via PPE_CLASSES[...]["violation"].
# Both current models (cloud ppe-cpxsz/2, local ppe_finetuned.pt) provide these
# plus "person" for Zone tracking. The pipeline logs a warning at runtime if a
# loaded model is missing those classes (see Pipeline._validate_ppe_classes).

# ================================================================
# PPE TAXONOMY — the SACRED 11-class order baked into ppe_finetuned.pt.
# This is the single source of truth for training/collection class INDICES.
# It MUST match backend/data/train_dataset/roboflow_dl/data.yaml and
# training/merge_external.py:DEPLOYED. Do NOT reorder — indices are baked into
# the weights. utils/collector.py writes YOLO labels in THIS index space (a past
# bug wrote sorted(display-label) indices, a different class space entirely).
# ================================================================
PPE_TAXONOMY: list[str] = [
    "Vest", "boots", "glasses", "gloves", "helmet",
    "no_boots", "no_glasses", "no_gloves", "no_helmet", "no_vest", "person",
]

# Map a RAW model class string (whatever the active model emits) → PPE_TAXONOMY
# index, or None to drop. Handles the local fine-tuned model (emits the taxonomy
# names verbatim) AND cloud-model variants (hardhat, a vest, goggles, …).
_PPE_TAXO_ALIAS: dict[str, str] = {
    "hardhat": "helmet", "a_hardhat": "helmet", "a_helmet": "helmet", "helmet_on": "helmet",
    "no_hardhat": "no_helmet",
    "a_vest": "Vest", "safety_vest": "Vest", "no_safety_vest": "no_vest",
    "goggles": "glasses", "a_goggles": "glasses", "safety_glasses": "glasses",
    "no_goggles": "no_glasses", "no_goggle": "no_glasses",
    "safety_boots": "boots", "a_boots": "boots", "a_shoes": "boots",
    "no_safety_boots": "no_boots",
    "a_gloves": "gloves", "a_glove": "gloves", "no_glove": "no_gloves",
    "a_person": "person", "worker": "person",
}

def ppe_taxo_index(raw: str) -> "int | None":
    """Raw model class string → PPE_TAXONOMY index (or None if unmapped)."""
    norm = " ".join(str(raw).lower().split()).replace("-", "_").replace(" ", "_")
    # direct match against the taxonomy itself (local model emits these)
    for i, name in enumerate(PPE_TAXONOMY):
        if norm == name.lower().replace(" ", "_"):
            return i
    tgt = _PPE_TAXO_ALIAS.get(norm)
    return PPE_TAXONOMY.index(tgt) if tgt else None

# ================================================================
# MEDIAPIPE — Slide Module 3: 33 Keypoints, 3 Detection Methods
# ================================================================
MEDIAPIPE_MODEL_COMPLEXITY    = 1
FALL_KEYPOINT_VELOCITY_THRESH = 0.30
FALL_BBOX_RATIO_THRESH        = 0.72
FALL_CONFIRM_FRAMES           = 6
GAIT_HISTORY_FRAMES           = 30
GAIT_ANOMALY_THRESH           = 0.20

# ── Fall fusion (Hybrid = YOLO primary + MediaPipe cross-check) ──
# mode: hybrid | yolo | pose
FALL_MODE                 = os.getenv("FALL_MODE", "hybrid")
FALL_YOLO_CONFIDENCE      = float(os.getenv("FALL_YOLO_CONFIDENCE", "0.50"))
FALL_YOLO_CONFIRM_FRAMES  = int(os.getenv("FALL_YOLO_CONFIRM_FRAMES", "4"))   # need N fall-frames per track
FALL_CONFIRM_WINDOW       = int(os.getenv("FALL_CONFIRM_WINDOW", "6"))        # ...within the last M frames
FALL_ASSOC_OVERLAP        = float(os.getenv("FALL_ASSOC_OVERLAP", "0.30"))    # min overlap of a fall box inside a person box
# MediaPipe Pose is heavy. Run it only every Nth frame (and never in 'yolo'
# mode) so it doesn't cap the live pipeline FPS / freeze the Live view.
FALL_POSE_EVERY_N         = int(os.getenv("FALL_POSE_EVERY_N", "3"))

# ── FALL: pose-sequence Transformer (punpayut/Fall-Detection, MIT) ──
# The classifier wants 30 CONSECUTIVE frames (~1 s at 30 fps). The PPE detect loop
# skips frames at ~13 fps, which would stretch a fall to ~2.3 s and wreck the
# temporal signal, so fall runs in its own fixed-cadence loop at FALL_LOOP_FPS.
FALL_TFLITE_PATH      = os.getenv(
    "FALL_TFLITE_PATH", str(BASE_DIR / "assets" / "models" / "fall_detection_transformer.tflite"))
# Pose backend: "yolo" (one pass, every person, far better on distant workers —
# measured 75% vs 25% landmark hit-rate below 150px) or "mediapipe" (faithful to
# upstream; single-person, so it runs per-crop on a thread pool).
FALL_POSE_BACKEND     = os.getenv("FALL_POSE_BACKEND", "yolo")
FALL_POSE_MODEL       = os.getenv("FALL_POSE_MODEL", str(MODELS_DIR / "yolo11n-pose.pt"))
FALL_POSE_WORKERS     = int(os.getenv("FALL_POSE_WORKERS", "4"))   # mediapipe backend only
FALL_LOOP_FPS         = int(os.getenv("FALL_LOOP_FPS", "15"))
FALL_PROB_THRESHOLD   = float(os.getenv("FALL_PROB_THRESHOLD", "0.90"))  # upstream's value
# A window that is mostly no-pose zeros makes the probability meaningless; below
# this coverage the track is judged by the rule layer alone.
FALL_MIN_POSE_COVERAGE = float(os.getenv("FALL_MIN_POSE_COVERAGE", "0.6"))
# ── Rule layer: a fall is a TRANSITION (upright → wide + dropped → stays down) ──
# The first version tested a posture in isolation (w/h > 0.72 AND still), which on a
# close-up or overhead camera is true forever: a seated person alarmed every cooldown.
# Everything below is measured against each PERSON'S OWN upright baseline instead.
FALL_MOTIONLESS_SEC   = float(os.getenv("FALL_MOTIONLESS_SEC", "2.0"))   # must STAY down this long
FALL_UPRIGHT_AR       = float(os.getenv("FALL_UPRIGHT_AR", "0.8"))       # w/h at or below this = upright
FALL_BASELINE_SEC     = float(os.getenv("FALL_BASELINE_SEC", "10.0"))    # window for the upright baseline
FALL_AR_SPIKE         = float(os.getenv("FALL_AR_SPIKE", "1.8"))         # prone = w/h > spike x baseline
FALL_AR_ABS_MIN       = float(os.getenv("FALL_AR_ABS_MIN", "0.9"))       # ...with an absolute floor
FALL_AR_CORROBORATE   = float(os.getenv("FALL_AR_CORROBORATE", "1.25"))  # posture backing for the model
FALL_TRANSITION_SEC   = float(os.getenv("FALL_TRANSITION_SEC", "1.5"))   # upright→prone must happen within
FALL_DROP_MIN         = float(os.getenv("FALL_DROP_MIN", "0.02"))        # box centre must fall this much
# A fall needs a plausible body. Reject a head-and-shoulders close-up that fills the
# frame, and specks in the distance — their aspect ratio says nothing about posture.
# Body plausibility, measured on the box DIAGONAL so it does not change when the
# person goes from standing to lying. Real falls span 0.04-0.51 of the frame
# diagonal; a face filling the lens spans 0.72 and is rejected.
FALL_MIN_BODY_SPAN    = float(os.getenv("FALL_MIN_BODY_SPAN", "0.05"))
FALL_MAX_BODY_SPAN    = float(os.getenv("FALL_MAX_BODY_SPAN", "0.60"))
# A fall makes ByteTrack drop the id (tall box -> wide box, IoU collapses). Carry
# the fall history over to the replacement id instead of starting from zero.
FALL_REBIND_SEC       = float(os.getenv("FALL_REBIND_SEC", "3.0"))
FALL_REBIND_IOU       = float(os.getenv("FALL_REBIND_IOU", "0.2"))
# Alert de-duplication is SPATIAL, not per-track: ByteTrack ids churn badly on some
# cameras (observed #515 → #933 in minutes), and a per-id cooldown lets every new id
# re-alarm. Two falls near the same place within this window are one incident.
FALL_DEDUPE_SEC       = float(os.getenv("FALL_DEDUPE_SEC", "20.0"))
FALL_DEDUPE_RADIUS    = float(os.getenv("FALL_DEDUPE_RADIUS", "0.15"))   # fraction of frame diagonal

# ================================================================
# BYTETRACK — Slide Module 2: Multi-Object Tracking
# ================================================================
BYTETRACK_TRACK_THRESH = 0.50
BYTETRACK_TRACK_BUFFER = 30
# IoU gate for matching a track to a detection. Our ByteTracker has NO motion
# model (no Kalman), so at 10 fps a walking person can drop below a high IoU
# between frames and get a NEW id (id churn breaks per-track logic). 0.50
# tolerates normal frame-to-frame motion while still rejecting unrelated boxes.
BYTETRACK_MATCH_THRESH = float(os.getenv("BYTETRACK_MATCH_THRESH", "0.50"))

# Tracker config actually used by the in-process ultralytics Detector
# (utils/detect_track.py). Points at our tuned bytetrack yaml; falls back to
# ultralytics' packaged "bytetrack.yaml" if the file is missing. This is what
# makes ByteTrack stop churning ids — the BYTETRACK_* scalars above are kept only
# for backward compatibility and do not feed the deployed in-process Detector path.
_ZENTRA_TRACKER = BASE_DIR / "assets" / "trackers" / "bytetrack_zentra.yaml"
PPE_TRACKER_CONFIG = os.getenv(
    "PPE_TRACKER_CONFIG",
    str(_ZENTRA_TRACKER) if _ZENTRA_TRACKER.exists() else "bytetrack.yaml",
)

# ================================================================
# SAFETY ZONE — Slide Module 2
# ================================================================
ZONE_POLYGON_FILE = str(DATA_DIR / "zones.json")
MAX_ZONES         = 10
ZONE_USE_FOOT_POINT = os.getenv("ZONE_USE_FOOT_POINT", "true").lower() == "true"  # test feet, not bbox centre
ZONE_TRACK_MIN_HITS = int(os.getenv("ZONE_TRACK_MIN_HITS", "3"))   # ignore unstable tracks
ZONE_CONFIRM_FRAMES = int(os.getenv("ZONE_CONFIRM_FRAMES", "3"))   # (legacy frame-count — no longer drives zone)
ZONE_CONFIRM_WINDOW = int(os.getenv("ZONE_CONFIRM_WINDOW", "5"))   # (legacy frame-count — kept for back-compat)
# FPS-INDEPENDENT zone confirm (time-based, replaces the frame-count above for zone):
# confirmed when the foot point was inside for >= MIN_RATIO of the last WINDOW_SEC
# seconds, with an absolute floor of MIN_HITS inside observations. A fixed N-of-M
# frame window's time span swings with FPS (5 frames = 0.17s @30fps vs ~1.3s @4fps),
# so a fast walk-through at low FPS could miss 3-of-5. Seconds behave the same at any
# FPS. Lower WINDOW_SEC / MIN_RATIO = more sensitive (catches quick crossings, more FP).
ZONE_CONFIRM_WINDOW_SEC = float(os.getenv("ZONE_CONFIRM_WINDOW_SEC", "0.8"))
ZONE_CONFIRM_MIN_RATIO  = float(os.getenv("ZONE_CONFIRM_MIN_RATIO",  "0.5"))
ZONE_CONFIRM_MIN_HITS   = int(os.getenv("ZONE_CONFIRM_MIN_HITS",   "2"))

# ================================================================
# PPE — accuracy / debounce
# ================================================================
PPE_CONFIRM_FRAMES = int(os.getenv("PPE_CONFIRM_FRAMES", "3"))     # need N violation frames per track
PPE_CONFIRM_WINDOW = int(os.getenv("PPE_CONFIRM_WINDOW", "5"))     # ...within the last M frames (per track)
PPE_ASSOC_OVERLAP  = float(os.getenv("PPE_ASSOC_OVERLAP", "0.30")) # min fraction of a PPE box inside a person box to associate

# REQUIRED-PPE ABSENCE POLICY (fixes "no helmet → shows ปลอดภัย").
# For these categories, a person is in VIOLATION when the item is NOT positively
# detected as worn — i.e. absence counts, we do NOT wait for the model to emit an
# explicit "no_helmet" box (those negative classes are unreliable). Detecting a
# WORN helmet is far more reliable than detecting its absence, so we invert:
#   required cat  → violation = (state != WORN)     [missing OR explicit no_*]
#   other cat     → violation = (state == VIOLATION) [only an explicit no_* box]
# The 3-of-5 temporal confirm still debounces flicker, so a worn helmet that
# blinks out for a frame won't false-alarm. Narrow this list (e.g. just "helmet")
# if the model can't reliably see a class. Values are ppe_association CATEGORIES:
#   helmet, vest, gloves, glasses, boots
PPE_REQUIRED = [c.strip().lower() for c in
                os.getenv("PPE_REQUIRED", "helmet,vest").split(",") if c.strip()]

# ================================================================
# ALERT COOLDOWN — Slide: 3 ระดับ
# ================================================================
VIOLATION_COOLDOWN_SECONDS = 30
ZONE_COOLDOWN_SECONDS      = 20
FALL_COOLDOWN_SECONDS      = 15

ALERT_LEVEL_WARNING   = "warning"
ALERT_LEVEL_ALERT     = "alert"
ALERT_LEVEL_EMERGENCY = "emergency"

# ================================================================
# LINE OA — Slide: ส่งถึง หัวหน้างาน / Safety / Emergency
# ================================================================
LINE_OA_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_OA_CHANNEL_ACCESS_TOKEN", "")
LINE_OA_GROUP_SUPERVISOR     = os.getenv("LINE_OA_GROUP_SUPERVISOR", "")
LINE_OA_GROUP_SAFETY         = os.getenv("LINE_OA_GROUP_SAFETY",     "")
LINE_OA_GROUP_EMERGENCY      = os.getenv("LINE_OA_GROUP_EMERGENCY",  "")

_FB = os.getenv("LINE_OA_GROUP_ID", "")
if not LINE_OA_GROUP_SUPERVISOR: LINE_OA_GROUP_SUPERVISOR = _FB
if not LINE_OA_GROUP_SAFETY:     LINE_OA_GROUP_SAFETY     = _FB
if not LINE_OA_GROUP_EMERGENCY:  LINE_OA_GROUP_EMERGENCY  = _FB

ALERT_RECIPIENTS: dict[str, list[str]] = {
    ALERT_LEVEL_WARNING:   [LINE_OA_GROUP_SUPERVISOR],
    ALERT_LEVEL_ALERT:     [LINE_OA_GROUP_SAFETY, LINE_OA_GROUP_SUPERVISOR],
    ALERT_LEVEL_EMERGENCY: [LINE_OA_GROUP_EMERGENCY, LINE_OA_GROUP_SAFETY, LINE_OA_GROUP_SUPERVISOR],
}

DAILY_REPORT_TIME = os.getenv("DAILY_REPORT_TIME", "20:00")
# External public image host used ONLY when LINE_UPLOAD_IMAGES is opted-in.
# LINE push image messages require a public HTTPS URL we must host ourselves —
# LINE provides no upload-and-get-URL endpoint — so attaching a photo to a LINE
# alert unavoidably sends it through an external host.
IMAGE_UPLOAD_URL  = "https://catbox.moe/user/api.php"
# PDPA: default OFF so no person image ever leaves the device. Evidence photos
# are still kept locally (History). Turn on only with explicit consent — then
# LINE alerts attach the photo via IMAGE_UPLOAD_URL above.
LINE_UPLOAD_IMAGES = os.getenv("LINE_UPLOAD_IMAGES", "false").lower() == "true"

# ================================================================
# DATA COLLECTION
# ================================================================
AUTO_COLLECT_FRAMES      = True
COLLECT_VIOLATION_FRAMES = True
COLLECT_NORMAL_INTERVAL  = 300
COLLECT_MAX_PER_CLASS    = 2000
COLLECT_JPEG_QUALITY     = 90
# Dataset quality: keep DIVERSE frames, not 30 near-identical ones per moment
COLLECT_MIN_INTERVAL_SEC = float(os.getenv("COLLECT_MIN_INTERVAL_SEC", "2.0"))  # min gap between saves / category
COLLECT_DEDUP_DIFF       = float(os.getenv("COLLECT_DEDUP_DIFF", "8.0"))        # skip if 32x32 gray diff < this

# ================================================================
# TRAINING
# ================================================================
TRAIN_EPOCHS        = int(os.getenv("TRAIN_EPOCHS",     "80"))
TRAIN_PATIENCE      = int(os.getenv("TRAIN_PATIENCE",   "15"))   # early stop if no gain for N epochs
TRAIN_BATCH_SIZE    = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
TRAIN_IMG_SIZE      = int(os.getenv("TRAIN_IMG_SIZE", "768"))  # 768 helps small PPE (glasses/gloves); closer to infer PPE_IMGSZ=960
TRAIN_DEVICE        = os.getenv("TRAIN_DEVICE", "0")
TRAIN_WORKERS       = int(os.getenv("TRAIN_WORKERS", "8"))
TRAIN_LR0           = 0.001
TRAIN_LRF           = 0.01
TRAIN_MOMENTUM      = 0.937
TRAIN_WEIGHT_DECAY  = 0.0005
TRAIN_WARMUP_EPOCHS = 5
TRAIN_VAL_SPLIT     = 0.15
TRAIN_AUG           = os.getenv("TRAIN_AUG", "true").lower() == "true"  # used by training/trainer.py
YOLO_BASE_MODEL     = os.getenv("YOLO_BASE_MODEL", "yolov8m.pt")

# ================================================================
# DISPLAY — Windows 11
# ================================================================
WINDOW_TITLE   = "ZENTRA Smart Detection"
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
FONT_SCALE     = 0.5
FONT_THICKNESS = 1
OSD_COLOR      = (255, 255, 255)
OSD_BG_COLOR   = (20, 20, 20)

# ================================================================
# PERFORMANCE
# ================================================================
TARGET_FPS        = 60
FRAME_BUFFER_SIZE = 4
ENABLE_THREADING  = True
