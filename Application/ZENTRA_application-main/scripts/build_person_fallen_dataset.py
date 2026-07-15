#!/usr/bin/env python3
"""Build the person dataset that finally includes people who are ON THE FLOOR.

The problem this exists to fix
------------------------------
Measured 2026-07-14 on URFD: a COCO-pretrained person detector — yolo11s, m, and
the fine-tuned person_v1 — returns ZERO detections for a person lying on the floor
who is plainly visible to a human. yolo11l and yolo11x find them only at conf
0.17-0.24, and are far too slow for this CPU.

Every module in ZENTRA stands on the person box. So a worker who falls stops
existing: fall detection cannot confirm they stayed down, the danger zone no
longer sees them, PPE no longer sees them. The person who most needs help is the
one the system loses. That is the bug.

It is not a model-size problem. It is a DATA problem: COCO's `person` annotations
are overwhelmingly upright people, so "human, horizontal" is a shape the detector
has essentially never been shown.

The fix
-------
Fine-tune on people in every posture, and keep showing it upright people at the
same time so it does not trade one for the other.

  fall-detection-ca3o8   4,497 images, ONE class: `Fall-Detected`. It boxes the
                         person who has fallen — and nobody else. Mapped to
                         `person`. We are not teaching the detector to classify a
                         fall (the Transformer does that); we are teaching it that
                         a human lying down is still a human.

  people-detection       the upright set person_v1 was trained on. Mixed back in
                         to prevent catastrophic forgetting. Not hypothetical: an
                         earlier person_v1 trained on coco128 forgot COCO entirely
                         and scored WORSE than stock yolo11s.

⚠️  THE TRAP IN THE FALL SET, AND WHY --autolabel-standing EXISTS
-----------------------------------------------------------------
fall-detection-ca3o8 labels ONLY the fallen person. Any bystander still on their
feet is left unboxed — and YOLO reads an unboxed person as background. Train on
that and you are explicitly teaching the detector to IGNORE people who are
standing up, which is the one thing it currently does well.

This is the exact trap that ruined the PPE dataset (docs/TRAINING_PIPELINE.md:
"a frame with no person box does not mean nobody is in the frame ... feeding it as
background teaches the model to overlook people").

The fix uses each model where it is strong:
  fallen people   → the dataset's human labels   (yolo11s is blind to them)
  standing people → yolo11s at high confidence   (it is excellent at them: 0.90)
A yolo11s box that overlaps no existing label is a person the annotator skipped,
so we add it. Boxes are never REPLACED — human labels always win.

NOT USED, DELIBERATELY: URFD. It is the fall evaluation set. Training the person
detector on it would make every recall number `fall_eval.py` reports a fiction.

Usage
-----
    python scripts/build_person_fallen_dataset.py \
        --fall  <roboflow fall-detection export dir> \
        --people <roboflow people-detection export dir> \
        --out    backend/data/train_dataset/person_v2
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

IMG_EXT = {".jpg", ".jpeg", ".png"}

# Every one of these is a person. `fallen` and `falling` are the whole point of
# this dataset; `standing` comes along so the model keeps seeing upright people in
# the same domain. Anything NOT matched here is reported and refused — silently
# dropping a class is how a label space quietly goes wrong.
PERSON_ALIASES = {
    "person", "people", "human", "pedestrian", "worker",
    "standing", "stand", "upright",
    "falling", "fall", "fall-detected", "fall_detected",
    "fallen", "fallen-person", "lying", "lie", "laying", "down",
    "sitting", "sit", "seated", "crouching", "bending",
}
# Which source classes are the ones we are short of. Used only to build a
# FALLEN-ONLY validation split — see why in `main`.
FLOOR_ALIASES = {"falling", "fall", "fall-detected", "fall_detected",
                 "fallen", "fallen-person", "lying", "lie", "laying", "down"}


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def _to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class StandingAutoLabeler:
    """Add the standing people the fall dataset's annotator never boxed.

    fall-detection-ca3o8 boxes the fallen person and nobody else. Every bystander
    left on their feet is, to YOLO, background — so training on those images
    teaches the detector to ignore upright people, destroying the one thing it is
    currently good at.

    yolo11s is the right tool for exactly this and no other: it is excellent on
    upright people (conf 0.90) and completely blind to fallen ones (0 detections,
    measured). So anything it finds at HIGH confidence that does not overlap a
    human label is a standing person the annotator skipped. Add it.

    Human labels are never touched — this only ADDS boxes that nothing covers.
    """

    def __init__(self, conf: float = 0.5, iou_thr: float = 0.3):
        from ultralytics import YOLO
        self.model = YOLO("yolo11s.pt")
        self.conf, self.iou_thr = conf, iou_thr
        self.added = 0

    def extra_boxes(self, img: Path, existing: list[tuple]) -> list[str]:
        r = self.model.predict(str(img), conf=self.conf, classes=[0],
                               verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        h, w = r.orig_shape
        out = []
        for b in r.boxes.xyxy.tolist():
            n = (b[0] / w, b[1] / h, b[2] / w, b[3] / h)          # → normalized xyxy
            if any(_iou(n, e) > self.iou_thr for e in existing):
                continue                                          # already labelled
            cx, cy = (n[0] + n[2]) / 2, (n[1] + n[3]) / 2
            bw, bh = n[2] - n[0], n[3] - n[1]
            out.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            self.added += 1
        return out


def _norm(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(name).strip().lower()).strip("-")


def base_name(p: Path) -> str:
    """Group key for a leak-free split.

    Roboflow bakes augmented copies of one photo into several files, all sharing a
    base name before the `.rf.<hash>` suffix. Splitting on the FILE puts copies of
    the same photo in both train and val, and the resulting mAP is a lie — this is
    exactly how the old PPE model came to report 0.698. Split on the PHOTO.
    """
    n = p.name
    return n.split(".rf.")[0] if ".rf." in n else p.stem


def split_of(base: str, val: float, test: float) -> str:
    """Deterministic split from a hash of the photo, not from file order — so a
    re-run, or a different filesystem ordering, produces the exact same split."""
    h = int(hashlib.md5(base.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def load_names(root: Path) -> dict[int, str]:
    y = root / "data.yaml"
    if not y.exists():
        sys.exit(f"ไม่พบ {y}")
    d = yaml.safe_load(y.read_text(encoding="utf-8"))
    names = d.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names or [])}


def collect(root: Path) -> list[Path]:
    out = []
    for split in ("train", "valid", "val", "test"):
        d = root / split / "images"
        if d.is_dir():
            out += [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fall", type=Path, required=True,
                    help="fall-detection-ca3o8 export (standing/falling/fallen)")
    ap.add_argument("--people", type=Path, default=None,
                    help="upright person export — mixed in to prevent forgetting")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--autolabel-standing", action="store_true",
                    help="เติมกล่อง 'คนยืน' ที่ dataset คนล้มไม่ได้ label ไว้ (ใช้ yolo11s) "
                         "— ถ้าไม่เติม เท่ากับสอนโมเดลว่าคนยืนคือพื้นหลัง")
    ap.add_argument("--autolabel-conf", type=float, default=0.5)
    args = ap.parse_args()

    sources = [("fall", args.fall)] + ([("people", args.people)] if args.people else [])
    if not args.people:
        print("⚠️  ไม่ได้ใส่ --people → เสี่ยงลืมคนยืน (catastrophic forgetting)\n")

    for _, r in sources:
        if not r.is_dir():
            sys.exit(f"ไม่พบโฟลเดอร์ {r}")

    # ── Resolve every source class to `person`, or refuse ────────────────────
    keep: dict[str, dict[int, bool]] = {}     # source -> {cls_id: is_floor_pose}
    for tag, root in sources:
        names = load_names(root)
        mapping, unknown = {}, []
        for i, n in names.items():
            k = _norm(n)
            if k in PERSON_ALIASES:
                mapping[i] = k in FLOOR_ALIASES
            else:
                unknown.append(n)
        if unknown:
            sys.exit(f"❌ {tag}: ไม่รู้จักคลาส {unknown}\n"
                     f"   เพิ่มลง PERSON_ALIASES (ถ้าเป็นคน) หรือแก้สคริปต์ให้ทิ้งอย่างตั้งใจ\n"
                     f"   — การทิ้งคลาสเงียบ ๆ คือวิธีที่ label space พังโดยไม่มีใครรู้")
        keep[tag] = mapping
        floor = [names[i] for i, f in mapping.items() if f]
        print(f"{tag:7} classes → person : {sorted(names.values())}")
        print(f"{' ':7} ของที่เราขาด     : {floor or '(ไม่มี)'}")

    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (args.out / split / sub).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    floor_val_imgs: list[str] = []      # val images that contain a person ON THE FLOOR

    # Only the fall set needs its standing people recovered; people-detection
    # already boxes everyone.
    auto = StandingAutoLabeler(conf=args.autolabel_conf) if args.autolabel_standing else None
    if auto:
        print("\n🔎 auto-label คนยืนที่ตกหล่นใน fall set (yolo11s, conf "
              f"{args.autolabel_conf}) — อาจใช้เวลาสักครู่\n")

    for tag, root in sources:
        imgs = collect(root)
        print(f"\n{tag}: {len(imgs)} ภาพ")
        for img in imgs:
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
            lines, has_floor, boxes = [], False, []
            if lbl.exists():
                for ln in lbl.read_text(encoding="utf-8").splitlines():
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    if cid not in keep[tag]:
                        continue
                    has_floor |= keep[tag][cid]
                    lines.append("0 " + " ".join(parts[1:5]))   # every class → person
                    boxes.append(_to_xyxy(*(float(v) for v in parts[1:5])))
            # Recover the people the annotator left out. Without this, an unboxed
            # bystander is a NEGATIVE example of a person, and we would be training
            # the detector to overlook exactly the upright people it is good at.
            if auto is not None and tag == "fall":
                lines += auto.extra_boxes(img, boxes)
            # An image with genuinely no person is a legitimate background frame and
            # is kept — it teaches what is NOT a person.
            base = base_name(img)
            sp = split_of(f"{tag}/{base}", args.val, args.test)
            name = f"{tag}_{img.stem}"
            shutil.copy2(img, args.out / sp / "images" / f"{name}{img.suffix}")
            (args.out / sp / "labels" / f"{name}.txt").write_text(
                "\n".join(lines), encoding="utf-8")
            stats[f"{sp}_img"] += 1
            stats[f"{sp}_box"] += len(lines)
            if has_floor:
                stats[f"{sp}_floor_img"] += 1
                if sp == "val":
                    floor_val_imgs.append(f"{name}{img.suffix}")

    (args.out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(args.out.resolve()),
        "train": "train/images", "val": "val/images", "test": "test/images",
        "nc": 1, "names": ["person"],
    }, sort_keys=False), encoding="utf-8")

    # A FALLEN-ONLY val list. Overall mAP hides exactly the failure we are fixing:
    # a model can score beautifully by nailing the upright majority while still
    # missing every person on the floor — which is the state person_v1 is in today
    # (great mAP, zero recall on a fallen worker). Score that subset on its own, or
    # you will not know whether this worked.
    (args.out / "val_fallen.txt").write_text("\n".join(floor_val_imgs), encoding="utf-8")

    print("\n" + "=" * 58)
    for sp in ("train", "val", "test"):
        print(f"  {sp:5}  {stats[f'{sp}_img']:6d} ภาพ  {stats[f'{sp}_box']:7d} กล่อง  "
              f"· มีคนอยู่บนพื้น {stats[f'{sp}_floor_img']:5d} ภาพ")
    print("=" * 58)
    if auto is not None:
        print(f"เติมกล่องคนยืนที่ตกหล่น: {auto.added} กล่อง "
              f"(ถ้าไม่เติม กล่องเหล่านี้จะกลายเป็นตัวอย่าง 'พื้นหลัง' ที่สอนให้โมเดลมองข้ามคนยืน)")
    elif any(t == "fall" for t, _ in sources):
        print("⚠️  ไม่ได้ใช้ --autolabel-standing: คนยืนใน fall set ที่ไม่ถูก label\n"
              "    จะถูกเรียนรู้เป็น 'พื้นหลัง' → โมเดลอาจเริ่มมองข้ามคนยืน")
    print(f"data.yaml   → {args.out / 'data.yaml'}")
    print(f"val ที่มีคนล้ม → {args.out / 'val_fallen.txt'}  ({len(floor_val_imgs)} ภาพ)")
    print("\n⚠️  URFD ไม่ได้ถูกใส่เข้ามาโดยเจตนา — มันคือชุดทดสอบของ fall_eval.py")


if __name__ == "__main__":
    main()
