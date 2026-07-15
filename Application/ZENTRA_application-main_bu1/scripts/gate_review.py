#!/usr/bin/env python3
"""Stage gate review — render what the model actually got wrong.

A single mAP number cannot tell you whether a detector is safe to build on.
mAP50 0.75 could mean "misses distant workers" or "hallucinates a person in
every shadow", and those demand opposite fixes. So this renders the failures.

Produces, on the held-out split:

  * per-image FN/FP counts, ranked — the worst images first
  * a contact sheet: green = matched (TP), red = MISSED (FN), yellow = FALSE (FP)
  * a size breakdown of misses, because "recall 0.66" hides that recall on
    small/distant people may be near zero — and distant workers are exactly the
    ones a factory camera sees

Usage:
    python3 scripts/gate_review.py --weights backend/models/person_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 200, 255)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 4), np.float32)
    out = []
    for ln in label_path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(out, np.float32) if out else np.zeros((0, 4), np.float32)


def size_bucket(box, diag: float) -> str:
    w, h = box[2] - box[0], box[3] - box[1]
    frac = float(np.hypot(w, h)) / diag
    if frac < 0.10:
        return "tiny (<10% diag)"
    if frac < 0.25:
        return "small (10-25%)"
    if frac < 0.50:
        return "medium (25-50%)"
    return "large (>50%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", type=Path,
                    default=REPO / "backend/data/train_dataset/person_v1/data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--sheet-n", type=int, default=24, help="worst images on the sheet")
    ap.add_argument("--out", type=Path, default=REPO / "backend/models/person_v1/gate_review")
    # A COCO baseline predicts 80 classes. Without this, its cars, trucks and
    # traffic lights are all scored as false-positive "people" — which is how the
    # first run of this script reported precision 0.54 for a model whose real
    # person-precision is ~0.98. The fine-tuned model has one class and needs it not.
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="restrict predictions to these class ids (use 0 for a COCO model)")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(args.data.read_text())
    root = Path(cfg["path"])
    img_dir = root / args.split / "images"
    lab_dir = root / args.split / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"{len(imgs)} images in {args.split}")

    model = YOLO(str(args.weights))
    args.out.mkdir(parents=True, exist_ok=True)

    per_image, miss_sizes, hit_sizes = [], {}, {}
    tot_tp = tot_fp = tot_fn = 0

    for i in range(0, len(imgs), 16):
        batch = imgs[i:i + 16]
        kw = {"classes": args.classes} if args.classes else {}
        results = model.predict([str(p) for p in batch], imgsz=args.imgsz, conf=args.conf,
                                device="mps", verbose=False, **kw)
        for p, r in zip(batch, results):
            h, w = r.orig_shape
            diag = float(np.hypot(w, h))
            pred = (r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes)
                    else np.zeros((0, 4), np.float32))
            gt = load_gt(lab_dir / (p.stem + ".txt"), w, h)

            M = iou_matrix(gt, pred)
            matched_p, matched_g = set(), set()
            # greedy highest-IoU matching, one prediction per ground-truth box
            if M.size:
                order = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
                for gi, pi in order:
                    if M[gi, pi] < args.iou_match:
                        break
                    if gi in matched_g or pi in matched_p:
                        continue
                    matched_g.add(int(gi)); matched_p.add(int(pi))

            fn = [int(g) for g in range(len(gt)) if g not in matched_g]
            fp = [int(x) for x in range(len(pred)) if x not in matched_p]
            tot_tp += len(matched_g); tot_fn += len(fn); tot_fp += len(fp)

            for g in range(len(gt)):
                b = size_bucket(gt[g], diag)
                (hit_sizes if g in matched_g else miss_sizes)[b] = \
                    (hit_sizes if g in matched_g else miss_sizes).get(b, 0) + 1

            per_image.append({"img": str(p), "fn": len(fn), "fp": len(fp),
                              "tp": len(matched_g), "gt": len(gt),
                              "boxes": {"gt": gt.tolist(), "pred": pred.tolist(),
                                        "matched_g": sorted(matched_g), "fp": fp}})

    per_image.sort(key=lambda d: (-(d["fn"] + d["fp"]), d["img"]))

    # ── contact sheet of the worst images ────────────────────────────────
    cell, cols = 320, 6
    rows = (min(args.sheet_n, len(per_image)) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for k, rec in enumerate(per_image[:args.sheet_n]):
        im = cv2.imread(rec["img"])
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = np.array(rec["boxes"]["gt"], np.float32).reshape(-1, 4)
        pr = np.array(rec["boxes"]["pred"], np.float32).reshape(-1, 4)
        mg = set(rec["boxes"]["matched_g"]); fps = set(rec["boxes"]["fp"])
        for gi in range(len(gt)):
            c = GREEN if gi in mg else RED
            t = 2 if gi in mg else 3
            cv2.rectangle(im, tuple(gt[gi][:2].astype(int)), tuple(gt[gi][2:].astype(int)), c, t)
        for pi in fps:
            cv2.rectangle(im, tuple(pr[pi][:2].astype(int)), tuple(pr[pi][2:].astype(int)), YELLOW, 2)
        s = cell / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        cv2.putText(im, f"FN{rec['fn']} FP{rec['fp']}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        r0, c0 = (k // cols) * cell, (k % cols) * cell
        sheet[r0:r0 + im.shape[0], c0:c0 + im.shape[1]] = im
    sheet_p = args.out / f"contact_sheet_{args.split}.jpg"
    cv2.imwrite(str(sheet_p), sheet)

    recall = tot_tp / max(1, tot_tp + tot_fn)
    prec = tot_tp / max(1, tot_tp + tot_fp)
    print(f"\n@conf={args.conf} iou={args.iou_match}: TP={tot_tp} FN={tot_fn} FP={tot_fp}")
    print(f"  recall={recall:.4f}  precision={prec:.4f}")

    print("\nrecall by person size (this is what 'recall' averages over):")
    print(f"  {'bucket':<20}{'found':>8}{'missed':>8}{'recall':>9}")
    for b in ["tiny (<10% diag)", "small (10-25%)", "medium (25-50%)", "large (>50%)"]:
        hit, miss = hit_sizes.get(b, 0), miss_sizes.get(b, 0)
        if hit + miss:
            print(f"  {b:<20}{hit:>8}{miss:>8}{hit/(hit+miss):>9.3f}")

    summary = {"weights": str(args.weights), "split": args.split, "imgsz": args.imgsz,
               "conf": args.conf, "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
               "recall": round(recall, 4), "precision": round(prec, 4),
               "recall_by_size": {b: {"found": hit_sizes.get(b, 0), "missed": miss_sizes.get(b, 0)}
                                  for b in set(hit_sizes) | set(miss_sizes)},
               "worst_images": [{k: r[k] for k in ("img", "fn", "fp", "tp", "gt")}
                                for r in per_image[:20]]}
    (args.out / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(f"\ncontact sheet → {sheet_p}")
    print(f"summary       → {args.out / f'summary_{args.split}.json'}")


if __name__ == "__main__":
    main()
