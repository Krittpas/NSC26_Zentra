#!/usr/bin/env python3
# training/merge_external.py — Merge external YOLO datasets into ZENTRA's 11-class
# taxonomy and combine with the balanced v3 subset (for a single MPS-efficient run).
# ================================================================
# WHY a new merger (not train_ppe_local.py): the old remap() used a dangerous
# PARTIAL substring match (e.g. "no-hardhat" could match the "hardhat" alias →
# helmet, i.e. a violation mislabelled as compliant) and mapped bare "head" → helmet.
# This module uses ONLY explicit, audited mappings — unknown classes are DROPPED
# with a printed warning, never guessed.
#
# Taxonomy index order is SACRED (must match the deployed ppe_finetuned.pt):
#   0=Vest 1=boots 2=glasses 3=gloves 4=helmet 5=no_boots 6=no_glasses
#   7=no_gloves 8=no_helmet 9=no_vest 10=person
#
# Non-destructive: writes remapped labels + image symlinks into <out>/ext_<name>/,
# never touches the source datasets. Produces a combined train list + data.yaml.
#
# USAGE:
#   python -m training.merge_external \
#       --balanced data/train_dataset/balanced/data.yaml \
#       --external data/external/data.yaml \
#       --out data/train_dataset/merged
# ================================================================
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import yaml

DEPLOYED = ["Vest", "boots", "glasses", "gloves", "helmet", "no_boots",
            "no_glasses", "no_gloves", "no_helmet", "no_vest", "person"]
IDX = {n: i for i, n in enumerate(DEPLOYED)}

# Explicit external-class-name (lowercased) -> ZENTRA class name, or None to DROP.
# Every entry is deliberate. Anything not listed is dropped with a warning.
ALIAS: dict[str, str | None] = {
    # ── Ultralytics Construction-PPE ──
    "helmet": "helmet", "gloves": "gloves", "vest": "Vest", "boots": "boots",
    "goggles": "glasses", "person": "person", "none": None,
    "no_helmet": "no_helmet", "no_goggle": "no_glasses",
    "no_gloves": "no_gloves", "no_boots": "no_boots",
    # ── Construction Site Safety (Roboflow) ──
    "hardhat": "helmet", "no-hardhat": "no_helmet",
    "safety vest": "Vest", "no-safety vest": "no_vest",
    "safety shoes": None,        # generic shoes != safety boots → drop (don't corrupt boots)
    "mask": None, "no-mask": None, "safety cone": None, "safety net": None,
    "barricade": None, "dumpster": None, "ladder": None,
    # ── siabar workplace-safety extra classes ──
    "glove": "gloves", "glass": "glasses",
    "ear-protection": None, "ear_protection": None, "earmuffs": None,
    "excavators": None, "excavator": None, "dump truck": None, "truck": None,
    "truck and trailer": None, "trailer": None, "semi": None, "van": None,
    "mini-van": None, "wheel loader": None, "machinery": None, "vehicle": None,
    "bus": None, "suv": None, "sedan": None, "fire hydrant": None,
    # ── common eyeglasses-dataset names (optional, selfie-domain) ──
    "glass": "glasses", "eyeglasses": "glasses", "with_glasses": "glasses",
    "wearing glasses": "glasses", "no_glasses": "no_glasses",
    "without_glasses": "no_glasses", "face": None, "sunglasses": None,
}


def _img_for_label(lbl: Path) -> Path | None:
    # image path = label path with last 'labels' -> 'images', trying extensions.
    # NOTE: build the name by string, NOT Path.with_suffix — Roboflow filenames
    # contain dots (e.g. "img_jpg.rf.<hash>.txt") and with_suffix would mangle
    # everything after the last dot.
    parts = list(lbl.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "labels":
            parts[i] = "images"
            break
    else:
        return None
    fname = parts[-1]
    stem_name = fname[:-4] if fname.lower().endswith(".txt") else fname.rsplit(".", 1)[0]
    img_dir = Path(*parts[:-1])
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp",
                ".JPG", ".JPEG", ".PNG"):
        cand = img_dir / (stem_name + ext)
        if cand.exists():
            return cand
    return None


def remap_external(ext_yaml: Path, out_dir: Path) -> tuple[list[str], Counter, Counter]:
    d = yaml.safe_load(ext_yaml.read_text())
    names = d["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    root = ext_yaml.parent
    name = root.name if root.name not in (".", "external") else ext_yaml.stem
    ext_out = out_dir / f"ext_{name}"
    (ext_out / "images").mkdir(parents=True, exist_ok=True)
    (ext_out / "labels").mkdir(parents=True, exist_ok=True)

    kept_paths: list[str] = []
    mapped, dropped = Counter(), Counter()
    unknown: set[str] = set()

    for lbl in root.rglob("*.txt"):
        if lbl.name.lower() in ("readme.txt",) or "labels" not in lbl.parts:
            continue
        img = _img_for_label(lbl)
        if img is None:
            continue
        new_lines = []
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            src = names[cid].lower().strip() if cid < len(names) else "?"
            if src not in ALIAS:
                unknown.add(src)
                dropped[src] += 1
                continue
            tgt = ALIAS[src]
            if tgt is None:
                dropped[src] += 1
                continue
            new_lines.append(f"{IDX[tgt]} {' '.join(line.split()[1:])}")
            mapped[tgt] += 1
        if not new_lines:
            continue  # image has no ZENTRA-relevant objects → skip entirely
        stem = f"{name}_{img.stem}"
        link = ext_out / "images" / f"{stem}{img.suffix}"
        if not link.exists():
            link.symlink_to(img.resolve())
        (ext_out / "labels" / f"{stem}.txt").write_text("\n".join(new_lines) + "\n")
        # IMPORTANT: append the SYMLINK path (already absolute — out_dir is
        # resolved in main), NOT link.resolve(). ultralytics derives the label
        # path by swapping images->labels on this string, so it must point at
        # our REMAPPED labels dir, not the original dataset's labels.
        kept_paths.append(str(link))

    if unknown:
        print(f"  ⚠️  DROPPED unknown classes (not in ALIAS): {sorted(unknown)}")
    return kept_paths, mapped, dropped


def main():
    ap = argparse.ArgumentParser(description="Merge external datasets into ZENTRA 11-class + balanced v3")
    ap.add_argument("--balanced", required=True, help="balanced v3 data.yaml (train list already 11-class)")
    ap.add_argument("--external", nargs="+", required=True, help="one or more external data.yaml paths")
    ap.add_argument("--out", required=True, help="output dir for merged data.yaml + combined list")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()   # absolute so symlink paths are absolute
    out_dir.mkdir(parents=True, exist_ok=True)

    bal = yaml.safe_load(Path(args.balanced).read_text())
    assert bal["names"] == DEPLOYED, f"balanced taxonomy mismatch: {bal['names']}"
    base_list = Path(bal["train"]).read_text().splitlines()
    val_dir = bal["val"]
    print(f"balanced v3 images: {len(base_list)}")

    all_paths = list(base_list)
    for ext in args.external:
        print(f"\n── remapping {ext} ──")
        paths, mapped, dropped = remap_external(Path(ext), out_dir)
        print(f"  kept images: {len(paths)}")
        print(f"  mapped instances: {dict(sorted(mapped.items()))}")
        all_paths.extend(paths)

    list_path = out_dir / "merged_train.txt"
    list_path.write_text("\n".join(all_paths) + "\n")
    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()),
        "train": str(list_path.resolve()),
        "val": val_dir,
        "nc": len(DEPLOYED),
        "names": DEPLOYED,
    }, sort_keys=False, allow_unicode=True))
    print(f"\n✅ combined train images: {len(all_paths)}")
    print(f"✅ wrote {out_yaml}")


if __name__ == "__main__":
    main()
