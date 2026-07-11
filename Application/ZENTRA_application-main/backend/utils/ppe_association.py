#!/usr/bin/env python3
# utils/ppe_association.py — assign PPE / no_* boxes to person tracks by
# CONTAINMENT, then derive each person's per-category compliance state.
# ================================================================
# Containment (not IoU): frac = area(item ∩ person) / area(item). A helmet box is
# tiny vs a person box, so IoU is ~0 and useless; containment answers "is this
# helmet ON this person?". Assign each item to the person with the highest frac
# above PPE_ASSOC_OVERLAP (0.30).
#
# Per-category state ∈ {WORN, VIOLATION, UNKNOWN}. WORN wins over VIOLATION for
# the same category on the same person (a positive detection of the item means
# they have it → avoids false alarms from a spurious no_* box). UNKNOWN (no box
# either way) does NOT alarm by default — see the plan's absence policy.
# ================================================================
from __future__ import annotations

import config as cfg

CATEGORIES = ["helmet", "vest", "gloves", "glasses", "boots"]

# taxonomy index → (category, state)
_TAXO_TO_CAT_STATE = {
    0: ("vest", "WORN"), 1: ("boots", "WORN"), 2: ("glasses", "WORN"),
    3: ("gloves", "WORN"), 4: ("helmet", "WORN"),
    5: ("boots", "VIOLATION"), 6: ("glasses", "VIOLATION"),
    7: ("gloves", "VIOLATION"), 8: ("helmet", "VIOLATION"), 9: ("vest", "VIOLATION"),
}


def _containment(item: dict, person: dict) -> float:
    """area(item ∩ person) / area(item)."""
    ix1 = max(item["x1"], person["x1"]); iy1 = max(item["y1"], person["y1"])
    ix2 = min(item["x2"], person["x2"]); iy2 = min(item["y2"], person["y2"])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a = max(1e-6, (item["x2"] - item["x1"]) * (item["y2"] - item["y1"]))
    return inter / a


def associate(dets: list[dict], assoc_overlap: float | None = None,
              min_item_conf: float | None = None) -> list[dict]:
    """Return one record per person:
       {person: <det>, states: {cat: WORN|VIOLATION|UNKNOWN}, items: [dets]}.

    Detection now runs at a LOW tracking floor (PPE_TRACK_CONF) so ByteTrack ids
    stay stable, which means low-confidence PPE boxes also arrive here. Filter
    those out at INFERENCE_CONFIDENCE (the PPE slider) BEFORE associating, so
    stable tracking doesn't cost PPE precision. Persons are NOT filtered — the
    tracker's own new_track_thresh already gates which persons get an id."""
    thr = cfg.PPE_ASSOC_OVERLAP if assoc_overlap is None else assoc_overlap
    base_conf = cfg.INFERENCE_CONFIDENCE if min_item_conf is None else min_item_conf
    # Per-CATEGORY confidence floor (cfg.PPE_CLASS_CONF), falling back to the global
    # PPE slider. Small classes (glasses/gloves) score lower than a helmet, so a
    # slightly lower floor for them recovers recall without dropping the whole
    # slider. Keyed by ppe_association CATEGORY (helmet/vest/gloves/glasses/boots).
    class_conf = getattr(cfg, "PPE_CLASS_CONF", {}) or {}
    persons = [d for d in dets if d.get("is_person")]

    def _keep(d) -> bool:
        taxo = d.get("taxo")
        if taxo not in _TAXO_TO_CAT_STATE:
            return False
        cat = _TAXO_TO_CAT_STATE[taxo][0]
        return d.get("conf", 0.0) >= class_conf.get(cat, base_conf)

    items = [d for d in dets if not d.get("is_person") and _keep(d)]

    recs = [{"person": p, "states": {c: "UNKNOWN" for c in CATEGORIES}, "items": []}
            for p in persons]

    for it in items:
        best_j, best_frac = -1, thr
        for j, p in enumerate(persons):
            frac = _containment(it, p)
            if frac >= best_frac:
                best_frac, best_j = frac, j
        if best_j < 0:
            continue
        cat, state = _TAXO_TO_CAT_STATE[it["taxo"]]
        rec = recs[best_j]
        rec["items"].append(it)
        cur = rec["states"][cat]
        if state == "WORN":
            rec["states"][cat] = "WORN"           # WORN wins
        elif state == "VIOLATION" and cur != "WORN":
            rec["states"][cat] = "VIOLATION"
    return recs


def violations_of(rec: dict) -> list[str]:
    """Categories currently in VIOLATION state for one person record."""
    return [c for c, s in rec["states"].items() if s == "VIOLATION"]
