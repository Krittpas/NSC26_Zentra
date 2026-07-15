#!/usr/bin/env python3
# utils/ppe_engine.py — one class that wraps the whole verified engine
# (ByteTrack detect_track + PPE association + temporal confirm + zone) behind a
# single `process(frame) -> (annotated_frame, events)` call, so the app pipeline
# (and the offline harness) share ONE implementation.
# ================================================================
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as cfg
from utils.detect_track import Detector, PersonDetector
from utils.ppe_association import associate, violations_of, CATEGORIES
from utils.temporal import TrackWindowConfirmer, TimeWindowConfirmer, CooldownGate
from utils import zone_geometry

GREEN, RED, CYAN = (0, 210, 0), (0, 0, 220), (255, 190, 0)

# Thai labels for confirmed missing-PPE categories (for alert messages)
_CAT_TH = {"helmet": "ไม่สวมหมวก", "vest": "ไม่สวมเสื้อกั๊ก", "gloves": "ไม่สวมถุงมือ",
           "glasses": "ไม่สวมแว่นตา", "boots": "ไม่สวมรองเท้าเซฟตี้"}
# Short Thai labels for on-frame box overlays
_CAT_SHORT_TH = {"helmet": "หมวก", "vest": "เสื้อกั๊ก", "gloves": "ถุงมือ",
                 "glasses": "แว่นตา", "boots": "รองเท้า"}

# Thai-capable font for on-frame overlays (cv2.putText can't render Thai → "???").
_FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Sarabun-SemiBold.ttf"
_FONT_CACHE: dict[int, "ImageFont.FreeTypeFont"] = {}


def _font(size: int):
    f = _FONT_CACHE.get(size)
    if f is None:
        try:
            f = ImageFont.truetype(str(_FONT_PATH), size)
        except Exception:
            f = ImageFont.load_default()
        _FONT_CACHE[size] = f
    return f


# ── Thai-correct overlay text ────────────────────────────────────────────────
# Pillow has no complex-script shaping, so Thai syllables with a stacked upper
# vowel AND a tone mark (e.g. "พื้นที่") render with the marks in the wrong place
# — looks broken on the video overlay. Shape with harfbuzz and rasterise glyphs
# with freetype so Thai renders correctly. Optional deps: if either is missing we
# fall back to plain Pillow (Latin/ASCII stays fine; Thai just isn't reshaped).
try:
    import uharfbuzz as _hb
    import freetype as _ft
    _SHAPE_OK = True
except Exception:
    _SHAPE_OK = False

_HB_FONT = None
_FT_FACE = None
_TEXT_IMG_CACHE: dict = {}       # (text, size, rgb) -> (RGBA strip, ascent_px)


def _shape_text_img(text: str, size: int, rgb: tuple):
    """RGBA image of `text` with Thai marks positioned correctly (harfbuzz +
    freetype), transparent background. Cached per (text, size, rgb) because
    overlay labels repeat every frame. Returns None if shaping is unavailable."""
    if not _SHAPE_OK or not text:
        return None
    key = (text, size, rgb)
    hit = _TEXT_IMG_CACHE.get(key)
    if hit is not None:
        return hit
    global _HB_FONT, _FT_FACE
    if _HB_FONT is None:
        _HB_FONT = _hb.Font(_hb.Face(_FONT_PATH.read_bytes()))
    if _FT_FACE is None:
        _FT_FACE = _ft.Face(str(_FONT_PATH))
    hbf, ftf = _HB_FONT, _FT_FACE
    hbf.scale = (size * 64, size * 64)
    buf = _hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    _hb.shape(hbf, buf)
    ftf.set_pixel_sizes(0, size)
    ascent = int(ftf.size.ascender / 64) or int(size * 1.15)
    descent = int(-ftf.size.descender / 64)
    height = max(1, ascent + descent + 2)
    glyphs, pen = [], 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        glyphs.append((info.codepoint, pen + pos.x_offset / 64.0, pos.y_offset / 64.0))
        pen += pos.x_advance / 64.0
    width = max(1, int(pen) + 2)
    strip = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for gid, gx, gy in glyphs:
        ftf.load_glyph(gid, _ft.FT_LOAD_RENDER)
        g = ftf.glyph
        w, h = g.bitmap.width, g.bitmap.rows
        if w == 0 or h == 0:
            continue
        alpha = np.frombuffer(bytes(g.bitmap.buffer), np.uint8).reshape(h, w)
        glyph = Image.new("RGBA", (w, h), tuple(rgb) + (0,))
        glyph.putalpha(Image.fromarray(alpha))
        strip.alpha_composite(glyph, (int(round(gx + g.bitmap_left)),
                                      int(round(ascent - g.bitmap_top - gy))))
    if len(_TEXT_IMG_CACHE) > 512:      # bound the cache (labels churn with ids)
        _TEXT_IMG_CACHE.clear()
    _TEXT_IMG_CACHE[key] = (strip, ascent)
    return _TEXT_IMG_CACHE[key]


class PPEEngine:
    @staticmethod
    def _resolve_required(ppe_items) -> set:
        """Which PPE categories THIS camera enforces (absence = violation). None →
        fall back to the global cfg.PPE_REQUIRED default (helmet, vest)."""
        if ppe_items is None:
            ppe_items = getattr(cfg, "PPE_REQUIRED", ["helmet", "vest"])
        return set(ppe_items) & set(CATEGORIES)

    def __init__(self, zones_path: str | None = None, device: str | None = None,
                 camera_id: str | None = None, roles: set | None = None,
                 ppe_items=None):
        # Two detectors, one job each: a COCO model finds + TRACKS people (high
        # recall, stable ids, works in crowds); the PPE fine-tune only classifies
        # items. They are merged back into one dets list before associate().
        self.person_detector = PersonDetector(device=device)
        self.person_detector.reset()
        # Per-camera roles gate what runs/draws (roles=None → all on, for the
        # offline harness). Computed here (was below) so the PPE item detector is
        # built ONLY when this camera actually enforces PPE.
        self._device = device
        self.ppe_enabled  = (roles is None) or ("ppe" in roles)
        self.zone_enabled = (roles is None) or ("zone" in roles)
        # ppe_finetuned.pt is git-ignored and absent on a fresh clone, while
        # person + zone + fall need NO custom PPE weights. If the PPE model is
        # missing, PAUSE PPE (status → off, honestly) instead of failing the whole
        # engine to a boxless passthrough — person/zone/fall keep detecting, which
        # is more honest than clean video with nothing drawn on it. Errors other
        # than a missing file still propagate.
        self.ppe_detector = None
        if self.ppe_enabled:
            try:
                self.ppe_detector = Detector(device=device)
            except FileNotFoundError as e:
                print(f"[PPEEngine] ⚠️ PPE model unavailable → PPE paused: {e}")
                self.ppe_enabled = False
        # Back-compat alias: pipeline logs + /api/status read engine.detector.*;
        # fall back to the person detector when there is no PPE model loaded.
        self.detector = self.ppe_detector or self.person_detector
        self.pconf = TrackWindowConfirmer(cfg.PPE_CONFIRM_FRAMES, cfg.PPE_CONFIRM_WINDOW)
        self.pcool = CooldownGate(cfg.VIOLATION_COOLDOWN_SECONDS)
        self.zconf = TimeWindowConfirmer(cfg.ZONE_CONFIRM_WINDOW_SEC,
                                         cfg.ZONE_CONFIRM_MIN_RATIO, cfg.ZONE_CONFIRM_MIN_HITS)
        self.zcool = CooldownGate(cfg.ZONE_COOLDOWN_SECONDS)
        self._zones_path = zones_path
        self._camera_id = camera_id            # only load zones for this camera (None = all)
        self.zones = zone_geometry.load_zones(zones_path, camera_id)
        # (tid, zone_id) currently confirmed-inside a danger zone. Drives rising-edge
        # alerts: a fresh entry (outside→inside) fires immediately, so leaving and
        # re-entering alerts again instead of being swallowed by the plain cooldown.
        self._zone_inside: set = set()
        # Fall is opt-IN even when roles is None: it loads a pose model + a TFLite
        # interpreter, which the PPE-only offline tools have no reason to pay for.
        self.fall_enabled = bool(roles) and ("fall" in roles)
        self.fconf = TrackWindowConfirmer(cfg.FALL_CONFIRM_FRAMES, cfg.FALL_CONFIRM_WINDOW)
        self.fcool = CooldownGate(cfg.FALL_COOLDOWN_SECONDS)
        self._fallen: set = set()      # confirmed-fallen track ids, for _draw
        self._fall_incidents: list = []   # (t, cx, cy) — spatial alert de-duplication
        self._fall = None
        self._build_fall()
        # Per-camera set of PPE categories to enforce (absence = violation).
        # Selectable in the app's per-camera roles modal; None → cfg.PPE_REQUIRED.
        self._required = self._resolve_required(ppe_items)

    def reload_zones(self):
        self.zones = zone_geometry.load_zones(self._zones_path, self._camera_id)

    def _build_fall(self):
        """Load the pose + fall models only for cameras that actually want fall.
        A failure must never take the pipeline down — the module just reports
        'standby' and PPE/zone keep running."""
        if not self.fall_enabled:
            self._fall = None
            return
        if self._fall is not None:
            return
        try:
            from utils.fall_detector import FallDetector
            self._fall = FallDetector()
            print(f"[PPEEngine] 🚑 fall detector ready "
                  f"(backend={self._fall.extractor.name}, mode={self._fall.mode})")
        except Exception as e:
            self._fall = None
            print(f"[PPEEngine] ⚠️ fall detector unavailable → {e}")

    @property
    def fall_ready(self) -> bool:
        return self.fall_enabled and self._fall is not None

    def apply_roles(self, roles: set | None):
        """Update which modules run/draw without reloading the model."""
        self.ppe_enabled  = (roles is None) or ("ppe" in roles)
        self.zone_enabled = (roles is None) or ("zone" in roles)
        # PPE just turned on for a camera that started without it → build the item
        # detector now. If the model is absent, pause PPE (off) rather than crash.
        if self.ppe_enabled and self.ppe_detector is None:
            try:
                self.ppe_detector = Detector(device=self._device)
                self.detector = self.ppe_detector
            except FileNotFoundError as e:
                print(f"[PPEEngine] ⚠️ PPE model unavailable → PPE paused: {e}")
                self.ppe_enabled = False
        was = self.fall_enabled
        self.fall_enabled = bool(roles) and ("fall" in roles)
        if self.fall_enabled and not was:
            self._build_fall()
        elif was and not self.fall_enabled:
            self.close_fall()

    def apply_ppe_items(self, ppe_items):
        """Update which PPE categories this camera enforces (no model reload)."""
        self._required = self._resolve_required(ppe_items)

    def refresh_tunables(self):
        """Re-read cfg-based gates (confirm windows + cooldowns) in place, so a
        Settings save takes effect without a costly YOLO reload. (Confidence is
        read per-call from cfg inside detect_track, so it needs nothing here.)"""
        self.pconf = TrackWindowConfirmer(cfg.PPE_CONFIRM_FRAMES, cfg.PPE_CONFIRM_WINDOW)
        self.pcool = CooldownGate(cfg.VIOLATION_COOLDOWN_SECONDS)
        self.zconf = TimeWindowConfirmer(cfg.ZONE_CONFIRM_WINDOW_SEC,
                                         cfg.ZONE_CONFIRM_MIN_RATIO, cfg.ZONE_CONFIRM_MIN_HITS)
        self.zcool = CooldownGate(cfg.ZONE_COOLDOWN_SECONDS)
        self.fconf = TrackWindowConfirmer(cfg.FALL_CONFIRM_FRAMES, cfg.FALL_CONFIRM_WINDOW)
        self.fcool = CooldownGate(cfg.FALL_COOLDOWN_SECONDS)
        if self._fall is not None:      # Settings sliders were dead until now
            self._fall.mode = cfg.FALL_MODE
            self._fall.threshold = float(cfg.FALL_PROB_THRESHOLD)
        # NOTE: _required is per-camera (set via ppe_items / apply_ppe_items), so it
        # is intentionally NOT reset here — refresh only re-reads cfg confirm/cooldown.

    def _cat_hit(self, rec, cat) -> bool:
        """Is `cat` a violation for this person THIS frame?
        Only categories this camera is set to check (self._required) can fire, and
        for those, ABSENCE counts (state != WORN) — we don't wait for an unreliable
        no_* box. Unchecked categories never fire."""
        if cat not in self._required:
            return False
        return rec["states"][cat] != "WORN"

    def _confirmed_missing(self, tid, rec) -> list[str]:
        """Categories currently hit AND temporally confirmed for this person —
        the single source of truth shared by detect() (alarms) and _draw()
        (labels) so the box colour never disagrees with the alert."""
        if tid is None:
            return []
        return [c for c in CATEGORIES
                if self._cat_hit(rec, c) and self.pconf.is_confirmed((tid, c))]

    def close_fall(self):
        """Release the pose backend (mediapipe leaks native graphs otherwise)."""
        if self._fall is not None:
            try:
                self._fall.close()
            except Exception:
                pass
        self._fall = None
        self._fallen.clear()
        self._fall_incidents.clear()

    def fall_step(self, frame, recs, now=None):
        """Runs from the pipeline's FIXED-CADENCE fall loop, not from detect():
        the classifier wants 30 evenly-spaced frames, and the detect loop skips
        frames. Every tracked person is evaluated — several people can fall in the
        same emergency, so each track gets its own confirm window and cooldown and
        emits its own event."""
        if not self.fall_ready or not recs:
            return []
        persons = [r["person"] for r in recs if r["person"].get("track_id") is not None]
        if not persons:
            return []
        try:
            results = self._fall.step(frame, persons, now=now)
        except Exception as e:
            print(f"[PPEEngine] fall_step: {e}")
            return []

        import time as _t
        h, w = frame.shape[:2]
        diag = (w * w + h * h) ** 0.5
        radius = float(getattr(cfg, "FALL_DEDUPE_RADIUS", 0.15)) * diag
        window = float(getattr(cfg, "FALL_DEDUPE_SEC", 20.0))
        tnow = _t.time()
        self._fall_incidents = [(t, x, y) for (t, x, y) in getattr(self, "_fall_incidents", [])
                                if tnow - t <= window]

        events, live = [], set()
        for tid, res in results.items():
            k = (tid,)
            live.add(k)
            confirmed = self.fconf.update(k, res.fallen)
            # Drive the RED box off the CONFIRMED state, not the instantaneous flag,
            # so a person on the floor doesn't flicker back to normal for a frame.
            if confirmed:
                self._fallen.add(tid)
            else:
                self._fallen.discard(tid)
            if not (confirmed and res.fallen and self.fcool.ready(k)):
                continue
            # De-duplicate by PLACE, not by track id. ByteTrack churns ids on some
            # cameras, and a per-id cooldown then lets each new id re-alarm for the
            # same person on the same patch of floor.
            p = next((r["person"] for r in recs if r["person"]["track_id"] == tid), None)
            if p is None:                      # track vanished between step() and here
                continue
            cx, cy = (p["x1"] + p["x2"]) / 2.0, (p["y1"] + p["y2"]) / 2.0
            if any(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 < radius
                   for (_, x, y) in self._fall_incidents):
                continue
            self._fall_incidents.append((tnow, cx, cy))
            events.append({"type": "fall", "track_id": tid, "key": k,
                           "level": cfg.ALERT_LEVEL_EMERGENCY,
                           "msg": f"⚠️ ตรวจพบการล้ม (คนที่ #{tid})"})
        self.fconf.gc(live)
        self._fallen &= {t for (t,) in live}
        return events

    def reset(self):
        self.person_detector.reset()
        if self.ppe_detector is not None:
            self.ppe_detector.reset()
        self._zone_inside.clear()

    def detect(self, frame):
        """Heavy step: track + associate + temporal confirm. Returns (recs, events).
        Call this from ONE worker thread (the tracker is stateful/persist=True)."""
        h, w = frame.shape[:2]
        # People (+ track ids) from the COCO detector; PPE items from the fine-tune.
        # Skip the PPE pass entirely when the role is off — saves a full forward.
        persons = self.person_detector.track(frame)
        items = (self.ppe_detector.detect_items(frame)
                 if self.ppe_enabled and self.ppe_detector is not None else [])
        recs = associate(persons + items)
        events = []
        live_p, live_z = set(), set()

        for rec in recs:
            person = rec["person"]
            tid = person["track_id"]
            if tid is None:
                continue
            # Exclusion zones (operator booth) mask a person out of ALL detection —
            # a privacy/masking primitive, applied even when the Zone ROLE is off so
            # a PPE-only camera still won't alarm people standing inside the booth.
            if zone_geometry.in_any_exclusion(person, self.zones, w, h):
                continue
            if self.ppe_enabled:
                for cat in CATEGORIES:
                    k = (tid, cat); live_p.add(k)
                    hit = self._cat_hit(rec, cat)     # required: absence counts
                    confirmed = self.pconf.update(k, hit)
                    if confirmed and hit and self.pcool.ready(k):
                        events.append({"type": "ppe", "track_id": tid, "key": k,
                                       "level": cfg.ALERT_LEVEL_WARNING,
                                       "msg": f"คนที่ #{tid}: {_CAT_TH.get(cat, 'ไม่สวม '+cat)}"})
            if self.zone_enabled:
                # Record inside/outside for EVERY danger zone every frame (like PPE)
                # so the 3-of-5 window is real. Previously only inside-frames were
                # pushed and gc() reset the key the moment the person stepped out, so
                # a worker straddling the zone edge (foot-point jitter) could never
                # reach 3-of-5 and evade the intrusion alarm.
                hit_ids = {z.id for z in zone_geometry.danger_hits(person, self.zones, w, h)}
                for z in self.zones:
                    if z.type != "danger":
                        continue
                    zk = (tid, z.id); live_z.add(zk)
                    inside = z.id in hit_ids
                    confirmed = self.zconf.update(zk, inside)   # ≥3 of last 5 inside
                    if confirmed and inside:
                        # Rising edge (fresh entry) alerts immediately; a continuous
                        # stay re-alerts every cooldown. Tracking "already inside" per
                        # (track, zone) is what makes leave→re-enter fire again — the
                        # plain 20s cooldown alone suppressed re-entries and, once the
                        # person kept one track id, effectively alarmed only once.
                        if zk not in self._zone_inside:
                            self._zone_inside.add(zk)
                            self.zcool.mark(zk)
                            fire = True
                        else:
                            fire = self.zcool.ready(zk)
                        if fire:
                            events.append({"type": "zone", "track_id": tid, "key": zk,
                                           "level": cfg.ALERT_LEVEL_ALERT,
                                           "msg": f"บุกรุกพื้นที่อันตราย '{z.name}' (คนที่ #{tid})"})
                    elif not confirmed:
                        self._zone_inside.discard(zk)   # debounced exit → next entry re-alerts
        self.pconf.gc(live_p); self.zconf.gc(live_z)
        self._zone_inside &= live_z      # forget zones for tracks that left the frame
        return recs, events

    def draw_on(self, frame, recs):
        """Light step: draw the latest recs onto ANY frame (fast — no inference).
        Lets the display run at camera FPS while detection runs slower."""
        h, w = frame.shape[:2]
        return self._draw(frame, recs or [], w, h)

    def process(self, frame):
        """Convenience (offline harness): detect + draw in one call."""
        recs, events = self.detect(frame)
        return self.draw_on(frame, recs), events

    def _draw(self, frame, recs, w, h):
        # Draw shapes with cv2 (fast); collect text to render once with PIL so
        # Thai labels show correctly (cv2.putText → "???"). Colors below are BGR.
        texts = []   # (x, y_top, text, rgb)
        if self.zone_enabled:
            for z in self.zones:
                poly = z.polygon_px(w, h)
                col = RED if z.type == "danger" else (140, 140, 140)
                cv2.polylines(frame, [poly], True, col, 2)
                x0, y0 = int(poly[0][0]), int(poly[0][1])
                texts.append((x0, max(2, y0 - 22), z.name, (col[2], col[1], col[0])))
        # Person boxes are relevant to both PPE and Zone; skip entirely only when
        # neither role is on. PPE status/colour is shown only when PPE is enabled.
        if self.ppe_enabled or self.zone_enabled or self.fall_enabled:
            for rec in recs:
                p = rec["person"]; tid = p["track_id"]
                x1, y1, x2, y2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
                # A confirmed fall outranks every PPE status — it's the emergency.
                if tid in self._fallen:
                    color = RED
                    label = f"#{tid} ล้ม!"
                # Show a PPE status only when PPE is on AND this camera actually
                # checks at least one item — otherwise a neutral box, never a
                # premature "ปลอดภัย" (which would falsely read as "PPE compliant"
                # when nothing is being checked).
                elif self.ppe_enabled and self._required:
                    # Same hit+confirm rule as detect() → box never disagrees with
                    # the alert. Three honest states:
                    #   RED   = a required item is confirmed missing
                    #   GREEN = every required item is positively worn (verified safe)
                    #   CYAN  = still verifying (a required item not yet seen worn,
                    #           but not confirmed-missing) — never a premature "safe"
                    cviol = self._confirmed_missing(tid, rec)
                    required_ok = all(rec["states"][c] == "WORN" for c in self._required)
                    if cviol:
                        color = RED
                        label = f"#{tid} ขาด: " + ",".join(_CAT_SHORT_TH.get(c, c) for c in cviol)
                    elif required_ok:
                        color = GREEN
                        label = f"#{tid} ปลอดภัย"
                    else:
                        color = CYAN
                        label = f"#{tid} กำลังตรวจสอบ"
                else:
                    color = CYAN
                    label = f"#{tid}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                texts.append((x1, max(2, y1 - 22), label, (color[2], color[1], color[0])))
                cv2.circle(frame, (int(p['foot'][0]), int(p['foot'][1])), 4, color, -1)
        return self._draw_texts(frame, texts)

    def _draw_texts(self, frame, items, size: int = 18):
        """Render all overlay text in a single BGR→PIL→BGR pass, each with a dark
        pill background for legibility on video. Uses harfbuzz shaping so Thai
        renders correctly (stacked vowel + tone mark); falls back to plain Pillow
        when the shaping deps are absent."""
        if not items:
            return frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if _SHAPE_OK:
            for (x, y, text, rgb) in items:
                shaped = _shape_text_img(text, size, tuple(int(c) for c in rgb))
                if not shaped:
                    continue
                strip, _asc = shaped
                w, h = strip.size
                x, y = int(x), int(y)
                pill = Image.new("RGBA", (w + 8, h + 4), (0, 0, 0, 150))
                img.paste(pill, (x - 4, y - 2), pill)
                img.paste(strip, (x, y), strip)
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # Fallback: plain Pillow (no complex-script shaping)
        d = ImageDraw.Draw(img, "RGBA")
        font = _font(size)
        for (x, y, text, rgb) in items:
            if not text:
                continue
            box = d.textbbox((x, y), text, font=font)
            d.rectangle([box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2], fill=(0, 0, 0, 150))
            d.text((x, y), text, font=font, fill=rgb, stroke_width=1, stroke_fill=(0, 0, 0, 255))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
