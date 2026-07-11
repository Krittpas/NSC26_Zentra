#!/usr/bin/env python3
# utils/temporal.py — generic per-key temporal confirmation + cooldown gate,
# shared by PPE (per track_id+category) and Zone (per track_id+zone_id).
# ================================================================
# TrackWindowConfirmer: a boolean is "confirmed" only when it was True on at
# least `confirm_frames` of the last `window` observations for that key. This is
# what turns a single-frame flicker into a real, alarm-worthy event.
#
# CooldownGate: after firing for a key, suppress re-firing for `seconds`.
# ================================================================
from __future__ import annotations
import time
from collections import defaultdict, deque


class TrackWindowConfirmer:
    def __init__(self, confirm_frames: int, window: int):
        self.confirm_frames = confirm_frames
        self.window = window
        self._hist: dict = defaultdict(lambda: deque(maxlen=window))

    def update(self, key, hit: bool) -> bool:
        """Record one observation for `key`; return True if now confirmed."""
        dq = self._hist[key]
        dq.append(bool(hit))
        return sum(dq) >= self.confirm_frames

    def is_confirmed(self, key) -> bool:
        dq = self._hist.get(key)
        return bool(dq) and sum(dq) >= self.confirm_frames

    def drop(self, key):
        self._hist.pop(key, None)

    def gc(self, live_keys: set):
        """Forget keys no longer present (e.g. tracks that left)."""
        for k in list(self._hist):
            if k not in live_keys:
                self._hist.pop(k, None)


class CooldownGate:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last: dict = {}

    def ready(self, key, now: float | None = None) -> bool:
        """True if `key` hasn't fired within `seconds`; records the fire time."""
        now = time.time() if now is None else now
        last = self._last.get(key, 0.0)
        if now - last >= self.seconds:
            self._last[key] = now
            return True
        return False
