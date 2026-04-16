"""
visHand — Gesture Interpreter
==============================
The heart of visHand.  Given smoothed 21-point landmarks it:

  1. Detects SNAP events (two-phase: contact → SNAP_READY, release → SNAP)
  2. Classifies the current gesture intent
  3. Drives the logic state machine  (LOCKED → HOVER → ACTIVE)
  4. Computes the transform payload  (anchor, delta, rotation)
  5. Returns a JSON-compatible dict matching the visHand spec

Primary API:
    interpreter = GestureInterpreter(settings)
    payload = interpreter.process(raw_landmarks_or_None, hand_side, timestamp, frame_id)

The filter is managed internally — feed raw MediaPipe landmarks directly.
"""

from __future__ import annotations

import time as _time
from typing import List, Optional, Tuple

from core.detector import Point3D
from core.filters import LandmarkFilter
from config.settings import Settings
import utils.math_tools as mt
from core.context import GestureContext
from core.gestures import registry


# ---------------------------------------------------------------------------
# String constants (avoid typos in comparisons)
# ---------------------------------------------------------------------------

LOCKED  = "LOCKED"
HOVER   = "HOVER"
ACTIVE  = "ACTIVE"

IDLE          = "IDLE"
EV_NONE        = "NONE"
EV_SNAP        = "SNAP"
EV_DOUBLE_TAP  = "DOUBLE_TAP"
EV_CLAP        = "CLAP"


# ---------------------------------------------------------------------------
# GestureInterpreter
# ---------------------------------------------------------------------------

class GestureInterpreter:
    """
    Stateful gesture interpreter.  One instance per tracked hand.

    Args:
        settings (Settings): project-wide configuration.
    """

    def __init__(self, settings: Settings):
        self.s = settings
        self._filter = LandmarkFilter(settings)
        self._reset()

    # ── Main entry point ─────────────────────────────────────────────────────

    def process(
        self,
        raw_landmarks: Optional[List[Point3D]],
        hand_side: str,
        timestamp: float,
        frame_id: int,
    ) -> dict:
        """
        Full pipeline: filter → classify → state machine → payload.

        Args:
            raw_landmarks:  21-point list from MediaPipe (un-filtered), or None.
            hand_side:      "LEFT" | "RIGHT"
            timestamp:      Unix timestamp in seconds (float)
            frame_id:       Monotonically increasing frame counter

        Returns:
            JSON-compatible dict matching the visHand spec.
        """
        if raw_landmarks is None:
            return self._handle_no_hand(hand_side, timestamp, frame_id)

        # Apply 1€ filter
        lm = self._filter.apply(raw_landmarks, timestamp)

        self._no_hand_frames = 0
        self._has_hand_frames += 1

        # Build Context for Evaluators
        ctx = GestureContext(lm, self.s, self._anchor_history)

        # --- Pipeline ---
        event  = self._detect_events(lm, timestamp)
        intent = self._classify_intent(ctx)
        logic  = self._update_logic(intent)

        anchor, delta, rotation = self._compute_transform(lm, intent)
        vel       = mt.instant_velocity(self._anchor_history, self.s.anchor_history_size)
        intensity = mt.pinch_intensity(lm)

        self._logic  = logic
        self._intent = intent

        return self._build_payload(
            frame_id, timestamp, hand_side,
            logic, intent, self._filter.is_stable,
            anchor, delta, rotation,
            intensity, vel, event,
        )

    # ── Internal state helpers ───────────────────────────────────────────────

    def _reset(self):
        """Full state reset — called at construction and when hand is lost."""
        self._logic  = LOCKED
        self._intent = IDLE
        self._intent_buffer: List[str] = []

        self._no_hand_frames  = 0
        self._has_hand_frames = 0

        # Transform
        self._prev_anchor: Optional[Tuple[float, float]] = None
        self._anchor_history: List[Tuple[float, float]] = []

        # SNAP two-phase machine
        self._snap_phase: str = "IDLE"        # "IDLE" | "READY"
        self._prev_mid_thumb_dist: Optional[float] = None
        self._prev_idx_thumb_dist: Optional[float] = None
        self._prev_mid_palm_dist: Optional[float] = None
        self._last_snap_time: float = 0.0

        # DOUBLE_TAP
        self._pinch_was_active = False
        self._pinch_onset_times: List[float] = []

        self._filter.reset()

    # ── No-hand path ──────────────────────────────────────────────────────────

    def _handle_no_hand(self, hand_side, timestamp, frame_id) -> dict:
        self._no_hand_frames  += 1
        self._has_hand_frames  = 0

        if self._no_hand_frames >= self.s.hover_to_locked_frames:
            self._reset()   # full reset: LOCKED + filter cleared

        return self._build_payload(
            frame_id, timestamp, hand_side,
            self._logic, IDLE, False,
            (0.0, 0.0, 0.0), (0.0, 0.0), 0.0,
            0.0, 0.0, EV_NONE,
        )

    # ── Event detection ───────────────────────────────────────────────────────

    def _detect_events(self, lm: List[Point3D], timestamp: float) -> str:
        s = self.s
        now_ms_ok = (timestamp - self._last_snap_time) > (s.snap_cooldown_ms / 1000.0)

        # ── SNAP (two-phase composite scoring) ──────────────────────────────
        mid_thumb = mt.normalized_distance(lm[mt.MIDDLE_TIP], lm[mt.THUMB_TIP], lm)
        idx_thumb = mt.normalized_distance(lm[mt.INDEX_TIP], lm[mt.THUMB_TIP], lm)
        mid_palm  = mt.normalized_distance(lm[mt.MIDDLE_TIP], lm[0], lm) # 0 is WRIST

        if self._snap_phase == "IDLE":
            if mid_thumb < s.snap_ready_threshold:
                self._snap_phase = "READY"

        elif self._snap_phase == "READY":
            if mid_thumb > s.snap_trigger_threshold:
                # Separation happened — check composite speeds
                if self._prev_mid_thumb_dist is not None:
                    sep_vel = mid_thumb - self._prev_mid_thumb_dist                # Expect > 0
                    idx_crash_vel = idx_thumb - self._prev_idx_thumb_dist          # Expect < 0
                    palm_crash_vel = mid_palm - self._prev_mid_palm_dist           # Expect < 0
                    
                    score = 0.0
                    if sep_vel > 0:       score += sep_vel * 10.0
                    if idx_crash_vel < 0: score += (-idx_crash_vel) * 10.0
                    if palm_crash_vel < 0: score += (-palm_crash_vel) * 10.0
                    
                    if score > s.snap_score_threshold and now_ms_ok:
                        self._last_snap_time = timestamp
                        self._snap_phase = "IDLE"
                        self._prev_mid_thumb_dist = mid_thumb
                        self._prev_idx_thumb_dist = idx_thumb
                        self._prev_mid_palm_dist = mid_palm
                        return EV_SNAP
                        
                self._snap_phase = "IDLE"
            elif mid_thumb > s.snap_ready_threshold * 1.6:
                # Drifted back without a snap
                self._snap_phase = "IDLE"

        self._prev_mid_thumb_dist = mid_thumb
        self._prev_idx_thumb_dist = idx_thumb
        self._prev_mid_palm_dist = mid_palm

        # ── DOUBLE_TAP ──────────────────────────────────────────────────────
        pinch_norm = mt.normalized_distance(lm[mt.THUMB_TIP], lm[mt.INDEX_TIP], lm)
        is_pinch   = pinch_norm < s.pinch_threshold

        if is_pinch and not self._pinch_was_active:
            self._pinch_onset_times.append(timestamp)
            cutoff = timestamp - s.double_tap_window_ms / 1000.0
            self._pinch_onset_times = [t for t in self._pinch_onset_times if t > cutoff]

            if len(self._pinch_onset_times) >= 2:
                self._pinch_onset_times.clear()
                self._pinch_was_active = True
                return EV_DOUBLE_TAP

        self._pinch_was_active = is_pinch
        return EV_NONE

    # ── Intent classification ─────────────────────────────────────────────────

    def _classify_intent(self, ctx: GestureContext) -> str:
        s = self.s

        # SNAP_READY has visual priority — let the user see the ready state
        if self._snap_phase == "READY":
            return "SNAP_READY"

        best_intent = IDLE
        best_score = 0.0

        for g in registry.enabled_intents():
            score = g.evaluator(ctx)
            if score > best_score:
                best_score = score
                best_intent = g.name

        if best_score < s.intent_min_confidence:
            raw_intent = IDLE
        else:
            raw_intent = best_intent

        # Check for swipe (fast horizontal motion while OPEN_PALM)
        if raw_intent == "OPEN_PALM":
            vel = mt.instant_velocity(self._anchor_history, 5)
            if vel > s.swipe_velocity_threshold and len(self._anchor_history) >= 3:
                dx = self._anchor_history[-1][0] - self._anchor_history[-3][0]
                dy = self._anchor_history[-1][1] - self._anchor_history[-3][1]
                if abs(dx) > abs(dy):  # horizontal motion dominates
                    raw_intent = "SWIPE_LEFT" if dx < 0 else "SWIPE_RIGHT"

        # --- Debounce Logic ---
        self._intent_buffer.append(raw_intent)
        if len(self._intent_buffer) > s.intent_debounce_frames:
            self._intent_buffer.pop(0)

        # ONLY switch if the new intent holds for N frames, otherwise keep previous
        # We check if all items in buffer are the same 
        if all(i == raw_intent for i in self._intent_buffer):
            return raw_intent
            
        return self._intent

    # ── Logic state machine ───────────────────────────────────────────────────

    def _update_logic(self, intent: str) -> str:
        s = self.s
        current = self._logic

        if current == LOCKED:
            if self._has_hand_frames >= s.locked_to_hover_frames:
                return HOVER
            return LOCKED

        if current == HOVER:
            if intent in ("PINCH_HOLD", "PINCH_DRAG", "CLOSED_FIST"):
                return ACTIVE
            return HOVER

        if current == ACTIVE:
            if intent in (IDLE, "OPEN_PALM"):
                return HOVER
            return ACTIVE

        return current

    # ── Transform computation ─────────────────────────────────────────────────

    def _compute_transform(
        self, lm: List[Point3D], intent: str
    ) -> Tuple[Tuple, Tuple, float]:

        # Choose anchor based on gesture type
        if intent in (PINCH_HOLD, PINCH_DRAG, SNAP_READY):
            anchor = mt.pinch_anchor(lm)
        else:
            anchor = mt.palm_anchor(lm)

        ax, ay = anchor[0], anchor[1]

        # Maintain anchor history for velocity calculation
        self._anchor_history.append((ax, ay))
        if len(self._anchor_history) > self.s.anchor_history_size:
            self._anchor_history.pop(0)

        delta    = mt.compute_delta(self._prev_anchor, (ax, ay))
        self._prev_anchor = (ax, ay)

        rotation = mt.palm_roll_angle(lm)

        return anchor, delta, rotation

    # ── Payload builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_payload(
        frame_id, timestamp, hand_side,
        logic, intent, is_stable,
        anchor, delta, rotation,
        intensity, velocity, event,
    ) -> dict:
        return {
            "header": {
                "frame_id":  frame_id,
                "timestamp": round(timestamp, 3),
                "hand_side": hand_side,
            },
            "state": {
                "logic":     logic,
                "intent":    intent,
                "is_stable": is_stable,
            },
            "transform": {
                "anchor": {
                    "x": round(anchor[0], 4),
                    "y": round(anchor[1], 4),
                    "z": round(anchor[2], 4),
                },
                "delta": {
                    "dx": round(delta[0], 5),
                    "dy": round(delta[1], 5),
                },
                "rotation": round(rotation, 2),
            },
            "dynamics": {
                "intensity": round(intensity, 3),
                "velocity":  round(velocity, 5),
                "event":     event,
            },
        }
