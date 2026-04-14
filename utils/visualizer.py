"""
visHand — Debug Visualizer
===========================
OpenCV-based debug overlay.  Draws:
  - Hand skeleton (colour-coded by logic state)
  - SNAP_READY highlight on thumb + middle finger tips
  - State / intent / event text panel
  - Fading "last event" banner
  - Coloured border matching current logic state
  - Anchor crosshair
  - FPS counter (caller supplies value)

Usage:
    vis = DebugVisualizer()
    annotated = vis.draw(frame, payload, raw_lm, timestamp)
    vis.show(annotated)
"""

from __future__ import annotations

import time as _time
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp


# ── Drawing utilities from MediaPipe ─────────────────────────────────────────

_mp_drawing = mp.solutions.drawing_utils
_mp_hands   = mp.solutions.hands

# ── Colour palette (BGR) ─────────────────────────────────────────────────────

_CLR = {
    "LOCKED":     (110, 110, 110),   # grey
    "HOVER":      (220, 170,  40),   # amber
    "ACTIVE":     ( 60, 220,  90),   # green
    "SNAP_READY": ( 40, 120, 255),   # orange-red
    "EVENT":      ( 40,  80, 255),   # vivid red
    "WHITE":      (240, 240, 240),
    "BLACK":      (  0,   0,   0),
    "DIM":        (160, 160, 160),
}

# Intent labels shown in the overlay (plain ASCII — OpenCV doesn't render emoji)
_INTENT_LABELS = {
    "IDLE":        "IDLE",
    "POINT":       "POINT          [1 finger]",
    "PINCH_HOLD":  "PINCH HOLD     [grip still]",
    "PINCH_DRAG":  "PINCH DRAG     [grip move]",
    "FIST":        "FIST           [all curl]",
    "OPEN_PALM":   "OPEN PALM      [all open]",
    "SNAP_READY":  "** SNAP READY **",
    "SWIPE_LEFT":  "SWIPE LEFT  <--",
    "SWIPE_RIGHT": "SWIPE RIGHT -->",
    "PALM_ROTATE": "PALM ROTATE",
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# DebugVisualizer
# ---------------------------------------------------------------------------

class DebugVisualizer:
    """
    Renders a rich debug overlay onto a copy of the video frame.

    Args:
        window_name: Title bar of the OpenCV window.
        event_linger_sec: How long the last event text stays visible.
    """

    def __init__(self, window_name: str = "visHand  |  Debug View",
                 event_linger_sec: float = 1.8):
        self.window_name       = window_name
        self._event_linger     = event_linger_sec
        self._last_event       = "NONE"
        self._last_event_time  = 0.0

    # ── Public API ───────────────────────────────────────────────────────────

    def draw(
        self,
        frame:        np.ndarray,
        payload:      dict,
        raw_lm=None,                   # mediapipe NormalizedLandmarkList or None
        timestamp:    float = 0.0,
        fps:          float = 0.0,
    ) -> np.ndarray:
        """
        Annotate the frame with hand tracking debug info.

        Args:
            frame:     BGR frame (will NOT be modified — a copy is returned).
            payload:   dict from GestureInterpreter.process()
            raw_lm:    MediaPipe raw landmark list for skeleton drawing.
            timestamp: Current Unix timestamp (for event fade calculation).
            fps:       Caller-computed FPS to display.

        Returns:
            Annotated BGR frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        now  = timestamp if timestamp else _time.time()

        logic   = payload["state"]["logic"]
        intent  = payload["state"]["intent"]
        stable  = payload["state"]["is_stable"]
        event   = payload["dynamics"]["event"]
        vel     = payload["dynamics"]["velocity"]
        anchor  = payload["transform"]["anchor"]
        rotation = payload["transform"]["rotation"]
        intensity = payload["dynamics"]["intensity"]

        state_color = _CLR.get(logic, _CLR["WHITE"])

        # ── 1. Hand skeleton ──────────────────────────────────────────────────
        if raw_lm is not None:
            if logic == "ACTIVE":
                lm_spec   = _mp_drawing.DrawingSpec(color=_CLR["ACTIVE"],  thickness=3, circle_radius=4)
                conn_spec = _mp_drawing.DrawingSpec(color=(30, 180, 60),   thickness=2)
            elif logic == "HOVER":
                lm_spec   = _mp_drawing.DrawingSpec(color=_CLR["HOVER"],   thickness=3, circle_radius=4)
                conn_spec = _mp_drawing.DrawingSpec(color=(160, 120, 20),  thickness=2)
            else:
                lm_spec   = _mp_drawing.DrawingSpec(color=_CLR["LOCKED"],  thickness=2, circle_radius=3)
                conn_spec = _mp_drawing.DrawingSpec(color=(80, 80, 80),    thickness=1)

            _mp_drawing.draw_landmarks(
                vis, raw_lm, _mp_hands.HAND_CONNECTIONS, lm_spec, conn_spec
            )

        # ── 2. SNAP_READY highlight ───────────────────────────────────────────
        if intent == "SNAP_READY" and raw_lm is not None:
            for lm_id in (4, 12):   # THUMB_TIP, MIDDLE_TIP
                lm = raw_lm.landmark[lm_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(vis, (cx, cy), 14, _CLR["SNAP_READY"], -1)
                cv2.circle(vis, (cx, cy), 16, _CLR["WHITE"],      2)

        # ── 3. Anchor crosshair ───────────────────────────────────────────────
        ax, ay = int(anchor["x"] * w), int(anchor["y"] * h)
        if 0 < ax < w and 0 < ay < h:
            cv2.drawMarker(vis, (ax, ay), state_color, cv2.MARKER_CROSS, 22, 2)

        # ── 4. Coloured border (logic state) ─────────────────────────────────
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), state_color, 3)

        # ── 5. Info panel (top-left) ─────────────────────────────────────────
        #  Semi-transparent dark background
        panel_w, panel_h = 260, 160
        overlay = vis.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

        # Logic state (large, coloured)
        self._text(vis, logic, (18, 42), state_color, scale=0.95, bold=True)

        # Intent
        intent_label = _INTENT_LABELS.get(intent, intent)
        self._text(vis, intent_label, (18, 72), _CLR["WHITE"], scale=0.52)

        # Stable indicator
        stable_col  = (60, 200, 80) if stable else (100, 100, 100)
        stable_txt  = "STABLE" if stable else "WARMUP"
        self._text(vis, stable_txt, (18, 96), stable_col, scale=0.48)

        # Metrics
        self._text(vis, f"vel:{vel:.4f}  int:{intensity:.2f}  rot:{rotation:.1f}",
                   (18, 116), _CLR["DIM"], scale=0.42)

        # FPS
        if fps > 0:
            self._text(vis, f"FPS {fps:.1f}", (18, 140), _CLR["DIM"], scale=0.45)

        # ── 6. Event fading banner (bottom-centre) ────────────────────────────
        if event != EV_NONE:
            self._last_event      = event
            self._last_event_time = now

        age = now - self._last_event_time
        if age < self._event_linger and self._last_event != "NONE":
            alpha = max(0.0, 1.0 - age / self._event_linger)
            ev_col = tuple(int(c * alpha) for c in _CLR["EVENT"])

            ev_text  = f">> {self._last_event} <<"
            txt_size = cv2.getTextSize(ev_text, _FONT, 0.9, 2)[0]
            tx = (w - txt_size[0]) // 2
            ty = h - 30

            # Shadow
            cv2.putText(vis, ev_text, (tx + 2, ty + 2), _FONT, 0.9,
                        _CLR["BLACK"], 3, cv2.LINE_AA)
            cv2.putText(vis, ev_text, (tx, ty), _FONT, 0.9,
                        ev_col, 2, cv2.LINE_AA)

        return vis

    def show(self, frame: np.ndarray):
        """Display the annotated frame in the debug window."""
        cv2.imshow(self.window_name, frame)

    def close(self):
        """Close the debug window."""
        cv2.destroyWindow(self.window_name)

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _text(frame, text, pos, color, scale=0.65, bold=False):
        thickness = 2 if bold else 1
        # Drop-shadow
        cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                    _FONT, scale, _CLR["BLACK"], thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, pos,
                    _FONT, scale, color, thickness, cv2.LINE_AA)


# Make the constant available to callers that import from this module
EV_NONE = "NONE"
