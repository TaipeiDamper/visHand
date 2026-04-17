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
    annotated = vis.draw(frame, payloads, raw_lms, timestamp)
    vis.show(annotated)
"""

from __future__ import annotations

import time as _time
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from core.gestures import registry


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
        payloads:     list,
        raw_lms:      list,
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
        
        # We will use the first hand's payload for the main panel and border color
        main_payload = payloads[0] if payloads else None
        
        state_color = _CLR["WHITE"]
        logic = "LOCKED"
        intent = "IDLE"
        stable = False
        event = EV_NONE
        vel, intensity, rotation = 0.0, 0.0, 0.0
        
        if main_payload:
            logic   = main_payload["state"]["logic"]
            intent  = main_payload["state"]["intent"]
            stable  = main_payload["state"]["is_stable"]
            event   = main_payload["dynamics"]["event"]
            vel     = main_payload["dynamics"]["velocity"]
            rotation = main_payload["transform"]["rotation"]
            intensity = main_payload["dynamics"]["intensity"]
            state_color = _CLR.get(logic, _CLR["WHITE"])
            
            # Check if any hand has an event to display
            for p in payloads:
                if p and p["dynamics"]["event"] != EV_NONE:
                    event = p["dynamics"]["event"]
                    break

        # ── 1-3. Skeletons & Anchors for all hands ────────────────────────────
        for i, raw_lm in enumerate(raw_lms):
            if not raw_lm: continue
            
            p_logic = payloads[i]["state"]["logic"] if i < len(payloads) else "LOCKED"
            p_intent = payloads[i]["state"]["intent"] if i < len(payloads) else "IDLE"
            p_color = _CLR.get(p_logic, _CLR["WHITE"])

            if p_logic == "ACTIVE":
                lm_spec   = _mp_drawing.DrawingSpec(color=_CLR["ACTIVE"],  thickness=3, circle_radius=4)
                conn_spec = _mp_drawing.DrawingSpec(color=(30, 180, 60),   thickness=2)
            elif p_logic == "HOVER":
                lm_spec   = _mp_drawing.DrawingSpec(color=_CLR["HOVER"],   thickness=3, circle_radius=4)
                conn_spec = _mp_drawing.DrawingSpec(color=(160, 120, 20),  thickness=2)
            else:
                lm_spec   = _mp_drawing.DrawingSpec(color=_CLR["LOCKED"],  thickness=2, circle_radius=3)
                conn_spec = _mp_drawing.DrawingSpec(color=(80, 80, 80),    thickness=1)

            _mp_drawing.draw_landmarks(
                vis, raw_lm, _mp_hands.HAND_CONNECTIONS, lm_spec, conn_spec
            )

            # SNAP_READY highlight
            if p_intent == "SNAP_READY" and raw_lm is not None:
                for lm_id in (4, 12):   # THUMB_TIP, MIDDLE_TIP
                    lm = raw_lm.landmark[lm_id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(vis, (cx, cy), 14, _CLR["SNAP_READY"], -1)
                    cv2.circle(vis, (cx, cy), 16, _CLR["WHITE"],      2)

            # Anchor crosshair
            anchor = payloads[i]["transform"]["anchor"] if i < len(payloads) else None
            if anchor:
                ax, ay = int(anchor["x"] * w), int(anchor["y"] * h)
                if 0 < ax < w and 0 < ay < h:
                    cv2.drawMarker(vis, (ax, ay), p_color, cv2.MARKER_CROSS, 22, 2)

        # ── 4. Coloured border (logic state) ─────────────────────────────────
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), state_color, 3)

        # ── 5. Info panels (per hand) ─────────────────────────────────────────
        panel_w, panel_h = 260, 160
        overlay = vis.copy()
        
        # Draw dark backgrounds
        for i in range(len(payloads)):
            if not payloads[i]: continue
            px = 8 if i == 0 else w - panel_w - 8
            py = 8
            cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
            
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
        
        # Number mapping including directional variants and hard series
        _NUM_MAP = {
            "CLOSED_FIST": "0 (FIST)",
            "POINTING_UP": "1 (POINT)",
            "INDEX_POINT_LEFT": "1 (POINT)",
            "INDEX_POINT_RIGHT": "1 (POINT)",
            "VICTORY": "2 (VICTORY)",
            "NUMBER_3": "3",
            "NUMBER_4": "4",
            "OPEN_PALM": "5 (PALM)",
            "PALM_TILT_LEFT": "5 (PALM)",
            "PALM_TILT_RIGHT": "5 (PALM)",
            # Hard Series mapping
            "SNAP_PREP": "SNAP [PREP]",
            "SNAP_ACTION": "SNAP [FIRE!]",
            "CROSS_FINGERS_SINGLE": "CROSS [SINGLE]",
            "CROSS_FINGERS_MULTI": "CROSS [MULTI]",
            "CROSS_PALMS": "CROSS [PALMS]",
            "GRIP_FIST_IN_HAND": "GRIP [FIST]",
            "GRIP_INTERLOCKED": "GRIP [LOCK]",
            "PRAYER": "PRAYER",
            "CLAP_SLOW": "CLAP [SLOW]",
        }
        
        # Sort payloads horizontally so the left hand corresponds to left panel
        sorted_payloads = [p for p in payloads if p]
        sorted_payloads.sort(key=lambda p: float(p.get("transform", {}).get("anchor", {}).get("x", 0.5)))
        
        for i, p in enumerate(sorted_payloads):
            px = 8 if i == 0 else w - panel_w - 8
            
            p_logic = p["state"]["logic"]
            p_intent = p["state"]["intent"]
            p_stable = p["state"]["is_stable"]
            p_vel = p["dynamics"]["velocity"]
            p_intensity = p["dynamics"]["intensity"]
            p_rot = p["transform"]["rotation"]
            p_color = _CLR.get(p_logic, _CLR["WHITE"])
            
            # Logic state
            self._text(vis, p_logic, (px + 10, 42), p_color, scale=0.95, bold=True)
            
            # Intent mapping
            if p_intent in _NUM_MAP:
                intent_label = _NUM_MAP[p_intent]
            elif p_intent == "IDLE":
                intent_label = "IDLE"
            else:
                g_def = registry.get(p_intent)
                intent_label = f"{g_def.name}" if g_def else p_intent
                
            self._text(vis, intent_label, (px + 10, 72), _CLR["WHITE"], scale=0.52)
            
            # Stable indicator
            stable_col  = (60, 200, 80) if p_stable else (100, 100, 100)
            stable_txt  = "STABLE" if p_stable else "WARMUP"
            self._text(vis, stable_txt, (px + 10, 96), stable_col, scale=0.48)
            
            # Metrics
            self._text(vis, f"vel:{p_vel:.4f} int:{p_intensity:.2f} rot:{p_rot:.1f}",
                       (px + 10, 116), _CLR["DIM"], scale=0.42)
                       
            # Frame info (only on first panel)
            if i == 0 and fps > 0:
                self._text(vis, f"FPS {fps:.1f}", (px + 10, 140), _CLR["DIM"], scale=0.45)


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
        """Close the debug window safely."""
        try:
            # Only try to destroy if it likely exists
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(self.window_name)
        except:
            pass

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
