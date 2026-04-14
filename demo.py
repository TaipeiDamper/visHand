"""
visHand Demo
=============
Live camera window with hand skeleton + state overlay.

Shows:
  - Colour-coded hand skeleton  (grey=LOCKED, amber=HOVER, green=ACTIVE)
  - Info panel: logic state / intent / stability / metrics
  - Fading event banner for SNAP, DOUBLE_TAP, etc.
  - Border colour matching logic state
  - FPS counter

Usage:
    python demo.py                  # default webcam (index 0)
    python demo.py --camera 1       # external webcam
    python demo.py --max-hands 2    # (future) track both hands

Keyboard:
    ESC or Q  — quit
"""

import sys
import os
import time
import json
import argparse

import cv2

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from core.detector import HandDetector
from core.interpreter import GestureInterpreter
from utils.visualizer import DebugVisualizer
from utils.minigame import Minigame


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="visHand — live debug demo")
    p.add_argument("--camera",    type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--max-hands", type=int, default=2, choices=[1, 2],
                   help="Max hands to track (default: 2)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    settings = Settings(
        camera_index=args.camera,
        max_hands=args.max_hands,
        debug_window=True,
    )

    print("[visHand Demo] Starting…  Press ESC or Q to quit.\n")

    try:
        detector     = HandDetector(settings)
        interpreters = [GestureInterpreter(settings) for _ in range(settings.max_hands)]
        visualizer   = DebugVisualizer()
        minigame     = Minigame()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    frame_id   = 0
    fps_buffer = []
    t_prev     = time.perf_counter()
    last_payload = None  # keep last valid payload for display

    try:
        while True:
            # ── Timing ───────────────────────────────────────────────────────
            t_now = time.perf_counter()
            dt    = t_now - t_prev
            t_prev = t_now
            fps_buffer.append(1.0 / max(dt, 1e-6))
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            fps = sum(fps_buffer) / len(fps_buffer)

            timestamp = time.time()

            # ── Capture ───────────────────────────────────────────────────────
            ok, frame = detector.read_frame()
            if not ok:
                print("[visHand Demo] Camera read failed — exiting.")
                break

            # ── Detection ─────────────────────────────────────────────────────
            results = detector.extract_landmarks(frame)

            payloads_this_frame = []
            raw_lms_this_frame = []

            if results:
                # Process each detected hand (up to max_hands)
                for i, result in enumerate(results[: settings.max_hands]):
                    interp   = interpreters[i]
                    payload  = interp.process(
                        result.landmarks, result.hand_side, timestamp, frame_id
                    )
                    raw_lm   = result.raw_landmarks

                    payloads_this_frame.append(payload)
                    raw_lms_this_frame.append(raw_lm)

                    # Print to stdout when something interesting happens
                    ev = payload["dynamics"]["event"]
                    it = payload["state"]["intent"]
                    if ev != "NONE" or it not in ("IDLE", "OPEN_PALM"):
                        print(json.dumps(payload, ensure_ascii=False))

            else:
                # No hand — feed None to keep state machine updated
                payload = interpreters[0].process(None, "RIGHT", timestamp, frame_id)
                payloads_this_frame = [payload]
                raw_lms_this_frame = [None]

            # ── Minigame & Draw ───────────────────────────────────────────────
            vis = visualizer.draw(
                frame, payloads_this_frame, raw_lms_this_frame, timestamp, fps
            )
            
            # Draw minigame overlay on top
            vis = minigame.update(vis, payloads_this_frame, raw_lms_this_frame, timestamp)
            
            visualizer.show(vis)

            frame_id += 1

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):   # ESC or Q
                break
                
            # Handle user closing the window with the 'X' button
            try:
                if cv2.getWindowProperty(visualizer.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

    except KeyboardInterrupt:
        print("\n[visHand Demo] Interrupted.")
    finally:
        detector.release()
        visualizer.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # ensure windows close properly
        print("[visHand Demo] Closed.")


if __name__ == "__main__":
    main()
