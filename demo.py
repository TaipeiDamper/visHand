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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="visHand — live debug demo")
    p.add_argument("--camera",    type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--max-hands", type=int, default=1, choices=[1, 2],
                   help="Max hands to track (default: 1)")
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

            if results:
                # Process each detected hand (up to max_hands)
                for i, result in enumerate(results[: settings.max_hands]):
                    interp   = interpreters[i]
                    payload  = interp.process(
                        result.landmarks, result.hand_side, timestamp, frame_id
                    )
                    raw_lm   = result.raw_landmarks

                    # Print to stdout when something interesting happens
                    ev = payload["dynamics"]["event"]
                    it = payload["state"]["intent"]
                    if ev != "NONE" or it not in ("IDLE", "OPEN_PALM"):
                        print(json.dumps(payload, ensure_ascii=False))

                    # Only draw overlay for the primary hand
                    if i == 0:
                        last_payload = payload
                        primary_raw  = raw_lm
            else:
                # No hand — feed None to keep state machine updated
                payload = interpreters[0].process(None, "RIGHT", timestamp, frame_id)
                if last_payload is None:
                    last_payload = payload
                primary_raw = None

            # ── Draw ──────────────────────────────────────────────────────────
            vis = visualizer.draw(
                frame, last_payload, primary_raw, timestamp, fps
            )
            visualizer.show(vis)

            frame_id += 1

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):   # ESC or Q
                break

    except KeyboardInterrupt:
        print("\n[visHand Demo] Interrupted.")
    finally:
        detector.release()
        visualizer.close()
        cv2.destroyAllWindows()
        print("[visHand Demo] Closed.")


if __name__ == "__main__":
    main()
