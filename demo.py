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
    P         — dump telemetry snapshot json
    B         — dump baseline to scratch/bench_baseline.json
    R         — reset telemetry buffer
    K         — print robustness KPI summary
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

import cv2

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bridge.factory import create_transport
from config.settings import Settings
from core.detector import HandDetector
from core.interpreter import GestureInterpreter
from utils.visualizer import DebugVisualizer
from utils.minigame import Minigame
from utils.profiler import RuntimeTelemetry


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

    fps_buffer = []
    t_prev     = time.perf_counter()
    telemetry  = RuntimeTelemetry(max_frames=settings.telemetry_buffer_size)
    transport = create_transport(settings)
    last_frame_id = -1
    no_hand_since = None
    bench_dir = Path("scratch")

    try:
        detector.start()
        while True:
            packet = detector.get_latest_packet(min_frame_id=last_frame_id)
            if packet is None:
                time.sleep(0.001)
                continue
            last_frame_id = packet.frame_id
            timestamp = packet.t_infer_done
            frame = packet.frame

            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            fps_buffer.append(1.0 / max(dt, 1e-6))
            if len(fps_buffer) > 30:
                fps_buffer.pop(0)
            fps = sum(fps_buffer) / len(fps_buffer)

            if frame is None:
                break

            # ── Detection ─────────────────────────────────────────────────────
            results = packet.results

            payloads_this_frame = []
            raw_lms_this_frame = []
            filter_t0 = time.perf_counter()

            if results:
                # Process each detected hand (up to max_hands)
                for i, result in enumerate(results[: settings.max_hands]):
                    interp   = interpreters[i]
                    payload  = interp.process(
                        result.landmarks,
                        result.hand_side,
                        timestamp,
                        packet.frame_id,
                        hand_id=result.track_id,
                        arm_features=packet.arm_features,
                        degraded_mode=packet.degraded_mode,
                        input_quality_score=packet.input_quality_score,
                        tracking_quality=packet.tracking_quality,
                    )
                    raw_lm   = result.raw_landmarks

                    payloads_this_frame.append(payload)
                    raw_lms_this_frame.append(raw_lm)

                    ev = payload["dynamics"]["event"]
                    should_emit_state = settings.output_mode == "stdout" or settings.bridge_transport in ("ws", "pipe")
                    if should_emit_state:
                        transport.send_state(payload)
                    if ev != "NONE":
                        transport.send_event(ev, payload)

            else:
                # No hand — feed None to keep state machine updated
                payload = interpreters[0].process(
                    None,
                    "RIGHT",
                    timestamp,
                    packet.frame_id,
                    hand_id="slot-0",
                    arm_features=packet.arm_features,
                    degraded_mode=packet.degraded_mode,
                    input_quality_score=packet.input_quality_score,
                    tracking_quality=packet.tracking_quality,
                )
                payloads_this_frame = [payload]
                raw_lms_this_frame = [None]
            filter_ms = (time.perf_counter() - filter_t0) * 1000.0

            # ── Minigame & Draw ───────────────────────────────────────────────
            render_t0 = time.perf_counter()
            vis = visualizer.draw(
                frame, payloads_this_frame, raw_lms_this_frame, timestamp, fps
            )
            
            # Draw minigame overlay on top
            vis = minigame.update(vis, payloads_this_frame, raw_lms_this_frame, timestamp)
            
            visualizer.show(vis)
            render_ms = (time.perf_counter() - render_t0) * 1000.0

            main_payload = payloads_this_frame[0] if payloads_this_frame else None
            if main_payload:
                detector.update_runtime_hint(
                    logic=main_payload["state"]["logic"],
                    intent=main_payload["state"]["intent"],
                    velocity=main_payload["dynamics"]["velocity"],
                )

            has_hand = bool(results)
            reacquire_time_ms = 0.0
            if not has_hand:
                if no_hand_since is None:
                    no_hand_since = time.time()
            elif no_hand_since is not None:
                reacquire_time_ms = max(0.0, (time.time() - no_hand_since) * 1000.0)
                no_hand_since = None

            telemetry.add({
                "frame_id": packet.frame_id,
                "capture_ms": packet.capture_ms,
                "mp_ms": packet.mp_ms,
                "pose_ms": packet.pose_ms,
                "filter_ms": filter_ms,
                "render_ms": render_ms,
                "e2e_latency_ms": max(0.0, (time.time() - packet.t_capture) * 1000.0),
                "infer_mode": packet.mode,
                "skipped": int(packet.skipped),
                "arm_assist_applied": int(main_payload["state"].get("arm_assist_applied", 0)) if main_payload else 0,
                "event": main_payload["dynamics"].get("event", "NONE") if main_payload else "NONE",
                "intent": main_payload["state"].get("intent", "IDLE") if main_payload else "IDLE",
                "tracking_quality": packet.tracking_quality,
                "input_quality_score": packet.input_quality_score,
                "reacquire_time_ms": reacquire_time_ms,
                "false_event_flag": int(
                    (main_payload["dynamics"].get("event", "NONE") != "NONE")
                    and (packet.tracking_quality != "good")
                ) if main_payload else 0,
            })

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):   # ESC or Q
                break
            if key in (ord("p"), ord("P")):
                bench_dir.mkdir(parents=True, exist_ok=True)
                if telemetry.size > 0:
                    out_file = bench_dir / f"bench_runtime_{int(time.time())}.json"
                    path = telemetry.dump_json(str(out_file))
                    print(f"[visHand Demo] Telemetry dumped: {path}")
                    print(json.dumps(telemetry.summary(), ensure_ascii=False, indent=2))
            if key in (ord("b"), ord("B")):
                bench_dir.mkdir(parents=True, exist_ok=True)
                baseline_path = bench_dir / "bench_baseline.json"
                if telemetry.size > 0:
                    path = telemetry.dump_json(str(baseline_path))
                    print(f"[visHand Demo] Baseline dumped: {path}")
                else:
                    print("[visHand Demo] Baseline not dumped (telemetry is empty).")
            if key in (ord("r"), ord("R")):
                telemetry.reset()
                print("[visHand Demo] Telemetry reset.")
            if key in (ord("k"), ord("K")):
                print(json.dumps(telemetry.robustness_summary(), ensure_ascii=False, indent=2))
                
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
        transport.close()
        visualizer.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # ensure windows close properly
        print("[visHand Demo] Closed.")


if __name__ == "__main__":
    main()
