"""
visHand — Gesture Calibration Tool
==================================
Record live gesture outputs with manual labels for threshold tuning.

Keys:
  Q / ESC  : quit
  SPACE    : start/stop recording
  E        : export current session buffer to JSONL
  C        : clear buffer
  0        : label NONE
  1        : OPEN_PALM
  2        : CLOSED_FIST
  3        : PINCH_HOLD
  4        : PINCH_DRAG
  5        : OK_SIGN
  6        : INDEX_POINT_LEFT
  7        : INDEX_POINT_RIGHT
  8        : PALM_TILT_LEFT
  9        : PALM_TILT_RIGHT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from config.calibration_profile import build_profile_from_jsonl_rows
from core.detector import HandDetector
from core.interpreter import GestureInterpreter
from utils.visualizer import DebugVisualizer


LABEL_KEYS: Dict[int, str] = {
    ord("0"): "NONE",
    ord("1"): "OPEN_PALM",
    ord("2"): "CLOSED_FIST",
    ord("3"): "PINCH_HOLD",
    ord("4"): "PINCH_DRAG",
    ord("5"): "OK_SIGN",
    ord("6"): "INDEX_POINT_LEFT",
    ord("7"): "INDEX_POINT_RIGHT",
    ord("8"): "PALM_TILT_LEFT",
    ord("9"): "PALM_TILT_RIGHT",
}


def parse_args():
    parser = argparse.ArgumentParser(description="visHand gesture calibration recorder")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--max-hands", type=int, default=1, choices=[1, 2], help="max tracked hands")
    parser.add_argument("--export-profile", default="", help="optional profile output path when pressing E")
    return parser.parse_args()


def _write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    settings = Settings(camera_index=args.camera, max_hands=args.max_hands, debug_window=True)

    detector = HandDetector(settings)
    interpreters = [GestureInterpreter(settings) for _ in range(settings.max_hands)]
    visualizer = DebugVisualizer(window_name="visHand | Calibration")

    recording = False
    current_label = "NONE"
    rows: List[dict] = []
    last_frame_id = -1
    output_dir = Path("scratch") / "calibration"
    session_id = f"session_{int(time.time())}"
    print("[Calibration] Ready. SPACE to record, E to export.")

    try:
        detector.start()
        while True:
            packet = detector.get_latest_packet(min_frame_id=last_frame_id)
            if packet is None:
                time.sleep(0.001)
                continue
            last_frame_id = packet.frame_id

            payloads = []
            raw_lms = []
            results = packet.results or []
            timestamp = packet.t_infer_done

            for i, result in enumerate(results[: settings.max_hands]):
                payload = interpreters[i].process(
                    result.landmarks,
                    result.hand_side,
                    timestamp,
                    packet.frame_id,
                    arm_features=packet.arm_features,
                )
                payloads.append(payload)
                raw_lms.append(result.raw_landmarks)

            if not payloads:
                payload = interpreters[0].process(
                    None, "RIGHT", timestamp, packet.frame_id, arm_features=packet.arm_features
                )
                payloads = [payload]
                raw_lms = [None]

            vis = visualizer.draw(packet.frame, payloads, raw_lms, timestamp=timestamp, fps=0.0)
            cv2.putText(vis, f"REC: {'ON' if recording else 'OFF'}", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.putText(vis, f"LABEL: {current_label}", (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
            cv2.putText(vis, f"ROWS: {len(rows)}", (14, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
            visualizer.show(vis)

            if recording:
                hand_landmarks = []
                if results:
                    hand_landmarks = [
                        {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
                        for p in results[0].landmarks
                    ]
                pred = payloads[0]
                rows.append(
                    {
                        "session_id": session_id,
                        "timestamp": round(timestamp, 6),
                        "frame_id": packet.frame_id,
                        "manual_label": current_label,
                        "pred_intent": pred["state"]["intent"],
                        "pred_event": pred["dynamics"]["event"],
                        "pred_confidence": pred["state"].get("intent_confidence", 0.0),
                        "pred_risk": pred["state"].get("intent_risk", "low"),
                        "arbitration": pred["state"].get("arbitration", []),
                        "arm_assist_applied": pred["state"].get("arm_assist_applied", 0),
                        "handedness_fused": pred["state"].get("handedness_fused", pred["header"]["hand_side"]),
                        "z_hint": pred["state"].get("z_hint", "neutral"),
                        "arm_confidence": pred["state"].get("arm_confidence", 0.0),
                        "forearm_vector": (packet.arm_features or {}).get(pred["header"]["hand_side"].lower(), {}).get("forearm_vector"),
                        "infer_mode": packet.mode,
                        "pose_ms": packet.pose_ms,
                        "landmarks": hand_landmarks,
                        "settings": asdict(settings),
                    }
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                recording = not recording
                print(f"[Calibration] Recording {'ON' if recording else 'OFF'}")
            if key in (ord("c"), ord("C")):
                rows.clear()
                print("[Calibration] Buffer cleared.")
            if key in (ord("e"), ord("E")):
                out = output_dir / f"{session_id}_{int(time.time())}.jsonl"
                _write_jsonl(out, rows)
                print(f"[Calibration] Exported {len(rows)} rows -> {out}")
                if str(args.export_profile).strip():
                    profile = build_profile_from_jsonl_rows(rows)
                    profile_path = Path(args.export_profile)
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    with profile_path.open("w", encoding="utf-8") as f:
                        json.dump(profile, f, ensure_ascii=False, indent=2)
                    print(f"[Calibration] Profile exported -> {profile_path}")
            if key in LABEL_KEYS:
                current_label = LABEL_KEYS[key]
                print(f"[Calibration] Label set -> {current_label}")

    finally:
        detector.release()
        visualizer.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
