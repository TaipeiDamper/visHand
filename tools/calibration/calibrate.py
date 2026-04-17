"""
Calibration Tool for Gestures (Modular Version)
Supports Standard and Hard series presets.
"""
import sys
import os
import time
import json
import cv2
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import Settings
from core.detector import HandDetector

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "1": {
        "name": "Standard Series",
        "labels": ["0", "1", "2", "3", "4", "5"],
        "record_time": 10.0,
        "cooldown": 2.0,
        "guide": "standard_guide.txt"
    },
    "2": {
        "name": "Hard Series",
        "labels": [
            "SNAP_PREP", "SNAP_ACTION", 
            "CROSS_FINGERS_SINGLE", "CROSS_FINGERS_MULTI", "CROSS_PALMS",
            "GRIP_FIST_IN_HAND", "GRIP_INTERLOCKED",
            "PRAYER", "CLAP_SLOW"
        ],
        "record_time": 20.0,
        "cooldown": 5.0,
        "guide": "hard_guide.txt"
    },
    "3": {
        "name": "Isolation Series (L/R Fix)",
        "labels": [
            "SNAP_PREP", "SNAP_ACTION", 
            "NUMBER_5"
        ],
        "record_time": 20.0,
        "cooldown": 5.0,
        "guide": "isolation_guide.txt"
    },
    "4": {
        "name": "Snap Pro (Speed & Dynamics)",
        "labels": [
            "SNAP_PREP", "SNAP_ACTION"
        ],
        "record_time": 30.0,
        "cooldown": 10.0,
        "guide": "snap_pro_guide.txt"
    }
}

DATA_FILE = Path(__file__).parent / "data" / "dataset_ml.json"

def dump_landmark(lm_result):
    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in lm_result.landmarks]

def main():
    print("=== visHand Calibration Module ===")
    print("1: Standard Series (Numbers 0-5, 10s per pose, 2s gap)")
    print("2: Hard Series (Composite/Snaps, 20s per pose, 5s gap)")
    print("3: Isolation Series (Fix 1-Hand Snaps & 5, 20s per pose, 5s gap)")
    print("4: Snap Pro (Speed & Dynamics, 30s per pose, 10s gap)")
    choice = input("Select Series [1/2/3/4]: ").strip()
    
    config = PRESETS.get(choice)
    if not config:
        print("Invalid choice, exiting.")
        return

    print(f"\n[Selected] {config['name']}")
    guide_path = Path(__file__).parent / "instructions" / config["guide"]
    if guide_path.exists():
        print("--- Instructions ---")
        print(guide_path.read_text(encoding="utf-8"))
        print("--------------------\n")

    input("Press Enter to start 5s countdown...")

    # Force 2 hands for calibration
    settings = Settings(max_hands=2)
    detector = HandDetector(settings)
    detector.start()
    
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 1024, 768)
    
    state = "COUNTDOWN"
    start_time = time.time()
    current_idx = 0
    dataset = []
    last_frame_id = -1
    
    try:
        while True:
            packet = detector.get_latest_packet(min_frame_id=last_frame_id)
            if not packet:
                time.sleep(0.001)
                continue
            last_frame_id = packet.frame_id
            vis = packet.frame.copy()
            h, w = vis.shape[:2]
            now = time.time()
            elapsed = now - start_time

            if state == "COUNTDOWN":
                rem = 5.0 - elapsed
                if rem <= 0:
                    state = "GAP"
                    start_time = time.time()
                else:
                    cv2.putText(vis, f"Starting in: {int(rem)+1}", (w//2 - 150, h//2), 1, 3.0, (0, 255, 255), 3)

            elif state == "GAP":
                rem = config["cooldown"] - elapsed
                if rem <= 0:
                    state = "RECORDING"
                    start_time = time.time()
                else:
                    label = config["labels"][current_idx]
                    cv2.putText(vis, f"NEXT: {label}", (w//2-100, h//2), 1, 2.5, (255, 255, 0), 2)
                    cv2.putText(vis, f"Prepare in {int(rem)+1}s", (w//2-100, h//2+50), 1, 1.5, (200, 200, 0), 2)

            elif state == "RECORDING":
                rem = config["record_time"] - elapsed
                label = config["labels"][current_idx]
                if rem <= 0:
                    current_idx += 1
                    if current_idx >= len(config["labels"]):
                        state = "DONE"
                    else:
                        state = "GAP"
                        start_time = time.time()
                else:
                    cv2.putText(vis, f"RECORDING: {label}", (30, 60), 1, 2.0, (0, 0, 255), 3)
                    cv2.putText(vis, f"Remaining: {rem:.1f}s", (30, 100), 1, 1.5, (0, 255, 0), 2)
                    if packet.results:
                        for r in packet.results:
                            dataset.append({
                                "label": label,
                                "hand_side": r.hand_side,
                                "landmarks": dump_landmark(r)
                            })
                            # Viz skeleton
                            for lm in r.landmarks:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(vis, (cx, cy), 5, (0,255,0), -1)

            elif state == "DONE":
                cv2.putText(vis, "COMPLETED!", (w//2-150, h//2), 1, 3.0, (0, 255, 0), 4)
                cv2.imshow("Calibration", vis)
                cv2.waitKey(2000)
                break

            cv2.imshow("Calibration", vis)
            if cv2.waitKey(1) == 27: break
            
    finally:
        detector.release()
        cv2.destroyAllWindows()
        if dataset:
            old_data = []
            if DATA_FILE.exists():
                old_data = json.load(open(DATA_FILE, "r"))
            old_data.extend(dataset)
            with open(DATA_FILE, "w") as f:
                json.dump(old_data, f, indent=2)
            print(f"Recorded {len(dataset)} samples. Total database: {len(old_data)}")

if __name__ == "__main__":
    main()
