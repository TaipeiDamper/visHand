"""
Calibration Tool for Gestures
Run this script to collect raw 3D landmark data for hands across different poses and orientations.
"""
import sys
import os
import time
import json
import cv2
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import Settings
from core.detector import HandDetector

def dump_landmark(lm_result):
    """提取 3D landmark"""
    lms = []
    for lm in lm_result.landmarks:
        lms.append({"x": lm.x, "y": lm.y, "z": lm.z})
    return lms

def main():
    # 強制開啟雙手偵測
    settings = Settings(max_hands=2)
    detector = HandDetector(settings)
    
    print("[Calibration] Starting Data Collection...")
    
    try:
        detector.start()
    except RuntimeError as e:
        print(e)
        sys.exit(1)
        
    cv2.namedWindow("visHand  |  Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("visHand  |  Calibration", 1024, 768)
    
    # 校正所需的手勢標籤
    labels = ["0", "1", "2", "3", "4", "5"]
    
    state = "START"
    start_time = time.time()
    current_label_idx = 0
    
    dataset = []
    last_frame_id = -1
    
    # 字體設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    try:
        while True:
            packet = detector.get_latest_packet(min_frame_id=last_frame_id)
            if not packet:
                time.sleep(0.001)
                continue
                
            last_frame_id = packet.frame_id
            frame = packet.frame
            if frame is None:
                break
                
            vis = frame.copy()
            h, w = vis.shape[:2]
            
            now = time.time()
            elapsed = now - start_time
            
            # --- 狀態機 ---
            if state == "START":
                rem = 5.0 - elapsed
                if rem <= 0:
                    state = "COOLDOWN"
                    start_time = time.time()
                else:
                    cv2.putText(vis, f"Calibration Starting: {int(rem)+1}", (w//2 - 250, h//2), font, 1.5, (0, 255, 255), 3)
                    cv2.putText(vis, "Please prepare BOTH HANDS", (w//2 - 220, h//2 + 50), font, 0.8, (200, 200, 200), 2)
            
            elif state == "COOLDOWN":
                rem = 2.0 - elapsed
                if rem <= 0:
                    state = "RECORDING"
                    start_time = time.time()
                else:
                    label = labels[current_label_idx]
                    cv2.putText(vis, f"Get ready to show number >>> {label} <<<", (w//2 - 350, h//2 - 30), font, 1.2, (0, 255, 255), 3)
                    cv2.putText(vis, f"Resuming in {int(rem)+1}", (w//2 - 120, h//2 + 30), font, 1.0, (255, 200, 0), 2)
            
            elif state == "RECORDING":
                rem = 5.0 - elapsed  # 記錄 5 秒
                label = labels[current_label_idx]
                if rem <= 0:
                    current_label_idx += 1
                    if current_label_idx >= len(labels):
                        state = "DONE"
                    else:
                        state = "COOLDOWN"
                        start_time = time.time()
                else:
                    cv2.putText(vis, f"[ RECORDING MODE ] TARGET: {label}", (30, 60), font, 1.2, (0, 0, 255), 3)
                    cv2.putText(vis, "Rotate and move your hands slowly!", (30, 100), font, 0.8, (0, 255, 255), 2)
                    cv2.putText(vis, f"Time left: {rem:.1f}s", (30, 140), font, 0.9, (0, 0, 255), 2)
                    
                    # 進行資料記錄 (雙手都會被記錄)
                    if packet.results:
                        for r in packet.results:
                            dataset.append({
                                "label": label,
                                "hand_side": r.hand_side,
                                "timestamp": packet.t_capture,
                                "landmarks": dump_landmark(r)
                            })
                            
                            # 繪製視覺化骨架
                            for lm in r.landmarks:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
            
            elif state == "DONE":
                cv2.putText(vis, "Calibration Complete!", (w//2 - 200, h//2), font, 1.2, (0, 255, 0), 3)
                cv2.putText(vis, "Saving data and exiting...", (w//2 - 180, h//2 + 50), font, 0.8, (0, 255, 0), 2)
                cv2.imshow("visHand  |  Calibration", vis)
                cv2.waitKey(2000)
                break
                
            cv2.imshow("visHand  |  Calibration", vis)
            if cv2.waitKey(1) in (27, ord('q'), ord('Q')):
                print("Cancelled by user.")
                break
                
    finally:
        detector.release()
        cv2.destroyAllWindows()
        
        if dataset:
            out_dir = Path("scratch")
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / "gesture_calibration_dataset.json"
            with open(out_file, "w") as f:
                json.dump(dataset, f, indent=2)
            print(f"\n[Success] Dataset successfully saved to '{out_file}'")
            print(f"[Success] Total recorded samples: {len(dataset)}")

if __name__ == "__main__":
    main()
