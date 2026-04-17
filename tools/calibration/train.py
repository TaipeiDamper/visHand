import os
import json
import math
import numpy as np
from pathlib import Path

def extract_features(landmarks, hand_side):
    """
    Convert a list of 21 {x, y, z} dicts into a normalized 63-element feature vector.
    """
    # 1. Extract raw points
    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks])
    
    # 2. Translate so WRIST (index 0) is at origin
    wrist = pts[0].copy()
    pts -= wrist
    
    # 3. Mirror X for Left hand (assume right hand standard)
    if hand_side.upper() == "LEFT":
        pts[:, 0] = -pts[:, 0]
        
    # 4. Scale by palm size (distance from wrist to MIDDLE_MCP index 9)
    # Using 9 (MIDDLE_MCP) as it's a stable joint
    scale = np.linalg.norm(pts[9] - pts[0])
    if scale > 1e-6:
        pts /= scale
        
    # 5. Flatten to 1D vector
    return pts.flatten()

def main():
    # Use the local data folder in tools/calibration
    data_path = Path(__file__).parent / "data" / "dataset_ml.json"
    if not data_path.exists():
        # Fallback to the original dataset if it exists
        data_path = Path(__file__).parent / "data" / "dataset_original.json"
        
    if not data_path.exists():
        print(f"Dataset not found.")
        return
        
    print(f"Loading dataset from {data_path}...")
    with open(data_path, "r") as f:
        dataset = json.load(f)
        
    X = []
    Y = []
    
    for item in dataset:
        lms = item.get("landmarks")
        label = item.get("label")
        side = item.get("hand_side", "RIGHT")
        
        if not lms or len(lms) != 21:
            continue
            
        feat = extract_features(lms, side)
        X.append(feat)
        Y.append(label)
        
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=str)
    
    print(f"Extracted {len(X)} valid samples. Feature size: {X.shape[1]}")
    
    # Simple evaluation (optional test split)
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    X = X[indices]
    Y = Y[indices]
    
    split = int(len(X) * 0.8)
    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]
    
    # 1-NN evaluation on test set
    if len(X_test) > 0:
        correct = 0
        for i in range(len(X_test)):
            q = X_test[i]
            # L2 distance
            dists = np.linalg.norm(X_train - q, axis=1)
            pred = Y_train[np.argmin(dists)]
            if pred == Y_test[i]:
                correct += 1
        print(f"1-NN Cross-Validation Accuracy (80/20 split): {correct/len(X_test)*100:.2f}%")
        
    # Model Save (Save to both local and global models folder)
    local_model_path = Path(__file__).parent / "models" / "gesture_knn.npz"
    global_model_path = Path(__file__).parent.parent.parent / "core" / "models" / "gesture_knn.npz"
    global_model_path.parent.mkdir(parents=True, exist_ok=True)
    local_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(local_model_path, features=X, labels=Y)
    np.savez_compressed(global_model_path, features=X, labels=Y)
    print(f"Successfully saved model to:\n  - {local_model_path}\n  - {global_model_path}")

if __name__ == "__main__":
    main()
