from __future__ import annotations

import math
import os
import time
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from core.types import HandLandmarkResult


class LegacyPoseBackend:
    def __init__(self, min_confidence: float):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[dict]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        rgb.flags.writeable = True
        if not result.pose_landmarks:
            return None
        return {"landmarks": result.pose_landmarks.landmark}

    def close(self) -> None:
        self._pose.close()


class TasksPoseBackend:
    def __init__(self, settings):
        if not hasattr(mp, "tasks"):
            raise RuntimeError("MediaPipe tasks API is unavailable.")
        vision = mp.tasks.vision
        base_options_cls = mp.tasks.BaseOptions

        model_path = settings.pose_landmarker_model_path.strip()
        if not model_path:
            raise RuntimeError("pose_landmarker_model_path is required for tasks backend.")
        if not os.path.exists(model_path):
            raise RuntimeError(f"PoseLandmarker model not found: {model_path}")

        base_options = base_options_cls(
            model_asset_path=model_path,
            delegate=base_options_cls.Delegate.CPU,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=settings.pose_min_detection_confidence,
            min_pose_presence_confidence=settings.pose_min_detection_confidence,
            min_tracking_confidence=settings.pose_min_detection_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> Optional[dict]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        return {"landmarks": result.pose_landmarks[0]}

    def close(self) -> None:
        self._landmarker.close()


def build_pose_backend(settings):
    try:
        if settings.pose_landmarker_model_path.strip():
            return TasksPoseBackend(settings)
        return LegacyPoseBackend(settings.pose_min_detection_confidence)
    except Exception as exc:
        print(f"[visHand] Pose backend unavailable ({exc}); arm assist disabled.")
        return None


def estimate_arm_features(
    pose_backend,
    frame: np.ndarray,
    results: Optional[List[HandLandmarkResult]],
) -> Tuple[Optional[dict], float]:
    if pose_backend is None:
        return None, 0.0
    t0 = time.perf_counter()
    pose = pose_backend.detect(frame)
    pose_ms = (time.perf_counter() - t0) * 1000.0
    if not pose:
        return None, pose_ms

    lms = pose["landmarks"]
    # BlazePose indices.
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16

    def get_lm(idx):
        lm = lms[idx]
        x = float(getattr(lm, "x", 0.0))
        y = float(getattr(lm, "y", 0.0))
        z = float(getattr(lm, "z", 0.0))
        vis = float(getattr(lm, "visibility", 0.0))
        return x, y, z, vis

    ls = get_lm(L_SHOULDER)
    rs = get_lm(R_SHOULDER)
    cx = (ls[0] + rs[0]) / 2.0

    def build_side(elbow_idx, wrist_idx):
        ex, ey, ez, ev = get_lm(elbow_idx)
        wx, wy, wz, wv = get_lm(wrist_idx)
        vx, vy, vz = wx - ex, wy - ey, wz - ez
        norm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if norm < 1e-6:
            fv = (0.0, 0.0, 0.0)
        else:
            fv = (vx / norm, vy / norm, vz / norm)
        conf = min(ev, wv)
        return {
            "forearm_vector": fv,
            "wrist_body_x": wx - cx,
            "forearm_depth_delta": wz - ez,
            "arm_confidence": conf,
        }

    out = {
        "left": build_side(L_ELBOW, L_WRIST),
        "right": build_side(R_ELBOW, R_WRIST),
        "has_pose": True,
    }

    if results:
        for hand in results:
            side = hand.hand_side.lower()
            if side in out:
                out[side]["hand_confidence"] = hand.confidence
    return out, pose_ms
