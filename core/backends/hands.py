from __future__ import annotations

import os
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from core.types import HandLandmarkResult, Point3D


class LegacyHandsBackend:
    def __init__(self, max_hands: int, min_det: float, min_track: float):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[List[HandLandmarkResult]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not mp_result.multi_hand_landmarks:
            return None

        out: List[HandLandmarkResult] = []
        for i, hand_lm in enumerate(mp_result.multi_hand_landmarks):
            if mp_result.multi_handedness:
                c = mp_result.multi_handedness[i].classification[0]
                side = c.label.upper()
                score = float(c.score)
            else:
                side = "RIGHT"
                score = 0.0
            out.append(
                HandLandmarkResult(
                    landmarks=[Point3D(lm.x, lm.y, lm.z) for lm in hand_lm.landmark],
                    hand_side=side,
                    raw_landmarks=hand_lm,
                    confidence=score,
                )
            )
        return out

    def close(self) -> None:
        self._hands.close()


class TasksHandsBackend:
    def __init__(self, settings, max_hands: int):
        if not hasattr(mp, "tasks"):
            raise RuntimeError("MediaPipe tasks API is unavailable.")

        base_options_cls = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        model_path = settings.hand_landmarker_model_path.strip()
        if not model_path:
            raise RuntimeError("hand_landmarker_model_path is required for tasks backend.")
        if not os.path.exists(model_path):
            raise RuntimeError(f"HandLandmarker model not found: {model_path}")

        # Windows GPU Support is tricky; try GPU if requested, fallback to CPU
        delegate = base_options_cls.Delegate.CPU
        if settings.use_gpu_delegate:
            delegate = base_options_cls.Delegate.GPU

        try:
            base_options = base_options_cls(model_asset_path=model_path, delegate=delegate)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_hands,
                min_hand_detection_confidence=settings.min_detection_confidence,
                min_hand_presence_confidence=settings.min_tracking_confidence,
                min_tracking_confidence=settings.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            if delegate == base_options_cls.Delegate.GPU:
                # Retry with CPU
                base_options = base_options_cls(model_asset_path=model_path, delegate=base_options_cls.Delegate.CPU)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_hands,
                    min_hand_detection_confidence=settings.min_detection_confidence,
                    min_hand_presence_confidence=settings.min_tracking_confidence,
                    min_tracking_confidence=settings.min_tracking_confidence,
                )
                self._landmarker = vision.HandLandmarker.create_from_options(options)
            else:
                raise e

    def detect(self, frame_bgr: np.ndarray) -> Optional[List[HandLandmarkResult]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.hand_landmarks:
            return None

        out: List[HandLandmarkResult] = []
        for i, hand_lm in enumerate(result.hand_landmarks):
            side = "RIGHT"
            score = 0.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                side = cat.category_name.upper()
                score = float(cat.score)

            raw = landmark_pb2.NormalizedLandmarkList()
            for lm in hand_lm:
                raw.landmark.add(x=lm.x, y=lm.y, z=lm.z)

            out.append(
                HandLandmarkResult(
                    landmarks=[Point3D(lm.x, lm.y, lm.z) for lm in hand_lm],
                    hand_side=side,
                    raw_landmarks=raw,
                    confidence=score,
                )
            )
        return out

    def close(self) -> None:
        self._landmarker.close()


def build_hands_backend(settings, max_hands: int):
    prefer = settings.preferred_detector_backend.lower()
    if prefer == "tasks":
        try:
            return TasksHandsBackend(settings, max_hands=max_hands)
        except Exception as exc:
            print(f"[visHand] Tasks backend unavailable ({exc}); fallback to legacy Hands.")
    return LegacyHandsBackend(
        max_hands=max_hands,
        min_det=settings.min_detection_confidence,
        min_track=settings.min_tracking_confidence,
    )
