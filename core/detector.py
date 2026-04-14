"""
visHand — Hand Detector
========================
Wraps OpenCV VideoCapture + MediaPipe Hands.

Usage:
    detector = HandDetector(settings)
    with detector:
        ok, frame = detector.read_frame()
        results = detector.extract_landmarks(frame)
        if results:
            result = results[0]            # HandLandmarkResult
            smooth_lm = result.landmarks   # list of 21 Point3D
            hand_side = result.hand_side   # "LEFT" | "RIGHT"
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Point3D:
    """A single 3-D landmark point (MediaPipe normalized coords, 0‥1)."""
    x: float
    y: float
    z: float  # relative depth; negative = closer to camera


@dataclass
class HandLandmarkResult:
    """Per-hand result for one frame."""
    landmarks: List[Point3D]   # 21 points in MediaPipe order
    hand_side: str             # "LEFT" | "RIGHT"
    raw_landmarks: object      # mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
                               # kept for drawing with mp.solutions.drawing_utils


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class HandDetector:
    """
    Opens the webcam and processes frames with MediaPipe Hands.

    Args:
        settings (Settings): project-wide configuration object.
    """

    # Expose connections so the visualizer can use them without importing mediapipe
    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    def __init__(self, settings):
        self.settings = settings

        # MediaPipe
        _mp_hands = mp.solutions.hands
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=settings.max_hands,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )

        # Camera
        self._cap = cv2.VideoCapture(settings.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[visHand] Cannot open camera at index {settings.camera_index}.\n"
                "  → Try changing camera_index in Settings."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, settings.target_fps)

    # ── Public API ───────────────────────────────────────────────────────────

    def read_frame(self) -> tuple[bool, np.ndarray]:
        """
        Capture one frame from the webcam.

        Returns:
            (success, frame_BGR) — frame is already mirrored (selfie view).
        """
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # mirror so left/right feel natural
        return ret, frame

    def extract_landmarks(self, frame: np.ndarray) -> Optional[List[HandLandmarkResult]]:
        """
        Run MediaPipe on a BGR frame.

        Returns:
            List of HandLandmarkResult (one per detected hand), or None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not mp_result.multi_hand_landmarks:
            return None

        results: List[HandLandmarkResult] = []
        for i, hand_lm in enumerate(mp_result.multi_hand_landmarks):
            # Determine handedness
            if mp_result.multi_handedness:
                label = mp_result.multi_handedness[i].classification[0].label
                # After the horizontal flip, MediaPipe's "Right" corresponds
                # to the user's right hand visually. We report as-is.
                hand_side = label.upper()  # "LEFT" | "RIGHT"
            else:
                hand_side = "RIGHT"

            landmarks = [
                Point3D(lm.x, lm.y, lm.z)
                for lm in hand_lm.landmark
            ]
            results.append(HandLandmarkResult(
                landmarks=landmarks,
                hand_side=hand_side,
                raw_landmarks=hand_lm,
            ))

        return results if results else None

    @property
    def frame_size(self) -> tuple[int, int]:
        """Actual (width, height) of the video stream."""
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        """Release camera and MediaPipe resources."""
        self._cap.release()
        self._hands.close()

    # ── Context manager support ──────────────────────────────────────────────

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, *_):
        self.release()
