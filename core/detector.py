"""
visHand — Async Hand Detector (Facade)
======================================
Combines capture and inference workers while keeping a stable public API.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from core.capture import CaptureWorker
from core.inference import InferenceWorker
from core.types import HandLandmarkResult, InferencePacket


class HandDetector:
    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    def __init__(self, settings):
        self.settings = settings
        self._cap = cv2.VideoCapture(settings.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[visHand] Cannot open camera at index {settings.camera_index}.\n"
                "  → Try changing camera_index in Settings."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, settings.target_fps)

        self._stop = threading.Event()
        self._capture = CaptureWorker(self._cap)
        self._inference = InferenceWorker(settings, frame_supplier=self._capture.get_latest_frame)

        self._capture_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None

    def start(self):
        if self._capture_thread and self._capture_thread.is_alive():
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(target=self._capture.run, args=(self._stop,), daemon=True, name="visHandCapture")
        self._infer_thread = threading.Thread(target=self._inference.run, args=(self._stop,), daemon=True, name="visHandInference")
        self._capture_thread.start()
        self._infer_thread.start()

    def update_runtime_hint(self, logic: str, intent: str, velocity: float):
        self._inference.update_runtime_hint(logic=logic, intent=intent, velocity=velocity)

    # Backward-compatible API for synchronous callers.
    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        return self._capture.read_frame()

    def extract_landmarks(self, frame: np.ndarray) -> Optional[List[HandLandmarkResult]]:
        return self._inference.extract_landmarks(frame)

    def get_latest_packet(self, min_frame_id: int = -1) -> Optional[InferencePacket]:
        return self._inference.get_latest_packet(min_frame_id=min_frame_id)

    def release(self):
        self._stop.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=0.8)
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=0.8)
        self._inference.close()
        self._cap.release()

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def __enter__(self) -> HandDetector:
        self.start()
        return self

    def __exit__(self, *_):
        self.release()
