from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from core.types import FramePacket


class CaptureWorker:
    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._frame_lock = threading.Lock()
        self._frame_slot: Optional[FramePacket] = None
        self._frame_counter = 0

    def run(self, stop_event: threading.Event):
        while not stop_event.is_set():
            t0 = time.perf_counter()
            ok, frame = self._cap.read()
            t1 = time.perf_counter()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            pkt = FramePacket(
                frame=frame,
                frame_id=self._frame_counter,
                t_capture=time.time(),
                capture_ms=(t1 - t0) * 1000.0,
            )
            self._frame_counter += 1
            with self._frame_lock:
                self._frame_slot = pkt

    def get_latest_frame(self) -> Optional[FramePacket]:
        with self._frame_lock:
            return self._frame_slot

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
        return ok, frame
