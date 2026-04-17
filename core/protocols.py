from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np

from core.types import HandLandmarkResult


class HandsBackend(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> Optional[List[HandLandmarkResult]]:
        ...

    def close(self) -> None:
        ...


class PoseBackend(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> Optional[dict]:
        ...

    def close(self) -> None:
        ...
