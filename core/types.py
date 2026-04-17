from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Point3D:
    x: float
    y: float
    z: float


@dataclass
class HandLandmarkResult:
    landmarks: List[Point3D]
    hand_side: str
    raw_landmarks: object
    confidence: float = 0.0
    track_id: Optional[str] = None


@dataclass
class FramePacket:
    frame: np.ndarray
    frame_id: int
    t_capture: float
    capture_ms: float


@dataclass
class InferencePacket:
    frame: np.ndarray
    frame_id: int
    t_capture: float
    t_infer_done: float
    capture_ms: float
    mp_ms: float
    pose_ms: float
    mode: str
    skipped: bool
    results: Optional[List[HandLandmarkResult]]
    arm_features: Optional[dict]
    input_quality_score: float = 1.0
    tracking_quality: str = "good"
    degraded_mode: bool = False


@dataclass
class QualityBlock:
    input_quality_score: float
    tracking_quality: str
