"""
visHand — Kinematics utilities
==============================
Bone length clamping + lightweight constant-velocity predictor.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from core.types import Point3D

# Parent-child pairs on hand skeleton.
_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _to_np(lm: List[Point3D]) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64)


def _to_points(arr: np.ndarray) -> List[Point3D]:
    return [Point3D(float(x), float(y), float(z)) for x, y, z in arr]


class BoneLengthClamp:
    def __init__(self, settings):
        self._calibration_frames = int(settings.clamp_calibration_frames)
        self._tol = float(settings.clamp_tolerance_ratio)
        self._samples: List[np.ndarray] = []
        self._base_lengths: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._base_lengths is not None

    def reset(self):
        self._samples.clear()
        self._base_lengths = None

    def apply(self, landmarks: List[Point3D]) -> List[Point3D]:
        arr = _to_np(landmarks)
        lengths = self._bone_lengths(arr)

        if self._base_lengths is None:
            self._samples.append(lengths)
            if len(self._samples) >= self._calibration_frames:
                stack = np.stack(self._samples, axis=0)
                median = np.median(stack, axis=0)
                spread = np.std(stack, axis=0)
                # If calibration is too unstable, restart calibration window.
                if np.any((spread / np.maximum(median, 1e-6)) > 0.15):
                    self._samples.clear()
                else:
                    self._base_lengths = median
                    self._samples.clear()
            return landmarks

        lo = self._base_lengths * (1.0 - self._tol)
        hi = self._base_lengths * (1.0 + self._tol)
        out = arr.copy()
        for i, (parent, child) in enumerate(_BONES):
            p = out[parent]
            c = out[child]
            v = c - p
            dist = float(np.linalg.norm(v))
            if dist < 1e-6:
                continue
            target = float(np.clip(dist, lo[i], hi[i]))
            out[child] = p + v * (target / dist)
        return _to_points(out)

    @staticmethod
    def _bone_lengths(arr: np.ndarray) -> np.ndarray:
        vals = []
        for parent, child in _BONES:
            vals.append(np.linalg.norm(arr[child] - arr[parent]))
        return np.array(vals, dtype=np.float64)


class LandmarkKalmanPredictor:
    """
    Lightweight alpha-beta tracker for selected landmarks only.
    """

    def __init__(self, settings, tracked_ids: Tuple[int, ...] = (4, 8, 12)):
        self._tracked_ids = tuple(tracked_ids)
        self._alpha = float(settings.kalman_alpha)
        self._beta = float(settings.kalman_beta)
        self._horizon = float(settings.kalman_predict_horizon_ms) / 1000.0
        self._residual_reset = float(settings.kalman_residual_reset)

        self._state: Dict[int, Dict[str, np.ndarray]] = {}
        self._last_ts: float | None = None
        self.last_residual_rms: float = 0.0

    def reset(self):
        self._state.clear()
        self._last_ts = None
        self.last_residual_rms = 0.0

    def update_and_predict(self, landmarks: List[Point3D], timestamp: float) -> List[Point3D]:
        if not landmarks:
            return landmarks
        arr = _to_np(landmarks)
        if self._last_ts is None:
            self._last_ts = timestamp
            for i in self._tracked_ids:
                self._state[i] = {
                    "pos": arr[i].copy(),
                    "vel": np.zeros(3, dtype=np.float64),
                }
            return landmarks

        dt = max(1e-3, timestamp - self._last_ts)
        self._last_ts = timestamp

        residuals = []
        for i in self._tracked_ids:
            m = arr[i]
            if i not in self._state:
                self._state[i] = {"pos": m.copy(), "vel": np.zeros(3, dtype=np.float64)}
                continue

            state = self._state[i]
            pos = state["pos"]
            vel = state["vel"]

            pred = pos + vel * dt
            residual = m - pred
            pos = pred + self._alpha * residual
            vel = vel + (self._beta / dt) * residual

            future = pos + vel * self._horizon
            state["pos"] = pos
            state["vel"] = vel
            arr[i] = future
            residuals.append(float(np.linalg.norm(residual)))

        if residuals:
            self.last_residual_rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
            if self.last_residual_rms > self._residual_reset:
                self.reset()

        return _to_points(arr)
