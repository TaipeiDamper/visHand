"""
visHand — Landmark Filters
===========================
Implements the 1 Euro Filter for smooth, low-latency hand tracking.

The 1 Euro Filter adapts its cutoff frequency based on movement speed:
  - Slow movement  → low cutoff → aggressive smoothing (removes jitter)
  - Fast movement  → high cutoff → minimal smoothing (preserves responsiveness)

Reference: Casiez et al. (2012) "1€ Filter: A Simple Speed-based Low-pass
           Filter for Noisy Input in Interactive Systems".
"""

from __future__ import annotations

import math
from typing import Optional, List
from core.detector import Point3D


# ---------------------------------------------------------------------------
# Low-pass filter (internal building block)
# ---------------------------------------------------------------------------

class _LowPassFilter:
    def __init__(self, alpha: float):
        self._alpha = alpha
        self._last: Optional[float] = None

    def set_alpha(self, alpha: float):
        self._alpha = alpha

    def filter(self, value: float) -> float:
        if self._last is None:
            self._last = value
        else:
            self._last = self._alpha * value + (1.0 - self._alpha) * self._last
        return self._last

    @property
    def last_value(self) -> Optional[float]:
        return self._last

    def reset(self):
        self._last = None


# ---------------------------------------------------------------------------
# 1 Euro Filter
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """
    Adaptive low-pass filter for a single scalar signal.

    Args:
        freq        Initial sampling rate estimate (Hz).
        min_cutoff  Minimum cutoff frequency (Hz). Lower = smoother at rest.
        beta        Speed coefficient. Higher = less lag during fast motion.
        d_cutoff    Cutoff for the derivative low-pass filter (Hz).
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self._freq = max(freq, 1.0)
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff

        self._x_filt = _LowPassFilter(self._alpha(min_cutoff))
        self._dx_filt = _LowPassFilter(self._alpha(d_cutoff))
        self._last_time: Optional[float] = None

    # ── Processing ───────────────────────────────────────────────────────────

    def __call__(self, x: float, timestamp: float) -> float:
        """Filter a new sample.  timestamp is Unix time in seconds."""
        # Update frequency from wall-clock delta
        if self._last_time is not None:
            dt = timestamp - self._last_time
            if dt > 1e-6:
                self._freq = 1.0 / dt
        self._last_time = timestamp

        # Compute derivative
        prev = self._x_filt.last_value
        dx = 0.0 if prev is None else (x - prev) * self._freq
        edx = self._dx_filt.filter(dx)

        # Adaptive cutoff
        cutoff = self._min_cutoff + self._beta * abs(edx)
        self._x_filt.set_alpha(self._alpha(cutoff))

        return self._x_filt.filter(x)

    def reset(self):
        """Clear state (call when tracking is lost and resumes)."""
        self._x_filt.reset()
        self._dx_filt.reset()
        self._last_time = None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self._freq
        return 1.0 / (1.0 + tau / te)


# ---------------------------------------------------------------------------
# Landmark Filter — wraps 21 × 3 = 63 OneEuroFilters
# ---------------------------------------------------------------------------

class LandmarkFilter:
    """
    Applies a OneEuroFilter independently to every axis of every landmark.

    Usage:
        lf = LandmarkFilter(settings)
        smooth_lm = lf.apply(raw_landmarks, timestamp)
        stable = lf.is_stable   # True after warmup_frames of continuous tracking
    """

    def __init__(self, settings):
        freq = float(settings.target_fps)
        mc   = settings.one_euro_min_cutoff
        beta = settings.one_euro_beta
        dc   = settings.one_euro_d_cutoff

        self._warmup = settings.stability_warmup_frames
        self._frame_count = 0

        # 21 landmarks × 3 axes (x, y, z)
        self._filters: List[List[OneEuroFilter]] = [
            [OneEuroFilter(freq, mc, beta, dc) for _ in range(3)]
            for _ in range(21)
        ]

    # ── Public API ───────────────────────────────────────────────────────────

    def apply(self, landmarks: List[Point3D], timestamp: float) -> List[Point3D]:
        """Return a new list of smoothed Point3D objects."""
        self._frame_count += 1
        result = []
        for i, lm in enumerate(landmarks):
            sx = self._filters[i][0](lm.x, timestamp)
            sy = self._filters[i][1](lm.y, timestamp)
            sz = self._filters[i][2](lm.z, timestamp)
            result.append(Point3D(sx, sy, sz))
        return result

    @property
    def is_stable(self) -> bool:
        """True once enough frames have accumulated for reliable smoothing."""
        return self._frame_count >= self._warmup

    def reset(self):
        """Reset all filters (call when hand disappears from frame)."""
        self._frame_count = 0
        for axes in self._filters:
            for f in axes:
                f.reset()
