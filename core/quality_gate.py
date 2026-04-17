from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQuality:
    brightness: float
    blur_score: float
    overexposed: bool
    overall_score: float
    tracking_quality: str
    degraded_mode: bool


class QualityGate:
    def __init__(self, settings):
        self.s = settings

    def evaluate(self, frame: np.ndarray) -> FrameQuality:
        if frame is None or frame.size == 0:
            return FrameQuality(
                brightness=0.0,
                blur_score=0.0,
                overexposed=True,
                overall_score=0.0,
                tracking_quality="poor",
                degraded_mode=True,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray) / 255.0)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        overexposed_ratio = float(np.mean(gray >= 245))
        overexposed = overexposed_ratio > self.s.quality_overexposed_ratio_max

        blur_norm = min(1.0, lap_var / max(self.s.quality_blur_var_min, 1e-6))
        bright_norm = min(1.0, brightness / max(self.s.quality_brightness_min, 1e-6))
        expose_penalty = 0.0 if overexposed else 1.0

        overall = (bright_norm * 0.35) + (blur_norm * 0.45) + (expose_penalty * 0.20)
        overall = float(max(0.0, min(1.0, overall)))

        if overall < self.s.quality_poor_threshold:
            tracking_quality = "poor"
        elif overall < self.s.quality_degrade_threshold:
            tracking_quality = "degraded"
        else:
            tracking_quality = "good"

        return FrameQuality(
            brightness=brightness,
            blur_score=lap_var,
            overexposed=overexposed,
            overall_score=overall,
            tracking_quality=tracking_quality,
            degraded_mode=self.should_degrade_score(overall),
        )

    def should_degrade(self, quality: FrameQuality) -> bool:
        return self.should_degrade_score(quality.overall_score)

    def should_degrade_score(self, score: float) -> bool:
        return float(score) < self.s.quality_degrade_threshold
