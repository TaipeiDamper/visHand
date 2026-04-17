"""
visHand — Gesture Context
=========================
提供快取機制的特徵計算環境，作為 Evaluator 判斷手勢的基底。
避免不同手勢重複呼叫 math_tools 進行昂貴的數學運算。
"""

from functools import cached_property
from typing import List, Optional, Tuple

from config.settings import Settings
from core.types import Point3D
import utils.math_tools as mt


class GestureContext:
    def __init__(
        self,
        lm: List[Point3D],
        settings: Settings,
        anchor_history: List[Tuple[float, float]],
        forearm_vector: Optional[Tuple[float, float, float]] = None,
        wrist_orientation_confidence: float = 0.0,
        wrist_body_x: float = 0.0,
        forearm_depth_delta: float = 0.0,
        arm_confidence: float = 0.0,
    ):
        self.lm = lm
        self.settings = settings
        self.anchor_history = anchor_history
        self.forearm_vector = forearm_vector
        self.wrist_orientation_confidence = float(wrist_orientation_confidence)
        self.wrist_body_x = float(wrist_body_x)
        self.forearm_depth_delta = float(forearm_depth_delta)
        self.arm_confidence = float(arm_confidence)

    @cached_property
    def palm_width(self) -> float:
        return mt.palm_width(self.lm)

    @cached_property
    def ext_array(self) -> Tuple[bool, ...]:
        """[Thumb, Index, Middle, Ring, Pinky]"""
        return tuple(mt.finger_is_extended(self.lm, i) for i in range(5))

    def is_extended(self, finger_id: int) -> bool:
        """Backward-compatible accessor; use ext_array for new code."""
        return bool(self.ext_array[finger_id])

    @cached_property
    def curl_array(self) -> Tuple[float, ...]:
        """[Thumb, Index, Middle, Ring, Pinky]"""
        return tuple(mt.finger_curl(self.lm, i) for i in range(5))

    def curl(self, finger_id: int) -> float:
        """Backward-compatible accessor; use curl_array for new code."""
        return float(self.curl_array[finger_id])

    @cached_property
    def pinch_distance(self) -> float:
        return mt.normalized_distance(self.lm[mt.THUMB_TIP], self.lm[mt.INDEX_TIP], self.lm)

    @cached_property
    def contact_score(self) -> float:
        return mt.pinch_contact_score(self.lm, pinch_threshold=self.settings.pinch_threshold)

    @cached_property
    def snap_ready_distance(self) -> float:
        return mt.normalized_distance(self.lm[mt.MIDDLE_TIP], self.lm[mt.THUMB_TIP], self.lm)

    @cached_property
    def velocity(self) -> float:
        return mt.instant_velocity(self.anchor_history, self.settings.anchor_history_size)
