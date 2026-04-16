"""
visHand — Gesture Context
=========================
提供快取機制的特徵計算環境，作為 Evaluator 判斷手勢的基底。
避免不同手勢重複呼叫 math_tools 進行昂貴的數學運算。
"""

from typing import List, Tuple
from dataclasses import dataclass
from core.detector import Point3D
from config.settings import Settings
import utils.math_tools as mt

class GestureContext:
    def __init__(self, lm: List[Point3D], settings: Settings, anchor_history: List[Tuple[float, float]]):
        self.lm = lm
        self.settings = settings
        self.anchor_history = anchor_history
        
        # Caches
        self._palm_width = None
        self._finger_ext = [None] * 5
        self._finger_curl = [None] * 5
        self._velocity = None

    @property
    def palm_width(self) -> float:
        if self._palm_width is None:
            self._palm_width = mt.palm_width(self.lm)
        return self._palm_width

    def is_extended(self, finger_id: int) -> bool:
        """finger_id: 0 = Thumb, 1 = Index, ..., 4 = Pinky"""
        if self._finger_ext[finger_id] is None:
            self._finger_ext[finger_id] = mt.finger_is_extended(self.lm, finger_id)
        return self._finger_ext[finger_id]

    def curl(self, finger_id: int) -> float:
        """finger_id: 0 = Thumb, 1 = Index, ..., 4 = Pinky"""
        if self._finger_curl[finger_id] is None:
            self._finger_curl[finger_id] = mt.finger_curl(self.lm, finger_id)
        return self._finger_curl[finger_id]

    @property
    def ext_array(self) -> List[bool]:
        """[Thumb, Index, Middle, Ring, Pinky]"""
        return [self.is_extended(i) for i in range(5)]

    @property
    def curl_array(self) -> List[float]:
        """[Thumb, Index, Middle, Ring, Pinky]"""
        return [self.curl(i) for i in range(5)]

    @property
    def pinch_distance(self) -> float:
        return mt.normalized_distance(self.lm[mt.THUMB_TIP], self.lm[mt.INDEX_TIP], self.lm)

    @property
    def snap_ready_distance(self) -> float:
        return mt.normalized_distance(self.lm[mt.MIDDLE_TIP], self.lm[mt.THUMB_TIP], self.lm)

    @property
    def velocity(self) -> float:
        if self._velocity is None:
            self._velocity = mt.instant_velocity(self.anchor_history, self.settings.anchor_history_size)
        return self._velocity
