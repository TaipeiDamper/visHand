from __future__ import annotations

from typing import List, Optional, Tuple

from core.gestures import GestureDef


class ArmAssistGate:
    def __init__(self, settings):
        self._settings = settings

    def apply(
        self,
        winners: List[Tuple[GestureDef, float]],
        arm_view: dict,
        hand_side: str,
        low_conf_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
    ) -> Tuple[List[Tuple[GestureDef, float]], bool]:
        if not winners:
            return winners, False

        low_conf_threshold = (
            self._settings.arm_assist_low_conf_threshold
            if low_conf_threshold is None
            else low_conf_threshold
        )
        margin_threshold = (
            self._settings.arm_assist_margin_threshold
            if margin_threshold is None
            else margin_threshold
        )

        arm_conf = float(arm_view.get("arm_confidence", 0.0))
        if arm_conf < self._settings.arm_confidence_min:
            return winners, False

        top_score = float(winners[0][1])
        margin = self.score_margin(winners)
        should_apply = top_score < low_conf_threshold or margin < margin_threshold
        if not should_apply:
            return winners, False

        boosted: List[Tuple[GestureDef, float]] = []
        for gesture, score in winners:
            if not getattr(gesture, "arm_assist_allowed", False):
                boosted.append((gesture, score))
                continue
            base_weight = getattr(gesture, "arm_assist_weight", self._settings.arm_assist_weight_default)
            boost = self.direction_boost(gesture.name, hand_side, arm_view)
            boosted.append((gesture, score + base_weight * boost))

        boosted.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        return boosted, True

    @staticmethod
    def score_margin(winners: List[Tuple[GestureDef, float]]) -> float:
        if len(winners) < 2:
            return 1.0
        return float(winners[0][1] - winners[1][1])

    @staticmethod
    def select_arm_view(hand_side: str, arm_features: Optional[dict]) -> dict:
        if not arm_features:
            return {}
        side = hand_side.lower()
        if side in arm_features:
            return dict(arm_features[side])
        return {}

    @staticmethod
    def direction_boost(intent_name: str, hand_side: str, arm_view: dict) -> float:
        _ = hand_side
        forearm_vector = arm_view.get("forearm_vector")
        if forearm_vector is None:
            return 0.0
        fx = float(forearm_vector[0])
        if intent_name in ("INDEX_POINT_LEFT", "SWIPE_LEFT"):
            return 1.0 if fx < -0.1 else -0.5
        if intent_name in ("INDEX_POINT_RIGHT", "SWIPE_RIGHT"):
            return 1.0 if fx > 0.1 else -0.5
        return 0.0

    @staticmethod
    def infer_z_hint(arm_view: dict) -> str:
        dz = float(arm_view.get("forearm_depth_delta", 0.0))
        if dz < -0.02:
            return "forward"
        if dz > 0.02:
            return "backward"
        return "neutral"

    def fuse_handedness(self, hand_side: str, arm_view: dict) -> str:
        fused = hand_side
        conf = float(arm_view.get("arm_confidence", 0.0))
        wrist_body_x = float(arm_view.get("wrist_body_x", 0.0))
        if conf >= self._settings.arm_confidence_min:
            pose_side = "LEFT" if wrist_body_x < 0 else "RIGHT"
            hand_side = hand_side.upper()
            if pose_side != hand_side and abs(wrist_body_x) > 0.03:
                fused = pose_side
        return fused
