from __future__ import annotations

from core.gestures import registry


class ActionSafetyLayer:
    def __init__(self, settings):
        self.s = settings
        self._emergency_cancel_gesture = "OPEN_PALM"
        self._emergency_hold_frames = max(1, int(settings.emergency_cancel_hold_frames))
        self._hold_count = 0

    def reset(self):
        self._hold_count = 0

    def check_emergency_cancel(self, raw_intent: str) -> bool:
        if raw_intent == self._emergency_cancel_gesture:
            self._hold_count += 1
        else:
            self._hold_count = 0
        return self._hold_count >= self._emergency_hold_frames

    def filter_high_risk(self, intent: str, degraded_mode: bool) -> str:
        if not degraded_mode or not self.s.degraded_low_risk_only:
            return intent
        if intent in ("IDLE", "OPEN_PALM"):
            return intent
        g = registry.get(intent)
        if g is None:
            return "IDLE"
        return intent if g.risk_level == "low" else "IDLE"
