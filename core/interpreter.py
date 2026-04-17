"""
visHand — Gesture Interpreter
==============================
The heart of visHand.  Given smoothed 21-point landmarks it:

  1. Detects SNAP events (two-phase: contact → SNAP_READY, release → SNAP)
  2. Classifies the current gesture intent
  3. Drives the logic state machine  (LOCKED → HOVER → ACTIVE)
  4. Computes the transform payload  (anchor, delta, rotation)
  5. Returns a JSON-compatible dict matching the visHand spec

Primary API:
    interpreter = GestureInterpreter(settings)
    payload = interpreter.process(raw_landmarks_or_None, hand_side, timestamp, frame_id)

The filter is managed internally — feed raw MediaPipe landmarks directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from bridge.schema_v1 import ensure_v1_payload
from core.arm_assist import ArmAssistGate
from core.filters import LandmarkFilter
from core.kinematics import BoneLengthClamp, LandmarkKalmanPredictor
from core.safety import ActionSafetyLayer
from core.types import Point3D
from config.settings import Settings
from config.calibration_profile import apply_profile_path
import utils.math_tools as mt
from core.context import GestureContext
from core.gestures import GestureDef, registry


# ---------------------------------------------------------------------------
# String constants (avoid typos in comparisons)
# ---------------------------------------------------------------------------

LOCKED  = "LOCKED"
HOVER   = "HOVER"
ACTIVE  = "ACTIVE"

IDLE          = "IDLE"
EV_NONE        = "NONE"
EV_SNAP        = "SNAP"
EV_DOUBLE_TAP  = "DOUBLE_TAP"
EV_CLAP        = "CLAP"
EV_SWIPE_LEFT  = "SWIPE_LEFT"
EV_SWIPE_RIGHT = "SWIPE_RIGHT"
EV_EMERGENCY_CANCEL = "EMERGENCY_CANCEL"
PINCH_HOLD     = "PINCH_HOLD"
PINCH_DRAG     = "PINCH_DRAG"
SNAP_READY     = "SNAP_READY"
OPEN_PALM      = "OPEN_PALM"
CLOSED_FIST    = "CLOSED_FIST"
GRASP_OPEN     = "GRASP_OPEN"
GRASP_ENTERING = "GRASP_ENTERING"
GRASP_HOLD     = "GRASP_HOLD"
GRASP_DRAG     = "GRASP_DRAG"


# ---------------------------------------------------------------------------
# GestureInterpreter
# ---------------------------------------------------------------------------

class GestureInterpreter:
    """
    Stateful gesture interpreter.  One instance per tracked hand.

    Args:
        settings (Settings): project-wide configuration.
    """

    def __init__(self, settings: Settings):
        self.s = settings
        if not getattr(self.s, "_calibration_profile_applied", False):
            apply_profile_path(self.s)
            setattr(self.s, "_calibration_profile_applied", True)
        self._filter = LandmarkFilter(settings)
        self._clamp = BoneLengthClamp(settings)
        self._predictor = LandmarkKalmanPredictor(settings, tracked_ids=(4, 8, 12))
        self._arm_assist = ArmAssistGate(settings)
        self._safety = ActionSafetyLayer(settings)
        self._event_defs = {g.name: g for g in registry.event_defs()}
        self._reset()

    # ── Main entry point ─────────────────────────────────────────────────────

    def process(
        self,
        raw_landmarks: Optional[List[Point3D]],
        hand_side: str,
        timestamp: float,
        frame_id: int,
        hand_id: Optional[str] = None,
        arm_features: Optional[dict] = None,
        degraded_mode: bool = False,
        input_quality_score: float = 1.0,
        tracking_quality: str = "good",
    ) -> dict:
        """
        Full pipeline: filter → classify → state machine → payload.

        Args:
            raw_landmarks:  21-point list from MediaPipe (un-filtered), or None.
            hand_side:      "LEFT" | "RIGHT"
            timestamp:      Unix timestamp in seconds (float)
            frame_id:       Monotonically increasing frame counter
            arm_features:   Optional dict containing forearm vector and orientation from Pose backend.

        Returns:
            JSON-compatible dict matching the visHand spec.
        """
        dt_ms = self._compute_dt_ms(timestamp)
        if raw_landmarks is None:
            return self._handle_no_hand(
                hand_side,
                timestamp,
                frame_id,
                hand_id=hand_id,
                dt_ms=dt_ms,
                input_quality_score=input_quality_score,
                tracking_quality=tracking_quality,
            )

        # Remap happened in detector, so apply Clamp -> Filter -> Prediction.
        clamped = self._clamp.apply(raw_landmarks)
        filtered = self._filter.apply(clamped, timestamp)
        lm = self._predictor.update_and_predict(filtered, timestamp)

        self._no_hand_frames = 0
        self._has_hand_frames += 1

        arm_view = self._arm_assist.select_arm_view(hand_side, arm_features)

        # Build Context for Evaluators
        af = arm_features or {}
        ctx = GestureContext(
            lm,
            self.s,
            self._anchor_history,
            forearm_vector=arm_view.get("forearm_vector"),
            wrist_orientation_confidence=arm_view.get("arm_confidence", 0.0),
            wrist_body_x=arm_view.get("wrist_body_x", 0.0),
            forearm_depth_delta=arm_view.get("forearm_depth_delta", 0.0),
            arm_confidence=arm_view.get("arm_confidence", 0.0),
        )

        # --- Pipeline ---
        event  = self._detect_events(lm, timestamp, arm_view)
        intent, intent_conf, intent_risk, arbitration, arm_meta = self._classify_intent(
            ctx, timestamp, event, hand_side, arm_view, degraded_mode
        )
        grasp_phase = self._update_grasp_phase(ctx)
        if grasp_phase == GRASP_HOLD:
            intent = PINCH_HOLD
            intent_conf = max(float(intent_conf), 0.75)
            intent_risk = "medium"
        elif grasp_phase == GRASP_DRAG:
            intent = PINCH_DRAG
            intent_conf = max(float(intent_conf), 0.80)
            intent_risk = "medium"
        if arm_meta.get("emergency_cancel", 0):
            event = EV_EMERGENCY_CANCEL
        logic  = self._update_logic(intent)

        anchor, delta, rotation, rotation_euler = self._compute_transform(lm, intent)
        vel       = mt.instant_velocity(self._anchor_history, self.s.anchor_history_size)
        intensity = mt.pinch_intensity(lm)
        event_phase = self._compute_event_phase(event, intent)

        self._logic  = logic
        self._intent = intent
        self._intent_confidence = intent_conf
        self._intent_risk = intent_risk

        return self._build_payload(
            frame_id, timestamp, hand_side,
            logic, intent, self._filter.is_stable,
            anchor, delta, rotation, rotation_euler,
            intensity, vel, event, event_phase,
            intent_conf, intent_risk, arbitration, arm_meta,
            input_quality_score=input_quality_score,
            tracking_quality=tracking_quality,
            emergency_cancel=(event == EV_EMERGENCY_CANCEL),
            grasp_phase=grasp_phase,
            contact_score=ctx.contact_score,
            hand_id=hand_id,
            dt_ms=dt_ms,
            enable_extended_transform=bool(getattr(self.s, "bridge_enable_extended_transform", False)),
            enable_event_phase=bool(getattr(self.s, "bridge_enable_event_phase", False)),
            enable_hand_identity=bool(getattr(self.s, "bridge_enable_hand_identity", False)),
        )

    # ── Internal state helpers ───────────────────────────────────────────────

    def _reset(self):
        """Full state reset — called at construction and when hand is lost."""
        self._logic  = LOCKED
        self._intent = IDLE
        self._intent_buffer: List[str] = []

        self._no_hand_frames  = 0
        self._has_hand_frames = 0

        # Transform
        self._prev_anchor: Optional[Tuple[float, float, float]] = None
        self._anchor_history: List[Tuple[float, float]] = []

        # SNAP two-phase machine
        self._snap_phase: str = "IDLE"        # "IDLE" | "READY"
        self._prev_mid_thumb_dist: Optional[float] = None
        self._prev_idx_thumb_dist: Optional[float] = None
        self._prev_mid_palm_dist: Optional[float] = None
        self._last_snap_time: float = 0.0

        # DOUBLE_TAP
        self._pinch_was_active = False
        self._pinch_onset_times: List[float] = []

        self._filter.reset()
        self._clamp.reset()
        self._predictor.reset()
        self._intent_confidence = 0.0
        self._intent_risk = "low"
        self._last_intent_switch_time: Dict[str, float] = {}
        self._event_cooldowns: Dict[str, float] = {}
        self._event_priority_lock = 0
        self._intent_hold_name = IDLE
        self._intent_hold_count = 0
        self._grasp_phase = GRASP_OPEN
        self._grasp_enter_count = 0
        self._grasp_exit_count = 0
        self._last_timestamp: Optional[float] = None
        self._event_end_pending = False
        self._safety.reset()

    # ── No-hand path ──────────────────────────────────────────────────────────

    def _handle_no_hand(
        self,
        hand_side,
        timestamp,
        frame_id,
        hand_id: Optional[str],
        dt_ms: float,
        input_quality_score: float,
        tracking_quality: str,
    ) -> dict:
        self._no_hand_frames  += 1
        self._has_hand_frames  = 0

        if self._no_hand_frames >= self.s.hover_to_locked_frames:
            self._reset()   # full reset: LOCKED + filter cleared

        return self._build_payload(
            frame_id, timestamp, hand_side,
            self._logic, IDLE, False,
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0, 0.0),
            0.0, 0.0, EV_NONE, self._compute_event_phase(EV_NONE, IDLE), 0.0, "low", [], self._default_arm_meta(hand_side),
            input_quality_score=input_quality_score,
            tracking_quality=tracking_quality,
            emergency_cancel=False,
            grasp_phase=GRASP_OPEN,
            contact_score=0.0,
            hand_id=hand_id,
            dt_ms=dt_ms,
            enable_extended_transform=bool(getattr(self.s, "bridge_enable_extended_transform", False)),
            enable_event_phase=bool(getattr(self.s, "bridge_enable_event_phase", False)),
            enable_hand_identity=bool(getattr(self.s, "bridge_enable_hand_identity", False)),
        )

    # ── Event detection ───────────────────────────────────────────────────────

    def _detect_events(self, lm: List[Point3D], timestamp: float, arm_view: dict) -> str:
        s = self.s
        now_ms_ok = (timestamp - self._last_snap_time) > (s.snap_cooldown_ms / 1000.0)

        # ── SNAP (two-phase composite scoring) ──────────────────────────────
        mid_thumb = mt.normalized_distance(lm[mt.MIDDLE_TIP], lm[mt.THUMB_TIP], lm)
        idx_thumb = mt.normalized_distance(lm[mt.INDEX_TIP], lm[mt.THUMB_TIP], lm)
        mid_palm  = mt.normalized_distance(lm[mt.MIDDLE_TIP], lm[0], lm) # 0 is WRIST

        if self._snap_phase == "IDLE":
            if mid_thumb < s.snap_ready_threshold:
                self._snap_phase = "READY"

        elif self._snap_phase == "READY":
            if mid_thumb > s.snap_trigger_threshold:
                # Separation happened — check composite speeds
                if self._prev_mid_thumb_dist is not None:
                    sep_vel = mid_thumb - self._prev_mid_thumb_dist                # Expect > 0
                    idx_crash_vel = idx_thumb - self._prev_idx_thumb_dist          # Expect < 0
                    palm_crash_vel = mid_palm - self._prev_mid_palm_dist           # Expect < 0
                    
                    score = 0.0
                    if sep_vel > 0:       score += sep_vel * 10.0
                    if idx_crash_vel < 0: score += (-idx_crash_vel) * 10.0
                    if palm_crash_vel < 0: score += (-palm_crash_vel) * 10.0
                    
                    if score > s.snap_score_threshold and now_ms_ok and self._event_allowed(EV_SNAP, timestamp):
                        self._last_snap_time = timestamp
                        self._snap_phase = "IDLE"
                        self._prev_mid_thumb_dist = mid_thumb
                        self._prev_idx_thumb_dist = idx_thumb
                        self._prev_mid_palm_dist = mid_palm
                        self._register_event(EV_SNAP, timestamp)
                        return EV_SNAP
                        
                self._snap_phase = "IDLE"
            elif mid_thumb > s.snap_ready_threshold * 1.6:
                # Drifted back without a snap
                self._snap_phase = "IDLE"

        self._prev_mid_thumb_dist = mid_thumb
        self._prev_idx_thumb_dist = idx_thumb
        self._prev_mid_palm_dist = mid_palm

        # ── DOUBLE_TAP ──────────────────────────────────────────────────────
        pinch_norm = mt.normalized_distance(lm[mt.THUMB_TIP], lm[mt.INDEX_TIP], lm)
        is_pinch   = pinch_norm < s.pinch_threshold

        if is_pinch and not self._pinch_was_active:
            self._pinch_onset_times.append(timestamp)
            cutoff = timestamp - s.double_tap_window_ms / 1000.0
            self._pinch_onset_times = [t for t in self._pinch_onset_times if t > cutoff]

            if len(self._pinch_onset_times) >= 2:
                self._pinch_onset_times.clear()
                self._pinch_was_active = True
                if self._event_allowed(EV_DOUBLE_TAP, timestamp):
                    self._register_event(EV_DOUBLE_TAP, timestamp)
                    return EV_DOUBLE_TAP

        self._pinch_was_active = is_pinch

        if self._is_swipe_event():
            dx = self._anchor_history[-1][0] - self._anchor_history[-3][0]
            if abs(dx) < 0.01 and arm_view.get("forearm_vector") is not None:
                fx = float(arm_view["forearm_vector"][0])
                event = EV_SWIPE_LEFT if fx < 0 else EV_SWIPE_RIGHT
            else:
                event = EV_SWIPE_LEFT if dx < 0 else EV_SWIPE_RIGHT
            if self._event_allowed(event, timestamp):
                self._register_event(event, timestamp)
                return event
        return EV_NONE

    # ── Intent classification ─────────────────────────────────────────────────

    def _classify_intent(
        self,
        ctx: GestureContext,
        timestamp: float,
        event: str,
        hand_side: str,
        arm_view: dict,
        degraded_mode: bool,
    ) -> Tuple[str, float, str, List[dict], dict]:
        s = self.s
        arm_meta = self._default_arm_meta(hand_side)
        arm_meta["arm_confidence"] = round(float(arm_view.get("arm_confidence", 0.0)), 4)

        # SNAP_READY has visual priority — let the user see the ready state
        if self._snap_phase == "READY":
            return "SNAP_READY", 1.0, "high", [{"name": "SNAP_READY", "score": 1.0, "group": "event_ready"}], arm_meta

        if event != EV_NONE:
            self._event_priority_lock = max(self._event_priority_lock, s.event_priority_lock_frames)

        if self._event_priority_lock > 0:
            self._event_priority_lock -= 1
            return self._intent, self._intent_confidence, self._intent_risk, [], arm_meta

        winners = self._resolve_mutex_candidates(ctx)
        margin_before = self._arm_assist.score_margin(winners)
        arm_meta["margin_before"] = round(margin_before, 4)
        winners, applied = self._arm_assist.apply(
            winners,
            arm_view,
            hand_side=hand_side,
            low_conf_threshold=s.arm_assist_low_conf_threshold,
            margin_threshold=s.arm_assist_margin_threshold,
        )
        arm_meta["assist_applied"] = int(applied)
        arm_meta["z_hint"] = self._arm_assist.infer_z_hint(arm_view)
        arm_meta["handedness_fused"] = self._arm_assist.fuse_handedness(hand_side, arm_view)

        arbitration = [
            {"name": g.name, "score": round(float(sc), 4), "group": g.mutex_group}
            for g, sc in winners[:5]
        ]
        if not winners:
            raw_intent = IDLE
            raw_score = 0.0
            raw_risk = "low"
        else:
            g, raw_score = winners[0]
            min_conf = s.intent_min_confidence + (s.degraded_confidence_boost if degraded_mode else 0.0)
            raw_intent = g.name if raw_score >= min_conf else IDLE
            raw_risk = g.risk_level if raw_intent != IDLE else "low"

        raw_intent = self._safety.filter_high_risk(raw_intent, degraded_mode)
        if self._safety.check_emergency_cancel(raw_intent):
            arm_meta["emergency_cancel"] = 1
            return IDLE, 1.0, "low", arbitration, arm_meta

        if raw_intent != IDLE:
            g = registry.get(raw_intent)
            if g and not self._intent_cooldown_ready(g, timestamp):
                raw_intent = self._intent
                raw_score = self._intent_confidence
                raw_risk = self._intent_risk

        # --- Debounce Logic ---
        self._intent_buffer.append(raw_intent)
        if len(self._intent_buffer) > s.intent_debounce_frames:
            self._intent_buffer.pop(0)

        # ONLY switch if the new intent holds for N frames, otherwise keep previous
        # We check if all items in buffer are the same 
        if all(i == raw_intent for i in self._intent_buffer):
            self._intent_hold_count = self._intent_hold_count + 1 if self._intent_hold_name == raw_intent else 1
            self._intent_hold_name = raw_intent
            if self._intent_hold_count >= s.intent_hold_frames:
                if raw_intent != self._intent and raw_intent != IDLE:
                    g = registry.get(raw_intent)
                    if g:
                        self._last_intent_switch_time[g.name] = timestamp
                return raw_intent, raw_score, raw_risk, arbitration, arm_meta
            return self._intent, self._intent_confidence, self._intent_risk, arbitration, arm_meta

        self._intent_hold_name = raw_intent
        self._intent_hold_count = 1
        return self._intent, self._intent_confidence, self._intent_risk, arbitration, arm_meta

    def _resolve_mutex_candidates(self, ctx: GestureContext) -> List[Tuple[GestureDef, float]]:
        group_best: Dict[str, Tuple[GestureDef, float]] = {}
        for g in registry.enabled_intents():
            score = g.evaluator(ctx)
            if score <= 0.0:
                continue
            prev = group_best.get(g.mutex_group)
            if prev is None or score > prev[1] or (score == prev[1] and g.priority > prev[0].priority):
                group_best[g.mutex_group] = (g, score)

        winners = list(group_best.values())
        winners.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        return winners

    def _update_grasp_phase(self, ctx: GestureContext) -> str:
        dist = float(ctx.pinch_distance)
        vel = float(ctx.velocity)
        enter_th = float(self.s.pinch_threshold)
        exit_th = float(self.s.pinch_release_threshold)
        enter_n = max(1, int(getattr(self.s, "grasp_enter_min_frames", 2)))
        exit_n = max(1, int(getattr(self.s, "grasp_exit_min_frames", 2)))

        if self._grasp_phase in (GRASP_OPEN, GRASP_ENTERING):
            if dist < enter_th:
                self._grasp_enter_count += 1
                if self._grasp_enter_count >= enter_n:
                    self._grasp_phase = GRASP_DRAG if vel > self.s.drag_start_velocity else GRASP_HOLD
                else:
                    self._grasp_phase = GRASP_ENTERING
            else:
                self._grasp_enter_count = 0
                self._grasp_phase = GRASP_OPEN
                self._grasp_exit_count = 0
            return self._grasp_phase

        if self._grasp_phase in (GRASP_HOLD, GRASP_DRAG):
            if dist > exit_th:
                self._grasp_exit_count += 1
                if self._grasp_exit_count >= exit_n:
                    self._grasp_phase = GRASP_OPEN
                    self._grasp_enter_count = 0
            else:
                self._grasp_exit_count = 0
                self._grasp_phase = GRASP_DRAG if vel > self.s.drag_start_velocity else GRASP_HOLD
            return self._grasp_phase

        self._grasp_phase = GRASP_OPEN
        self._grasp_enter_count = 0
        self._grasp_exit_count = 0
        return self._grasp_phase

    @staticmethod
    def _default_arm_meta(hand_side: str) -> dict:
        return {
            "assist_applied": 0,
            "margin_before": 1.0,
            "z_hint": "neutral",
            "handedness_fused": hand_side,
            "arm_confidence": 0.0,
            "emergency_cancel": 0,
        }

    def _event_allowed(self, event_name: str, timestamp: float) -> bool:
        if event_name == EV_NONE:
            return False
        g = self._event_defs.get(event_name)
        default_cd = 0.0
        if event_name in (EV_SWIPE_LEFT, EV_SWIPE_RIGHT):
            default_cd = self.s.swipe_event_cooldown_ms
        if event_name == EV_SNAP:
            default_cd = self.s.snap_cooldown_ms
        cooldown = (g.cooldown_ms if g else default_cd) / 1000.0
        last = self._event_cooldowns.get(event_name, 0.0)
        return (timestamp - last) >= cooldown

    def _register_event(self, event_name: str, timestamp: float):
        self._event_cooldowns[event_name] = timestamp

    def _intent_cooldown_ready(self, gesture: GestureDef, timestamp: float) -> bool:
        if gesture.cooldown_ms <= 0:
            return True
        last = self._last_intent_switch_time.get(gesture.name, 0.0)
        return (timestamp - last) >= (gesture.cooldown_ms / 1000.0)

    def _is_swipe_event(self) -> bool:
        if len(self._anchor_history) < 3:
            return False
        vel = mt.instant_velocity(self._anchor_history, 5)
        if vel <= self.s.swipe_velocity_threshold:
            return False
        dx = self._anchor_history[-1][0] - self._anchor_history[-3][0]
        dy = self._anchor_history[-1][1] - self._anchor_history[-3][1]
        return abs(dx) > abs(dy)

    def _compute_dt_ms(self, timestamp: float) -> float:
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return 0.0
        dt_ms = max(0.0, (float(timestamp) - float(self._last_timestamp)) * 1000.0)
        self._last_timestamp = timestamp
        return dt_ms

    def _compute_event_phase(self, event: str, intent: str) -> str:
        if event != EV_NONE:
            self._event_end_pending = True
            return "START"
        if self._event_end_pending:
            self._event_end_pending = False
            return "END"
        if intent in (PINCH_HOLD, PINCH_DRAG):
            return "UPDATE"
        return "NONE"

    # ── Logic state machine ───────────────────────────────────────────────────

    def _update_logic(self, intent: str) -> str:
        s = self.s
        current = self._logic

        if current == LOCKED:
            if self._has_hand_frames >= s.locked_to_hover_frames:
                return HOVER
            return LOCKED

        if current == HOVER:
            if intent in ("PINCH_HOLD", "PINCH_DRAG", "CLOSED_FIST"):
                return ACTIVE
            return HOVER

        if current == ACTIVE:
            if intent in (IDLE, "OPEN_PALM"):
                return HOVER
            return ACTIVE

        return current

    # ── Transform computation ─────────────────────────────────────────────────

    def _compute_transform(
        self, lm: List[Point3D], intent: str
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, Tuple[float, float, float]]:

        # Choose anchor based on gesture type
        if intent in (PINCH_HOLD, PINCH_DRAG, SNAP_READY):
            anchor = mt.pinch_anchor(lm)
        else:
            anchor = mt.palm_anchor(lm)

        ax, ay, az = anchor[0], anchor[1], anchor[2]

        # Maintain anchor history for velocity calculation
        self._anchor_history.append((ax, ay))
        if len(self._anchor_history) > self.s.anchor_history_size:
            self._anchor_history.pop(0)

        delta = mt.compute_delta3d(self._prev_anchor, (ax, ay, az))
        self._prev_anchor = (ax, ay, az)

        rotation = mt.palm_roll_angle(lm)
        rotation_euler = mt.palm_euler_angles(lm)

        return anchor, delta, rotation, rotation_euler

    # ── Payload builder ───────────────────────────────────────────────────────

    @staticmethod
    def _build_payload(
        frame_id, timestamp, hand_side,
        logic, intent, is_stable,
        anchor, delta, rotation, rotation_euler,
        intensity, velocity, event, event_phase, intent_confidence, intent_risk, arbitration, arm_meta,
        input_quality_score, tracking_quality, emergency_cancel, grasp_phase, contact_score,
        hand_id, dt_ms, enable_extended_transform, enable_event_phase, enable_hand_identity,
    ) -> dict:
        payload = {
            "header": {
                "frame_id":  frame_id,
                "timestamp": round(timestamp, 3),
                "hand_side": hand_side,
            },
            "state": {
                "logic":     logic,
                "intent":    intent,
                "is_stable": is_stable,
                "intent_confidence": round(float(intent_confidence), 3),
                "intent_risk": intent_risk,
                "arbitration": arbitration,
                "arm_assist_applied": arm_meta.get("assist_applied", 0),
                "handedness_fused": arm_meta.get("handedness_fused", hand_side),
                "z_hint": arm_meta.get("z_hint", "neutral"),
                "arm_confidence": round(float(arm_meta.get("arm_confidence", 0.0)), 3),
                "arm_margin_before": round(float(arm_meta.get("margin_before", 1.0)), 4),
                "input_quality_score": round(float(input_quality_score), 3),
                "tracking_quality": str(tracking_quality),
                "grasp_phase": str(grasp_phase),
                "contact_score": round(float(contact_score), 3),
            },
            "transform": {
                "anchor": {
                    "x": round(anchor[0], 4),
                    "y": round(anchor[1], 4),
                    "z": round(anchor[2], 4),
                },
                "delta": {
                    "dx": round(delta[0], 5),
                    "dy": round(delta[1], 5),
                },
                "rotation": round(rotation, 2),
            },
            "dynamics": {
                "intensity": round(intensity, 3),
                "velocity":  round(velocity, 5),
                "event":     event,
                "emergency_cancel": bool(emergency_cancel),
            },
        }
        if enable_extended_transform:
            payload["header"]["dt_ms"] = round(float(dt_ms), 3)
            payload["header"]["coord_space"] = "camera_normalized_right_handed"
            payload["header"]["axis_convention"] = {
                "x": "right_positive",
                "y": "down_positive",
                "z": "away_from_camera_positive",
            }
            payload["header"]["unit_scale"] = "normalized_0_1_xy"
            payload["transform"]["delta"]["dz"] = round(delta[2], 5)
            payload["transform"]["rotation_euler"] = {
                "roll": round(rotation_euler[0], 3),
                "pitch": round(rotation_euler[1], 3),
                "yaw": round(rotation_euler[2], 3),
            }
        if enable_event_phase:
            payload["dynamics"]["event_phase"] = str(event_phase)
        if enable_hand_identity and hand_id:
            payload["header"]["hand_id"] = str(hand_id)
        return ensure_v1_payload(payload, hand_side=hand_side)
