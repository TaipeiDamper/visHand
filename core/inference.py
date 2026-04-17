from __future__ import annotations

import copy
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2

from core.backends.hands import build_hands_backend
from core.backends.pose import build_pose_backend, estimate_arm_features
from core.quality_gate import QualityGate
from core.types import FramePacket, HandLandmarkResult, InferencePacket, Point3D


class InferenceWorker:
    def __init__(self, settings, frame_supplier: Callable[[], Optional[FramePacket]]):
        self.settings = settings
        self._frame_supplier = frame_supplier

        self._backend_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._hint_lock = threading.Lock()

        self._active_max_hands = settings.max_hands if not settings.adaptive_max_hands else 1
        self._backend = build_hands_backend(settings, self._active_max_hands)
        self._probe_backend = None
        self._pose_backend = build_pose_backend(settings) if settings.enable_pose_assist else None
        self._quality_gate = QualityGate(settings)

        self._result_slot: Optional[InferencePacket] = None
        self._last_inferred_id = -1
        self._skip_toggle = False
        self._tracking_anchor: Optional[Tuple[float, float]] = None
        self._roi_miss_frames = 0
        self._hands_peak_hold = 0
        self._runtime_hint = {"logic": "LOCKED", "intent": "IDLE", "velocity": 0.0}
        self._slot_refs: List[Optional[Dict[str, float]]] = [None for _ in range(max(1, int(settings.max_hands)))]
        self._last_quality_cached = self._quality_gate.evaluate(np.zeros((64, 64, 3), dtype=np.uint8))

    def run(self, stop_event: threading.Event):
        while not stop_event.is_set():
            frame_pkt = self._frame_supplier()
            if frame_pkt is None:
                time.sleep(0.001)
                continue
            if frame_pkt.frame_id <= self._last_inferred_id:
                time.sleep(0.001)
                continue

            if self._should_skip():
                self._publish(
                    frame_pkt,
                    mp_ms=0.0,
                    pose_ms=0.0,
                    mode="SKIP",
                    skipped=True,
                    results=self._last_results(),
                    arm_features=self._last_arm_features(),
                    input_quality_score=self._last_quality_score(),
                    tracking_quality=self._last_tracking_quality(),
                    degraded_mode=self._last_degraded_mode(),
                )
                self._last_inferred_id = frame_pkt.frame_id
                continue

            mode = "GLOBAL"
            target_frame = frame_pkt.frame
            remap_meta = None
            quality = self._last_quality_cached if frame_pkt.frame_id % 4 != 0 else self._quality_gate.evaluate(frame_pkt.frame)
            self._last_quality_cached = quality

            if self.settings.enable_roi_tracking and self.settings.max_hands == 1 and self._tracking_anchor is not None:
                roi = self._compute_roi(frame_pkt.frame.shape[:2], self._tracking_anchor)
                if roi is not None:
                    mode = "ROI"
                    target_frame, remap_meta = self._letterbox_roi(frame_pkt.frame, roi)

            t0 = time.perf_counter()
            with self._backend_lock:
                results = self._backend.detect(target_frame)
            mp_ms = (time.perf_counter() - t0) * 1000.0

            if mode == "ROI":
                if results:
                    results = [self._remap_result(r, remap_meta, frame_pkt.frame.shape[:2]) for r in results]
                if (not results) or max(r.confidence for r in results) < self.settings.roi_confidence_fallback:
                    self._roi_miss_frames += 1
                    if self._roi_miss_frames >= self.settings.roi_max_miss_frames:
                        mode = "GLOBAL_FALLBACK"
                        self._roi_miss_frames = 0
                        with self._backend_lock:
                            t0 = time.perf_counter()
                            results = self._backend.detect(frame_pkt.frame)
                            mp_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    self._roi_miss_frames = 0

            results = self._associate_results(results)
            self._update_tracking_anchor(results)
            arm_features, pose_ms = estimate_arm_features(self._pose_backend, frame_pkt.frame, results)
            self._maybe_probe_dual_hand(frame_pkt)
            self._publish(
                frame_pkt,
                mp_ms=mp_ms,
                pose_ms=pose_ms,
                mode=mode,
                skipped=False,
                results=results,
                arm_features=arm_features,
                input_quality_score=quality.overall_score,
                tracking_quality=quality.tracking_quality,
                degraded_mode=quality.degraded_mode,
            )
            self._last_inferred_id = frame_pkt.frame_id

    def update_runtime_hint(self, logic: str, intent: str, velocity: float):
        with self._hint_lock:
            self._runtime_hint = {"logic": logic, "intent": intent, "velocity": float(velocity)}

    def extract_landmarks(self, frame: np.ndarray) -> Optional[List[HandLandmarkResult]]:
        with self._backend_lock:
            return self._backend.detect(frame)

    def get_latest_packet(self, min_frame_id: int = -1) -> Optional[InferencePacket]:
        with self._result_lock:
            if self._result_slot is None or self._result_slot.frame_id <= min_frame_id:
                return None
            pkt = self._result_slot
        return InferencePacket(
            frame=pkt.frame.copy(),
            frame_id=pkt.frame_id,
            t_capture=pkt.t_capture,
            t_infer_done=pkt.t_infer_done,
            capture_ms=pkt.capture_ms,
            mp_ms=pkt.mp_ms,
            pose_ms=pkt.pose_ms,
            mode=pkt.mode,
            skipped=pkt.skipped,
            results=copy.deepcopy(pkt.results) if pkt.results else None,
            arm_features=copy.deepcopy(pkt.arm_features) if pkt.arm_features else None,
            input_quality_score=pkt.input_quality_score,
            tracking_quality=pkt.tracking_quality,
            degraded_mode=pkt.degraded_mode,
        )

    def close(self):
        with self._backend_lock:
            self._backend.close()
            if self._probe_backend is not None:
                self._probe_backend.close()
        if self._pose_backend is not None:
            self._pose_backend.close()

    def _should_skip(self) -> bool:
        if not self.settings.enable_adaptive_skipping:
            return False
        with self._hint_lock:
            hint = dict(self._runtime_hint)
        static_logic = hint["logic"] == "LOCKED" or (hint["logic"] == "HOVER" and hint["intent"] == "IDLE")
        if static_logic and hint["velocity"] < self.settings.adaptive_skip_velocity_threshold:
            self._skip_toggle = not self._skip_toggle
            return self._skip_toggle
        self._skip_toggle = False
        return False

    def _last_results(self) -> Optional[List[HandLandmarkResult]]:
        with self._result_lock:
            if self._result_slot is None:
                return None
            return copy.deepcopy(self._result_slot.results) if self._result_slot.results else None

    def _last_arm_features(self) -> Optional[dict]:
        with self._result_lock:
            if self._result_slot is None:
                return None
            return copy.deepcopy(self._result_slot.arm_features) if self._result_slot.arm_features else None

    def _publish(
        self,
        frame_pkt: FramePacket,
        mp_ms: float,
        pose_ms: float,
        mode: str,
        skipped: bool,
        results: Optional[List[HandLandmarkResult]],
        arm_features: Optional[dict],
        input_quality_score: float,
        tracking_quality: str,
        degraded_mode: bool,
    ):
        out = InferencePacket(
            frame=frame_pkt.frame,
            frame_id=frame_pkt.frame_id,
            t_capture=frame_pkt.t_capture,
            t_infer_done=time.time(),
            capture_ms=frame_pkt.capture_ms,
            mp_ms=mp_ms,
            pose_ms=pose_ms,
            mode=mode,
            skipped=skipped,
            results=results,
            arm_features=arm_features,
            input_quality_score=float(input_quality_score),
            tracking_quality=str(tracking_quality),
            degraded_mode=bool(degraded_mode),
        )
        with self._result_lock:
            self._result_slot = out

    def _last_quality_score(self) -> float:
        with self._result_lock:
            if self._result_slot is None:
                return 1.0
            return float(self._result_slot.input_quality_score)

    def _last_tracking_quality(self) -> str:
        with self._result_lock:
            if self._result_slot is None:
                return "good"
            return str(self._result_slot.tracking_quality)

    def _last_degraded_mode(self) -> bool:
        with self._result_lock:
            if self._result_slot is None:
                return False
            return bool(self._result_slot.degraded_mode)

    def _compute_roi(self, shape_hw: Tuple[int, int], anchor_xy: Tuple[float, float]) -> Optional[Tuple[int, int, int, int]]:
        h, w = shape_hw
        cx = int(anchor_xy[0] * w)
        cy = int(anchor_xy[1] * h)
        size = int(min(h, w) * self.settings.roi_box_ratio)
        size = max(96, min(size, min(h, w)))
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        return x1, y1, x2, y2

    def _letterbox_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        target = int(self.settings.roi_input_size)
        ch, cw = crop.shape[:2]
        scale = min(target / max(cw, 1), target / max(ch, 1))
        rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target, target, 3), dtype=frame.dtype)
        ox = (target - rw) // 2
        oy = (target - rh) // 2
        canvas[oy:oy + rh, ox:ox + rw] = resized
        meta = {"x1": x1, "y1": y1, "rw": rw, "rh": rh, "ox": ox, "oy": oy, "target": target}
        return canvas, meta

    def _remap_result(self, result: HandLandmarkResult, meta: dict, full_hw: Tuple[int, int]) -> HandLandmarkResult:
        h, w = full_hw
        x1, y1 = meta["x1"], meta["y1"]
        rw, rh = meta["rw"], meta["rh"]
        ox, oy = meta["ox"], meta["oy"]
        target = meta["target"]
        scale_x = rw / float(target)
        scale_y = rh / float(target)

        remapped: List[Point3D] = []
        raw = landmark_pb2.NormalizedLandmarkList()
        for lm in result.landmarks:
            px = (lm.x * target - ox) / max(scale_x, 1e-6)
            py = (lm.y * target - oy) / max(scale_y, 1e-6)
            gx = (x1 + px) / float(w)
            gy = (y1 + py) / float(h)
            gx = min(max(gx, 0.0), 1.0)
            gy = min(max(gy, 0.0), 1.0)
            remapped.append(Point3D(gx, gy, lm.z))
            raw.landmark.add(x=gx, y=gy, z=lm.z)

        return HandLandmarkResult(
            landmarks=remapped,
            hand_side=result.hand_side,
            raw_landmarks=raw,
            confidence=result.confidence,
            track_id=result.track_id,
        )

    def _update_tracking_anchor(self, results: Optional[List[HandLandmarkResult]]) -> None:
        """Update the tracking anchor based on detection results."""
        if not results:
            self._tracking_anchor = None
            return
        # In multi-hand mode, anchor follows slot-0 if possible
        target_result = results[0]
        for r in results:
            if r.track_id == "slot-0":
                target_result = r
                break
        wrist = target_result.landmarks[0]
        self._tracking_anchor = (wrist.x, wrist.y)

    @staticmethod
    def _result_wrist(result: HandLandmarkResult) -> Tuple[float, float]:
        wrist = result.landmarks[0]
        return float(wrist.x), float(wrist.y)

    def _association_cost(self, ref: dict, result: HandLandmarkResult) -> float:
        rx, ry = self._result_wrist(result)
        dx = rx - float(ref.get("x", 0.0))
        dy = ry - float(ref.get("y", 0.0))
        dist = float(np.sqrt(dx * dx + dy * dy))
        mismatch = 0.0
        if str(ref.get("hand_side", "")).upper() and str(ref.get("hand_side", "")).upper() != str(result.hand_side).upper():
            mismatch = float(self.settings.association_handedness_mismatch_cost)
        return dist + mismatch

    def _associate_results(self, results: Optional[List[HandLandmarkResult]]) -> Optional[List[HandLandmarkResult]]:
        if not results:
            self._slot_refs = [None for _ in self._slot_refs]
            return results
        if not getattr(self.settings, "enable_hand_association", True):
            for idx, result in enumerate(results):
                result.track_id = f"slot-{idx}"
            return results

        max_slots = len(self._slot_refs)
        remaining = list(results)
        assigned: List[Optional[HandLandmarkResult]] = [None for _ in range(max_slots)]
        max_dist = float(getattr(self.settings, "association_max_dist_norm", 0.35))

        # First pass: match known slots.
        for slot, ref in enumerate(self._slot_refs):
            if ref is None or not remaining:
                continue
            best_i = -1
            best_cost = 1e9
            best_dist = 1e9
            for i, r in enumerate(remaining):
                cost = self._association_cost(ref, r)
                rx, ry = self._result_wrist(r)
                dist = float(np.sqrt((rx - float(ref.get("x", 0.0))) ** 2 + (ry - float(ref.get("y", 0.0))) ** 2))
                if cost < best_cost:
                    best_cost = cost
                    best_dist = dist
                    best_i = i
            if best_i >= 0 and best_dist <= max_dist:
                assigned[slot] = remaining.pop(best_i)

        # Second pass: fill empty slots by confidence.
        remaining.sort(key=lambda r: float(r.confidence), reverse=True)
        for slot in range(max_slots):
            if assigned[slot] is None and remaining:
                assigned[slot] = remaining.pop(0)

        # Refresh slot references.
        for slot, result in enumerate(assigned):
            if result is None:
                self._slot_refs[slot] = None
                continue
            result.track_id = f"slot-{slot}"
            wx, wy = self._result_wrist(result)
            self._slot_refs[slot] = {"x": wx, "y": wy, "hand_side": str(result.hand_side)}

        return [r for r in assigned if r is not None]

    def _maybe_probe_dual_hand(self, frame_pkt: FramePacket):
        if not self.settings.adaptive_max_hands:
            return
        if self._active_max_hands >= self.settings.max_hands:
            return
        if frame_pkt.frame_id % max(1, self.settings.dual_hand_probe_interval) != 0:
            return

        if self._probe_backend is None:
            self._probe_backend = build_hands_backend(self.settings, self.settings.max_hands)

        probe_results = self._probe_backend.detect(frame_pkt.frame)
        if probe_results and len(probe_results) >= 2:
            self._hands_peak_hold = self.settings.dual_hand_hold_frames
            with self._backend_lock:
                self._backend.close()
                self._backend = build_hands_backend(self.settings, self.settings.max_hands)
                self._active_max_hands = self.settings.max_hands
        elif self._hands_peak_hold > 0:
            self._hands_peak_hold -= 1
        elif self._active_max_hands != 1:
            with self._backend_lock:
                self._backend.close()
                self._backend = build_hands_backend(self.settings, 1)
                self._active_max_hands = 1
