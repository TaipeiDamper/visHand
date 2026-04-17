"""
visHand — Configuration Layer
==============================
All tunable thresholds and runtime options live here.
Override any field when constructing Settings():

    s = Settings(camera_index=1, pinch_threshold=0.18)
"""
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── Camera ────────────────────────────────────────────────────────────────
    camera_index: int = 0          # 0 = built-in webcam; change for external cam
    target_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    camera_read_timeout_ms: int = 80

    # ── MediaPipe Hands ────────────────────────────────────────────────────────
    max_hands: int = 2             # 2 for dual-hand mode
    adaptive_max_hands: bool = True
    dual_hand_probe_interval: int = 5
    dual_hand_hold_frames: int = 120
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.60
    hand_landmarker_model_path: str = ""
    hand_landmarker_num_threads: int = 2
    preferred_detector_backend: str = "legacy"  # "tasks" | "legacy"
    use_gpu_delegate: bool = False  # enable GPU delegate for Tasks backend
    enable_pose_assist: bool = False
    pose_landmarker_model_path: str = ""
    pose_landmarker_num_threads: int = 2
    pose_min_detection_confidence: float = 0.55

    # ── ROI Tracking / Remapping ───────────────────────────────────────────────
    enable_roi_tracking: bool = True
    roi_input_size: int = 256
    roi_box_ratio: float = 0.46
    roi_confidence_fallback: float = 0.30
    roi_max_miss_frames: int = 3

    # ── One Euro Filter ────────────────────────────────────────────────────────
    one_euro_min_cutoff: float = 1.0    # lower → smoother at rest, more lag
    one_euro_beta: float = 0.015        # higher → faster response on fast motion
    one_euro_d_cutoff: float = 1.0
    stability_warmup_frames: int = 15   # frames before is_stable = True
    use_vectorized_filter: bool = True

    # ── Kinematics / Prediction ────────────────────────────────────────────────
    clamp_calibration_frames: int = 30
    clamp_tolerance_ratio: float = 0.15
    kalman_predict_horizon_ms: float = 16.0
    kalman_alpha: float = 0.65
    kalman_beta: float = 0.20
    kalman_residual_reset: float = 0.12

    # ── Gesture Thresholds (normalized by palm width) ──────────────────────────
    # Pinch (thumb tip ↔ index tip)
    pinch_threshold: float = 0.20       # enter PINCH below this
    pinch_release_threshold: float = 0.28  # exit PINCH above this

    # Snap two-phase detection (middle tip ↔ thumb tip)
    snap_ready_threshold: float = 0.20  # contact zone → SNAP_READY
    snap_trigger_threshold: float = 0.28  # separation needed to fire SNAP
    snap_score_threshold: float = 0.25  # composite score (separation + hits) to confirm snap

    # Finger curl (open palm / fist)
    fist_curl_threshold: float = 0.45   # all fingers above this → FIST
    ok_sign_threshold: float = 0.14
    index_point_lateral_threshold: float = 0.09
    palm_tilt_angle_threshold_deg: float = 24.0

    # ── Motion Thresholds ──────────────────────────────────────────────────────
    swipe_velocity_threshold: float = 0.040   # fast lateral motion → SWIPE
    drag_start_velocity: float = 0.008        # pinch + this velocity → PINCH_DRAG
    anchor_history_size: int = 8              # frames kept for velocity smoothing

    # ── State Machine Timing ───────────────────────────────────────────────────
    hover_to_locked_frames: int = 12    # consecutive no-hand frames → LOCKED
    locked_to_hover_frames: int = 3     # consecutive hand frames → HOVER

    # ── Event Timing ──────────────────────────────────────────────────────────
    snap_cooldown_ms: float = 100.0        # min ms between SNAP events
    double_tap_window_ms: float = 350.0   # two pinch onsets within this → DOUBLE_TAP
    swipe_event_cooldown_ms: float = 180.0

    # ── Intent Arbitration ────────────────────────────────────────────────────
    intent_hold_frames: int = 2
    event_priority_lock_frames: int = 2
    arm_assist_low_conf_threshold: float = 0.65
    arm_assist_margin_threshold: float = 0.10
    arm_assist_weight_default: float = 0.2
    arm_confidence_min: float = 0.55

    # ── Output / Debug ────────────────────────────────────────────────────────
    output_mode: str = "library"   # "library" (return dict) | "stdout" (print JSON)
    debug_window: bool = False     # True = open OpenCV window (used by demo.py)
    bridge_transport: str = "stdout"  # "stdout" | "ws" | "pipe"
    bridge_ws_host: str = "127.0.0.1"
    bridge_ws_port: int = 9876
    bridge_ws_max_clients: int = 4
    bridge_ws_fallback_stdout: bool = True
    bridge_pipe_name: str = "vishand"
    bridge_pipe_reconnect_ms: int = 1200
    bridge_pipe_fallback_stdout: bool = True
    bridge_enable_extended_transform: bool = False
    bridge_enable_event_phase: bool = False
    bridge_enable_hand_identity: bool = False

    # ── Intent Engine ─────────────────────────────────────────────────────────
    intent_min_confidence: float = 0.6     # 最低信心門檻，低於此則為 IDLE
    intent_debounce_frames: int = 2        # 真實 Intent 必須連續勝出多少幀才切換

    # ── Runtime Optimization / Telemetry ───────────────────────────────────────
    telemetry_buffer_size: int = 300
    enable_adaptive_skipping: bool = True
    adaptive_skip_velocity_threshold: float = 0.005
    quality_brightness_min: float = 0.12
    quality_blur_var_min: float = 45.0
    quality_overexposed_ratio_max: float = 0.25
    quality_degrade_threshold: float = 0.50
    quality_poor_threshold: float = 0.32
    degraded_confidence_boost: float = 0.15
    degraded_low_risk_only: bool = True
    emergency_cancel_hold_frames: int = 5
    grasp_enter_min_frames: int = 2
    grasp_exit_min_frames: int = 2
    enable_hand_association: bool = True
    association_max_dist_norm: float = 0.35
    association_handedness_mismatch_cost: float = 0.45
    calibration_profile_path: str = ""

