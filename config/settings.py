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

    # ── MediaPipe Hands ────────────────────────────────────────────────────────
    max_hands: int = 1             # 1 for single-hand mode (extend to 2 later)
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.60

    # ── One Euro Filter ────────────────────────────────────────────────────────
    one_euro_min_cutoff: float = 1.0    # lower → smoother at rest, more lag
    one_euro_beta: float = 0.007        # higher → faster response on fast motion
    one_euro_d_cutoff: float = 1.0
    stability_warmup_frames: int = 15   # frames before is_stable = True

    # ── Gesture Thresholds (normalized by palm width) ──────────────────────────
    # Pinch (thumb tip ↔ index tip)
    pinch_threshold: float = 0.20       # enter PINCH below this
    pinch_release_threshold: float = 0.28  # exit PINCH above this

    # Snap two-phase detection (middle tip ↔ thumb tip)
    snap_ready_threshold: float = 0.22  # contact zone → SNAP_READY
    snap_trigger_threshold: float = 0.45  # separation needed to fire SNAP
    snap_velocity_threshold: float = 0.05  # min per-frame dist change to confirm snap

    # Finger curl (open palm / fist)
    fist_curl_threshold: float = 0.60   # all fingers above this → FIST

    # ── Motion Thresholds ──────────────────────────────────────────────────────
    swipe_velocity_threshold: float = 0.040   # fast lateral motion → SWIPE
    drag_start_velocity: float = 0.008        # pinch + this velocity → PINCH_DRAG
    anchor_history_size: int = 8              # frames kept for velocity smoothing

    # ── State Machine Timing ───────────────────────────────────────────────────
    hover_to_locked_frames: int = 12    # consecutive no-hand frames → LOCKED
    locked_to_hover_frames: int = 3     # consecutive hand frames → HOVER

    # ── Event Timing ──────────────────────────────────────────────────────────
    snap_cooldown_ms: float = 600.0        # min ms between SNAP events
    double_tap_window_ms: float = 350.0   # two pinch onsets within this → DOUBLE_TAP

    # ── Output / Debug ────────────────────────────────────────────────────────
    output_mode: str = "library"   # "library" (return dict) | "stdout" (print JSON)
    debug_window: bool = False     # True = open OpenCV window (used by demo.py)
