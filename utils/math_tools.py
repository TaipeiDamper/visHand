"""
visHand — Math Tools
=====================
Pure geometry helpers that operate on MediaPipe-style landmark lists.
No state, no side-effects — all functions are safe to call from anywhere.

MediaPipe hand landmark indices (21 points):
  0          WRIST
  1-4        THUMB   CMC → MCP → IP → TIP
  5-8        INDEX   MCP → PIP → DIP → TIP
  9-12       MIDDLE  MCP → PIP → DIP → TIP
  13-16      RING    MCP → PIP → DIP → TIP
  17-20      PINKY   MCP → PIP → DIP → TIP
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

from core.detector import Point3D

# ---------------------------------------------------------------------------
# Landmark index constants (import these everywhere instead of magic numbers)
# ---------------------------------------------------------------------------

WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP,     THUMB_TIP   =  1,  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP   =  5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP  =  9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP    = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP   = 17, 18, 19, 20

# Finger lookup tables  (index 0 = thumb … 4 = pinky)
_TIP_IDS = [THUMB_TIP,  INDEX_TIP,  MIDDLE_TIP,  RING_TIP,  PINKY_TIP]
_PIP_IDS = [THUMB_IP,   INDEX_PIP,  MIDDLE_PIP,  RING_PIP,  PINKY_PIP]
_MCP_IDS = [THUMB_CMC,  INDEX_MCP,  MIDDLE_MCP,  RING_MCP,  PINKY_MCP]


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------

def euclidean_3d(p1: Point3D, p2: Point3D) -> float:
    """3-D Euclidean distance in normalised landmark space."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def euclidean_2d(p1: Point3D, p2: Point3D) -> float:
    """2-D Euclidean distance (ignores depth)."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def palm_width(lm: List[Point3D]) -> float:
    """
    Distance between INDEX_MCP and PINKY_MCP.
    Used as the normalisation reference so all thresholds are
    scale-invariant (work at any hand distance from camera).
    """
    return euclidean_3d(lm[INDEX_MCP], lm[PINKY_MCP])


def normalized_distance(p1: Point3D, p2: Point3D, lm: List[Point3D]) -> float:
    """3-D distance between p1 and p2, divided by palm width."""
    ref = palm_width(lm)
    if ref < 1e-6:
        return 0.0
    return euclidean_3d(p1, p2) / ref


# ---------------------------------------------------------------------------
# Finger state
# ---------------------------------------------------------------------------

def finger_is_extended(lm: List[Point3D], finger_id: int) -> bool:
    """
    True if the finger appears extended.

    For fingers 1-4 (index to pinky):
        Extended when TIP is *above* (smaller y) the PIP joint in image space.
        (MediaPipe y increases downward; 0 = top, 1 = bottom.)

    For thumb (0):
        Extended when THUMB_TIP is farther from INDEX_MCP than THUMB_IP is.
        (Scale-invariant, works for both hands.)
    """
    if finger_id == 0:
        return (euclidean_2d(lm[THUMB_TIP], lm[INDEX_MCP]) >
                euclidean_2d(lm[THUMB_IP],  lm[INDEX_MCP]))
    tip_id = _TIP_IDS[finger_id]
    pip_id = _PIP_IDS[finger_id]
    return lm[tip_id].y < lm[pip_id].y


def finger_curl(lm: List[Point3D], finger_id: int) -> float:
    """
    Curl amount for a finger.  0.0 = fully extended, 1.0 = fully curled.

    For thumb: binary approximation via finger_is_extended.
    For other fingers: continuous mapping using the TIP–PIP–MCP y-geometry.
    """
    if finger_id == 0:
        return 0.0 if finger_is_extended(lm, 0) else 0.85

    tip_id = _TIP_IDS[finger_id]
    pip_id = _PIP_IDS[finger_id]
    mcp_id = _MCP_IDS[finger_id]

    tip_y = lm[tip_id].y
    pip_y = lm[pip_id].y
    mcp_y = lm[mcp_id].y

    # Span between MCP and PIP acts as the scale reference for this finger
    span = abs(mcp_y - pip_y)
    if span < 1e-4:
        return 0.5

    # raw > 0 means TIP is above PIP (extended); raw < 0 means curled
    raw = (pip_y - tip_y) / span
    # Map [-1, 1] → curl [1, 0] and clamp
    curl = (1.0 - min(max(raw, -1.0), 1.0)) / 2.0
    return curl


def all_fingers_extended(lm: List[Point3D], include_thumb: bool = False) -> bool:
    start = 0 if include_thumb else 1
    return all(finger_is_extended(lm, i) for i in range(start, 5))


def all_fingers_curled(lm: List[Point3D]) -> bool:
    """True if all four non-thumb fingers are curled."""
    return all(not finger_is_extended(lm, i) for i in range(1, 5))


# ---------------------------------------------------------------------------
# Gesture metrics
# ---------------------------------------------------------------------------

def pinch_intensity(lm: List[Point3D]) -> float:
    """
    0.0 = fully open, 1.0 = fully pinched (thumb tip touching index tip).
    Normalised by palm width so the value is distance-independent.
    """
    pw = palm_width(lm)
    if pw < 1e-6:
        return 0.0
    dist = euclidean_3d(lm[THUMB_TIP], lm[INDEX_TIP])
    # ~0.5× palm_width is a fully-open pinch position
    intensity = 1.0 - min(dist / (pw * 0.5), 1.0)
    return max(0.0, intensity)


def palm_anchor(lm: List[Point3D]) -> Tuple[float, float, float]:
    """Centroid of the five MCP joints — stable palm centre estimate."""
    mcp_ids = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    x = sum(lm[i].x for i in mcp_ids) / 5.0
    y = sum(lm[i].y for i in mcp_ids) / 5.0
    z = sum(lm[i].z for i in mcp_ids) / 5.0
    return x, y, z


def pinch_anchor(lm: List[Point3D]) -> Tuple[float, float, float]:
    """Midpoint between thumb tip and index tip — gesture control point."""
    x = (lm[THUMB_TIP].x + lm[INDEX_TIP].x) / 2.0
    y = (lm[THUMB_TIP].y + lm[INDEX_TIP].y) / 2.0
    z = (lm[THUMB_TIP].z + lm[INDEX_TIP].z) / 2.0
    return x, y, z


def palm_roll_angle(lm: List[Point3D]) -> float:
    """
    Roll angle of the palm in degrees.
    Computed from the INDEX_MCP → PINKY_MCP vector projected onto the XY plane.
    0° = horizontal; positive = clockwise tilt.
    """
    dx = lm[PINKY_MCP].x - lm[INDEX_MCP].x
    dy = lm[PINKY_MCP].y - lm[INDEX_MCP].y
    return math.degrees(math.atan2(dy, dx))


# ---------------------------------------------------------------------------
# Motion utilities
# ---------------------------------------------------------------------------

def instant_velocity(history: List[Tuple[float, float]], window: int = 5) -> float:
    """
    Average frame-to-frame 2-D displacement of the anchor point.
    history is a list of (x, y) anchor positions (most recent last).
    """
    n = min(len(history), window)
    if n < 2:
        return 0.0
    recent = history[-n:]
    total = sum(
        math.sqrt((recent[i][0] - recent[i-1][0]) ** 2 +
                  (recent[i][1] - recent[i-1][1]) ** 2)
        for i in range(1, len(recent))
    )
    return total / (len(recent) - 1)


def compute_delta(
    prev: Optional[Tuple[float, float]],
    curr: Tuple[float, float],
) -> Tuple[float, float]:
    """Frame-to-frame displacement (dx, dy) of the anchor."""
    if prev is None:
        return 0.0, 0.0
    return curr[0] - prev[0], curr[1] - prev[1]
