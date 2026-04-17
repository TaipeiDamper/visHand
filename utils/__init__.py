from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

_MATH_EXPORTS = [
    "euclidean_3d",
    "euclidean_2d",
    "palm_width",
    "normalized_distance",
    "finger_is_extended",
    "finger_curl",
    "all_fingers_extended",
    "all_fingers_curled",
    "pinch_intensity",
    "palm_anchor",
    "pinch_anchor",
    "palm_roll_angle",
    "palm_euler_angles",
    "instant_velocity",
    "compute_delta",
    "compute_delta3d",
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

__all__ = ["DebugVisualizer", "RuntimeTelemetry", *_MATH_EXPORTS]

_LAZY_IMPORTS = {
    "DebugVisualizer": ("utils.visualizer", "DebugVisualizer"),
    "RuntimeTelemetry": ("utils.profiler", "RuntimeTelemetry"),
    **{name: ("utils.math_tools", name) for name in _MATH_EXPORTS},
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'utils' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from utils.math_tools import (
        INDEX_DIP,
        INDEX_MCP,
        INDEX_PIP,
        INDEX_TIP,
        MIDDLE_DIP,
        MIDDLE_MCP,
        MIDDLE_PIP,
        MIDDLE_TIP,
        PINKY_DIP,
        PINKY_MCP,
        PINKY_PIP,
        PINKY_TIP,
        RING_DIP,
        RING_MCP,
        RING_PIP,
        RING_TIP,
        THUMB_CMC,
        THUMB_IP,
        THUMB_MCP,
        THUMB_TIP,
        WRIST,
        all_fingers_curled,
        all_fingers_extended,
        compute_delta,
        compute_delta3d,
        euclidean_2d,
        euclidean_3d,
        finger_curl,
        finger_is_extended,
        instant_velocity,
        normalized_distance,
        palm_anchor,
        palm_euler_angles,
        palm_roll_angle,
        palm_width,
        pinch_anchor,
        pinch_intensity,
    )
    from utils.profiler import RuntimeTelemetry
    from utils.visualizer import DebugVisualizer
