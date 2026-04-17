from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "ArmAssistGate",
    "ActionSafetyLayer",
    "FramePacket",
    "FrameQuality",
    "InferencePacket",
    "HandDetector",
    "HandLandmarkResult",
    "Point3D",
    "QualityBlock",
    "QualityGate",
    "OneEuroFilter",
    "LandmarkFilter",
    "GestureInterpreter",
    "GestureDef",
    "GestureRegistry",
    "registry",
    "BoneLengthClamp",
    "LandmarkKalmanPredictor",
    "HandsBackend",
    "PoseBackend",
]

_LAZY_IMPORTS = {
    "ArmAssistGate": ("core.arm_assist", "ArmAssistGate"),
    "OneEuroFilter": ("core.filters", "OneEuroFilter"),
    "LandmarkFilter": ("core.filters", "LandmarkFilter"),
    "GestureDef": ("core.gestures", "GestureDef"),
    "GestureRegistry": ("core.gestures", "GestureRegistry"),
    "registry": ("core.gestures", "registry"),
    "BoneLengthClamp": ("core.kinematics", "BoneLengthClamp"),
    "LandmarkKalmanPredictor": ("core.kinematics", "LandmarkKalmanPredictor"),
    "HandsBackend": ("core.protocols", "HandsBackend"),
    "PoseBackend": ("core.protocols", "PoseBackend"),
    "FrameQuality": ("core.quality_gate", "FrameQuality"),
    "QualityGate": ("core.quality_gate", "QualityGate"),
    "ActionSafetyLayer": ("core.safety", "ActionSafetyLayer"),
    "FramePacket": ("core.types", "FramePacket"),
    "HandLandmarkResult": ("core.types", "HandLandmarkResult"),
    "InferencePacket": ("core.types", "InferencePacket"),
    "Point3D": ("core.types", "Point3D"),
    "QualityBlock": ("core.types", "QualityBlock"),
    "HandDetector": ("core.detector", "HandDetector"),
    "GestureInterpreter": ("core.interpreter", "GestureInterpreter"),
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'core' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from core.arm_assist import ArmAssistGate
    from core.detector import HandDetector
    from core.filters import LandmarkFilter, OneEuroFilter
    from core.gestures import GestureDef, GestureRegistry, registry
    from core.kinematics import BoneLengthClamp, LandmarkKalmanPredictor
    from core.protocols import HandsBackend, PoseBackend
    from core.quality_gate import FrameQuality, QualityGate
    from core.safety import ActionSafetyLayer
    from core.types import FramePacket, HandLandmarkResult, InferencePacket, Point3D, QualityBlock
