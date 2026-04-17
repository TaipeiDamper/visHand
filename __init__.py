"""visHand — Hand Gesture Recognition Module
============================================

Quick start (library mode):
    from config.settings import Settings
    from core.detector import HandDetector
    from core.interpreter import GestureInterpreter

Quick start (demo mode):
    python demo.py
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"
__all__ = [
    "Settings",
    "create_transport",
    "HandDetector", "HandLandmarkResult", "Point3D",
    "GestureInterpreter",
    "GestureDef", "GestureRegistry", "registry",
    "DebugVisualizer",
]

_LAZY_IMPORTS = {
    "Settings": ("config.settings", "Settings"),
    "create_transport": ("bridge.factory", "create_transport"),
    "HandDetector": ("core.detector", "HandDetector"),
    "HandLandmarkResult": ("core.types", "HandLandmarkResult"),
    "Point3D": ("core.types", "Point3D"),
    "GestureInterpreter": ("core.interpreter", "GestureInterpreter"),
    "GestureDef": ("core.gestures", "GestureDef"),
    "GestureRegistry": ("core.gestures", "GestureRegistry"),
    "registry": ("core.gestures", "registry"),
    "DebugVisualizer": ("utils.visualizer", "DebugVisualizer"),
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'visHand' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from bridge.factory import create_transport
    from config.settings import Settings
    from core.detector import HandDetector
    from core.gestures import GestureDef, GestureRegistry, registry
    from core.interpreter import GestureInterpreter
    from core.types import HandLandmarkResult, Point3D
    from utils.visualizer import DebugVisualizer
