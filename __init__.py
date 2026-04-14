"""visHand — Hand Gesture Recognition Module
============================================

Quick start (library mode):
    from config.settings import Settings
    from core.detector import HandDetector
    from core.interpreter import GestureInterpreter

Quick start (demo mode):
    python demo.py
"""
from config.settings import Settings
from core.detector import HandDetector, HandLandmarkResult, Point3D
from core.interpreter import GestureInterpreter
from utils.visualizer import DebugVisualizer

__version__ = "0.1.0"
__all__ = [
    "Settings",
    "HandDetector", "HandLandmarkResult", "Point3D",
    "GestureInterpreter",
    "DebugVisualizer",
]
