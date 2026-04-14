"""Standalone verification script — not part of the main module."""
import sys, os, math, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from core.detector import Point3D
from core.filters import OneEuroFilter, LandmarkFilter
from core.interpreter import GestureInterpreter
from utils.math_tools import palm_width, palm_roll_angle
from utils.visualizer import DebugVisualizer

print("All imports OK")

# ── Test 1: OneEuroFilter convergence ─────────────────────────────────────────
f = OneEuroFilter(freq=30)
vals = [0.5 + 0.01 * math.sin(i * 0.5) for i in range(60)]
out = [f(v, i / 30.0) for i, v in enumerate(vals)]
drift = abs(out[-1] - vals[-1])
assert drift < 0.05, f"Filter drift too large: {drift}"
print("Test 1 OneEuroFilter: OK")

# ── Test 2: Interpreter starts LOCKED ─────────────────────────────────────────
s = Settings()
interp = GestureInterpreter(s)
p = interp.process(None, "RIGHT", time.time(), 0)
assert p["state"]["logic"] == "LOCKED", "Expected LOCKED, got: " + p["state"]["logic"]
assert p["dynamics"]["event"] == "NONE"
print("Test 2 Interpreter LOCKED: OK")

# ── Test 3: JSON schema ────────────────────────────────────────────────────────
for k in ("header", "state", "transform", "dynamics"):
    assert k in p, "Missing top-level key: " + k
assert "frame_id" in p["header"]
assert "anchor"   in p["transform"]
assert "event"    in p["dynamics"]
print("Test 3 JSON schema: OK")

# ── Test 4: Feeding hand frames → state transitions ──────────────────────────
# Use synthetic landmarks (all zeros — just to drive the state machine)
dummy_lm = [Point3D(0.5, 0.5, 0.0)] * 21
for i in range(5):
    p = interp.process(dummy_lm, "RIGHT", time.time(), i + 1)

# After locked_to_hover_frames (3) with hand, should be HOVER or ACTIVE
assert p["state"]["logic"] in ("HOVER", "ACTIVE"), (
    "Expected HOVER/ACTIVE after hand frames, got: " + p["state"]["logic"]
)
print("Test 4 State transition LOCKED→HOVER: OK")

# ── Test 5: math_tools ────────────────────────────────────────────────────────
pts = [Point3D(float(i) * 0.05, float(i) * 0.03, 0.0) for i in range(21)]
pw = palm_width(pts)
assert pw >= 0.0, "palm_width should be non-negative"
roll = palm_roll_angle(pts)
assert -180.0 <= roll <= 180.0, "Roll angle out of range"
print("Test 5 math_tools: OK")

print()
print("=== All checks passed ===")
