# visHand

> Real-time hand gesture recognition module built on MediaPipe + OpenCV.  
> Outputs a structured JSON payload for every camera frame — ready to drive
> mouse replacement, 3D object interaction, music module control, and more.

---

## Quick Start

### 1. Install dependencies

```bash
cd visHand
pip install -r requirements.txt
```

### 2. Run the live demo

```bash
python demo.py
```

Opens your webcam with a debug overlay showing:
- Hand skeleton colour-coded by state (grey → amber → green)
- Current intent label
- Fading event banner for SNAP / DOUBLE\_TAP
- FPS counter

Keyboard: **ESC** or **Q** to quit.

### 3. Pure JSON stream (for piping)

```bash
python examples/basic_test.py
# or pipe to another program:
python examples/basic_test.py | python my_consumer.py
```

---

## Output Format

Every frame produces one JSON object:

```json
{
  "header": {
    "frame_id": 1024,
    "timestamp": 1713123456.78,
    "hand_side": "RIGHT"
  },
  "state": {
    "logic":     "ACTIVE",       // LOCKED | HOVER | ACTIVE
    "intent":    "PINCH_DRAG",   // see Intent table below
    "is_stable": true            // false during 1€ filter warmup
  },
  "transform": {
    "anchor": { "x": 0.512, "y": 0.489, "z": -0.052 },
    "delta":  { "dx": 0.002, "dy": -0.001 },
    "rotation": 15.5
  },
  "dynamics": {
    "intensity": 0.98,
    "velocity":  0.015,
    "event":     "NONE"          // NONE | SNAP | DOUBLE_TAP | CLAP
  }
}
```

### Logic States

| State | Meaning |
|-------|---------|
| `LOCKED` | No hand detected |
| `HOVER` | Hand visible, no active gesture |
| `ACTIVE` | Active gesture in progress |

### Intents

| Intent | Trigger |
|--------|---------|
| `IDLE` | Hand present, no gesture |
| `POINT` | Index finger extended, rest curled |
| `PINCH_HOLD` | Thumb + index close, static |
| `PINCH_DRAG` | Thumb + index close, moving |
| `FIST` | All fingers curled |
| `OPEN_PALM` | All fingers extended |
| `SNAP_READY` | Middle + thumb in contact (pre-snap) |
| `SWIPE_LEFT` | Fast lateral motion leftward |
| `SWIPE_RIGHT` | Fast lateral motion rightward |

### Events (one-shot, in `dynamics.event`)

| Event | Trigger |
|-------|---------|
| `SNAP` | Middle finger snaps away from thumb (two-phase) |
| `DOUBLE_TAP` | Two pinch onsets within `double_tap_window_ms` |

---

## Library Usage

```python
import sys
sys.path.insert(0, "/path/to/visHand")   # or install as a package

import time
from config.settings import Settings
from core.detector import HandDetector
from core.interpreter import GestureInterpreter

settings    = Settings(camera_index=0)
detector    = HandDetector(settings)
interpreter = GestureInterpreter(settings)

frame_id = 0
with detector:
    while True:
        ok, frame = detector.read_frame()
        if not ok:
            break
        results = detector.extract_landmarks(frame)
        lm = results[0].landmarks if results else None
        hs = results[0].hand_side if results else "RIGHT"
        payload = interpreter.process(lm, hs, time.time(), frame_id)

        # Use payload in your app
        if payload["dynamics"]["event"] == "SNAP":
            print("Snap detected!")

        frame_id += 1
```

---

## Configuration

Edit `config/settings.py` or pass overrides:

```python
s = Settings(
    camera_index=1,          # external webcam
    pinch_threshold=0.18,    # tighter pinch required
    snap_cooldown_ms=400,    # faster snap repeat
)
```

Key thresholds are **normalised by palm width**, so they work at any
distance from the camera.

---

## Project Structure

```
visHand/
├── core/
│   ├── detector.py      # MediaPipe wrapper, 21-point extraction
│   ├── filters.py       # 1 Euro Filter + LandmarkFilter
│   └── interpreter.py   # State machine, intent + event detection
├── utils/
│   ├── math_tools.py    # Pure geometry (distances, angles, velocity)
│   └── visualizer.py    # OpenCV skeleton + overlay drawing
├── config/
│   └── settings.py      # All thresholds and options
├── examples/
│   └── basic_test.py    # stdout JSON stream for piping
├── demo.py              # Live visual debug demo
└── requirements.txt
```

---

## Extending to Dual-Hand

Set `max_hands=2` in Settings and instantiate two `GestureInterpreter` objects
(one per hand).  The `demo.py` already scaffolds this with a list of interpreters.
Dual-hand events like `CLAP` and `PINCH_ROTATE` can be added in a higher-level
coordinator that receives both payload dicts.
