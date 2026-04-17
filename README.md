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

## Documentation Map

為了避免重複文件與版本漂移，文件責任如下：

- `bridge/INTEGRATION_CONTRACT.md`：外部系統對接契約（唯一來源，含必需欄位與相容規則）
- `bridge/README.md`：Bridge 模組用途與檔案入口
- `ARCHITECTURE_FLOW.md`：內部架構與資料流
- `CHANGELOG.md`：開發歷程與歷史變更記錄

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
    bridge_transport="ws",   # stdout | ws | pipe
    bridge_enable_extended_transform=True,  # add dz/rotation_euler/dt_ms
    bridge_enable_event_phase=True,         # add dynamics.event_phase
    bridge_enable_hand_identity=True,       # add header.hand_id
)
```

Key thresholds are **normalised by palm width**, so they work at any
distance from the camera.

For bridge payload compatibility details, refer to `bridge/INTEGRATION_CONTRACT.md`.

---

## Project Structure (Current)

```
visHand/
├── bridge/
│   ├── INTEGRATION_CONTRACT.md # 外部對接契約（SSOT）
│   ├── schema_v1.py            # payload schema 規則
│   ├── transport.py            # stdout/ws/pipe transport 抽象
│   └── factory.py              # transport 工廠
├── core/
│   ├── detector.py      # facade
│   ├── capture.py       # camera capture worker
│   ├── inference.py     # inference worker + ROI + quality gate
│   ├── interpreter.py   # state machine + intent/event
│   └── ...              # gestures / arm_assist / safety / types
├── utils/
│   ├── math_tools.py
│   ├── visualizer.py
│   └── profiler.py
├── config/
│   └── settings.py
├── examples/
│   └── ...              # calibration / validation tools
├── demo.py
└── requirements.txt
```
