"""
examples/basic_test.py
======================
Minimal stdout JSON stream — no GUI, pure data output every frame.

Useful for:
  - Piping visHand output to another program
  - Debugging the JSON schema
  - Integration testing

Usage:
    cd visHand
    python examples/basic_test.py

    # Pipe to another consumer:
    python examples/basic_test.py | python my_app.py

Output:
    One JSON line per frame (always, even if LOCKED) to stdout.
    Status messages go to stderr so they don't pollute the data stream.
"""

import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from core.detector import HandDetector
from core.interpreter import GestureInterpreter


def main():
    settings = Settings(
        output_mode="stdout",
        debug_window=False,
    )

    print("[visHand] Stream starting…  Ctrl+C to stop.", file=sys.stderr)
    print("[visHand] Output: one JSON line per frame on stdout.", file=sys.stderr)

    try:
        detector    = HandDetector(settings)
        interpreter = GestureInterpreter(settings)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    frame_id = 0

    try:
        while True:
            timestamp = time.time()

            ok, frame = detector.read_frame()
            if not ok:
                print("[visHand] Camera read failed.", file=sys.stderr)
                break

            results = detector.extract_landmarks(frame)
            if results:
                result  = results[0]
                payload = interpreter.process(
                    result.landmarks, result.hand_side, timestamp, frame_id
                )
            else:
                payload = interpreter.process(None, "RIGHT", timestamp, frame_id)

            # Write one JSON line to stdout (flush so pipe consumers see it immediately)
            print(json.dumps(payload, ensure_ascii=False), flush=True)

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[visHand] Stream stopped.", file=sys.stderr)
    finally:
        detector.release()


if __name__ == "__main__":
    main()
