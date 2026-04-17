"""
visHand — Runtime Telemetry Buffer
==================================
Collect frame-level timings and dump percentile summaries.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Dict, List

import numpy as np


class RuntimeTelemetry:
    def __init__(self, max_frames: int = 300):
        self.max_frames = max(30, int(max_frames))
        self._frames = deque(maxlen=self.max_frames)

    def add(self, row: Dict):
        self._frames.append(dict(row))

    def reset(self):
        self._frames.clear()

    @property
    def size(self) -> int:
        return len(self._frames)

    def summary(self) -> Dict:
        if not self._frames:
            return {"frames": 0}

        keys = [
            "capture_ms",
            "mp_ms",
            "filter_ms",
            "render_ms",
            "e2e_latency_ms",
        ]
        out = {"frames": len(self._frames)}
        for k in keys:
            vals = np.array([float(r.get(k, 0.0)) for r in self._frames], dtype=np.float64)
            out[k] = {
                "p50": float(np.percentile(vals, 50)),
                "p90": float(np.percentile(vals, 90)),
                "p99": float(np.percentile(vals, 99)),
                "avg": float(np.mean(vals)),
            }
        return out

    def robustness_summary(self) -> Dict:
        if not self._frames:
            return {
                "frames": 0,
                "false_event_rate": 0.0,
                "intent_switch_rate": 0.0,
                "avg_reacquire_time_ms": 0.0,
                "quality_degraded_ratio": 0.0,
            }

        rows = list(self._frames)
        frames_n = len(rows)
        event_rows = [r for r in rows if str(r.get("event", "NONE")) != "NONE"]
        false_events = [r for r in event_rows if int(r.get("false_event_flag", 0)) == 1]

        switches = 0
        prev_intent = None
        for r in rows:
            intent = str(r.get("intent", "IDLE"))
            if prev_intent is not None and intent != prev_intent:
                switches += 1
            prev_intent = intent

        reacquire_vals = [
            float(r.get("reacquire_time_ms", 0.0))
            for r in rows
            if float(r.get("reacquire_time_ms", 0.0)) > 0.0
        ]
        degraded_count = sum(1 for r in rows if str(r.get("tracking_quality", "good")) != "good")

        return {
            "frames": frames_n,
            "false_event_rate": float(len(false_events) / max(1, len(event_rows))),
            "intent_switch_rate": float(switches / max(1, frames_n)),
            "avg_reacquire_time_ms": float(np.mean(reacquire_vals)) if reacquire_vals else 0.0,
            "quality_degraded_ratio": float(degraded_count / max(1, frames_n)),
        }

    def dump_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "summary": self.summary(),
            "robustness": self.robustness_summary(),
            "frames": list(self._frames),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path
