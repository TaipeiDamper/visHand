from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from core.gestures import registry

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class BridgeHeader:
    schema_version: str
    frame_id: int
    timestamp: float
    hand_side: str
    capabilities: List[str]


def capabilities_snapshot() -> List[str]:
    names = [g.name for g in registry.all() if g.enabled]
    names.sort()
    return names

def _capabilities_meta(payload: Dict[str, Any]) -> Dict[str, bool]:
    header = payload.get("header", {})
    transform = payload.get("transform", {})
    dynamics = payload.get("dynamics", {})
    return {
        "extended_transform": bool(transform.get("rotation_euler")) and "dz" in transform.get("delta", {}),
        "event_phase": "event_phase" in dynamics,
        "hand_identity": "hand_id" in header,
    }


def ensure_v1_payload(payload: Dict[str, Any], hand_side: str) -> Dict[str, Any]:
    header = payload.setdefault("header", {})
    header["schema_version"] = SCHEMA_VERSION
    header.setdefault("hand_side", hand_side)
    header["capabilities"] = capabilities_snapshot()
    header["capabilities_meta"] = _capabilities_meta(payload)

    state = payload.setdefault("state", {})
    state.setdefault("input_quality_score", 1.0)
    state.setdefault("tracking_quality", "good")

    dynamics = payload.setdefault("dynamics", {})
    dynamics.setdefault("emergency_cancel", False)
    return payload
