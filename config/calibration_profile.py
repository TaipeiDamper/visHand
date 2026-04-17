"""
Calibration profile helpers.

This module keeps profile loading/applying isolated from runtime logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import utils.math_tools as mt
from core.types import Point3D


def load_profile(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Calibration profile not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_to_settings(settings, profile: Dict[str, Any]) -> None:
    """
    Override only keys that exist in Settings.
    """
    for key, value in profile.items():
        if key in ("schema_version", "notes"):
            continue
        if hasattr(settings, key):
            setattr(settings, key, value)


def apply_profile_path(settings) -> None:
    path = str(getattr(settings, "calibration_profile_path", "") or "").strip()
    if not path:
        return
    profile = load_profile(path)
    apply_to_settings(settings, profile)


def _row_landmarks(row: dict) -> Optional[List[Point3D]]:
    landmarks = row.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) < 21:
        return None
    try:
        return [Point3D(float(p["x"]), float(p["y"]), float(p.get("z", 0.0))) for p in landmarks]
    except (TypeError, ValueError, KeyError):
        return None


def build_profile_from_jsonl_rows(
    rows: List[dict],
    pinch_open_quantile: float = 0.90,
    pinch_hold_quantile: float = 0.55,
) -> Dict[str, Any]:
    open_dists: List[float] = []
    pinch_dists: List[float] = []

    for row in rows:
        lm = _row_landmarks(row)
        if lm is None:
            continue
        label = str(row.get("manual_label", "NONE")).upper()
        dist = mt.normalized_distance(lm[mt.THUMB_TIP], lm[mt.INDEX_TIP], lm)
        if label in ("OPEN_PALM", "NONE"):
            open_dists.append(dist)
        elif label in ("PINCH_HOLD", "PINCH_DRAG"):
            pinch_dists.append(dist)

    profile: Dict[str, Any] = {"schema_version": "1"}
    if pinch_dists:
        pinch_dists.sort()
        idx = int((len(pinch_dists) - 1) * pinch_hold_quantile)
        profile["pinch_threshold"] = float(max(0.08, min(0.35, pinch_dists[idx] * 1.05)))
    if open_dists:
        open_dists.sort()
        idx = int((len(open_dists) - 1) * pinch_open_quantile)
        profile["pinch_release_threshold"] = float(max(0.12, min(0.45, open_dists[idx] * 0.95)))

    if "pinch_threshold" in profile and "pinch_release_threshold" in profile:
        if profile["pinch_release_threshold"] <= profile["pinch_threshold"]:
            profile["pinch_release_threshold"] = round(profile["pinch_threshold"] + 0.06, 4)
    return profile


def export_profile_from_jsonl(input_path: str, output_path: str) -> str:
    rows: List[dict] = []
    with Path(input_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    profile = build_profile_from_jsonl_rows(rows)
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return str(outp)
