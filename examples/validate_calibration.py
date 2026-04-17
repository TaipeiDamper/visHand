"""
Validate calibration JSONL quality.

Usage:
  python examples/validate_calibration.py --input scratch/calibration/xxx.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.calibration_profile import export_profile_from_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="Validate visHand calibration JSONL")
    p.add_argument("--input", required=True, help="path to calibration jsonl")
    p.add_argument("--ambiguity-gap", type=float, default=0.08, help="top2 score gap threshold")
    p.add_argument("--export-profile", default="", help="optional output path for generated profile json")
    return p.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON parse error at line {i}: {exc}") from exc
    return rows


def main():
    args = parse_args()
    path = Path(args.input)
    rows = load_rows(path)
    if not rows:
        raise RuntimeError("No rows found in calibration file.")

    required = ("timestamp", "frame_id", "pred_intent", "pred_event", "landmarks")
    for idx, row in enumerate(rows):
        for k in required:
            if k not in row:
                raise RuntimeError(f"Missing required key '{k}' at row {idx}")

    switches = 0
    ambiguity = 0
    arm_assist_count = 0
    handedness_flips = 0
    directional_conflicts = 0
    directional_set = {"INDEX_POINT_LEFT", "INDEX_POINT_RIGHT", "SWIPE_LEFT", "SWIPE_RIGHT"}
    prev_intent = rows[0].get("pred_intent", "IDLE")
    prev_handedness = rows[0].get("handedness_fused", "")
    for row in rows[1:]:
        intent = row.get("pred_intent", "IDLE")
        if intent != prev_intent:
            switches += 1
        prev_intent = intent

        hand_now = row.get("handedness_fused", "")
        if hand_now and prev_handedness and hand_now != prev_handedness:
            handedness_flips += 1
        if hand_now:
            prev_handedness = hand_now

        arb = row.get("arbitration", [])
        if isinstance(arb, list) and len(arb) >= 2:
            s0 = float(arb[0].get("score", 0.0))
            s1 = float(arb[1].get("score", 0.0))
            if abs(s0 - s1) <= args.ambiguity_gap:
                ambiguity += 1
            n0 = str(arb[0].get("name", ""))
            n1 = str(arb[1].get("name", ""))
            if n0 in directional_set and n1 in directional_set and n0 != n1:
                directional_conflicts += 1
        if int(row.get("arm_assist_applied", 0)) == 1:
            arm_assist_count += 1

    n = len(rows)
    switch_rate = switches / max(n - 1, 1)
    ambiguity_rate = ambiguity / n
    arm_assist_rate = arm_assist_count / n
    handedness_flip_rate = handedness_flips / max(n - 1, 1)
    directional_conflict_rate = directional_conflicts / n
    print(json.dumps(
        {
            "rows": n,
            "switch_rate": round(switch_rate, 4),
            "ambiguity_rate": round(ambiguity_rate, 4),
            "arm_assist_rate": round(arm_assist_rate, 4),
            "handedness_flip_rate": round(handedness_flip_rate, 4),
            "directional_conflict_rate": round(directional_conflict_rate, 4),
            "parse_ok": True,
        },
        ensure_ascii=False,
        indent=2,
    ))
    if str(args.export_profile).strip():
        out = export_profile_from_jsonl(str(path), str(args.export_profile))
        print(json.dumps({"profile_exported": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
