from config.settings import Settings
from core.types import Point3D
from core.interpreter import GestureInterpreter

def _fake_landmarks():
    return [Point3D(0.5 + (i * 0.001), 0.5 - (i * 0.001), 0.02 * (i % 3)) for i in range(21)]

def test_default_payload_compat():
    s = Settings()
    interp = GestureInterpreter(s)
    payload = interp.process(_fake_landmarks(), "RIGHT", 12345.0, 1)
    assert "dz" not in payload["transform"]["delta"], "dz should stay disabled by default"
    assert "rotation_euler" not in payload["transform"], "rotation_euler should stay disabled by default"
    assert "dt_ms" not in payload["header"], "dt_ms should stay disabled by default"
    assert "event_phase" not in payload["dynamics"], "event_phase should stay disabled by default"
    assert "hand_id" not in payload["header"], "hand_id should stay disabled by default"
    print("[PASS] default payload compatibility")

def test_extended_payload_flags():
    s = Settings(
        bridge_enable_extended_transform=True,
        bridge_enable_event_phase=True,
        bridge_enable_hand_identity=True,
    )
    interp = GestureInterpreter(s)
    payload_a = interp.process(_fake_landmarks(), "RIGHT", 20000.0, 10, hand_id="slot-0")
    payload_b = interp.process(_fake_landmarks(), "RIGHT", 20000.033, 11, hand_id="slot-0")
    assert "dz" in payload_b["transform"]["delta"], "dz should exist when extended transform is enabled"
    assert "rotation_euler" in payload_b["transform"], "rotation_euler should exist when extended transform is enabled"
    assert "dt_ms" in payload_b["header"], "dt_ms should exist when extended transform is enabled"
    assert "event_phase" in payload_b["dynamics"], "event_phase should exist when event phase is enabled"
    assert payload_b["header"].get("hand_id") == "slot-0", "hand_id should be emitted when enabled"
    assert payload_b["header"]["capabilities_meta"]["extended_transform"] is True
    assert payload_b["header"]["capabilities_meta"]["event_phase"] is True
    assert payload_b["header"]["capabilities_meta"]["hand_identity"] is True
    assert payload_b["header"]["dt_ms"] >= 0.0
    print("[PASS] extended payload flags")

def test():
    test_default_payload_compat()
    test_extended_payload_flags()
    print("All interpreter checks passed.")

if __name__ == "__main__":
    test()
