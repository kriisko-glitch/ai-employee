"""State.json — atomic write + load."""
import json
from pathlib import Path

from ai_employee.heartbeat.state import (
    default_state, load_state, write_state_atomic, update_state,
)


def test_default_state_shape():
    s = default_state()
    assert "state" in s
    assert "intensity" in s
    assert "next_tick_delay_seconds" in s
    assert s["tick_count_in_state"] == 0


def test_load_state_missing_file(tmp_path: Path):
    s = load_state(tmp_path / "state.json", default_delay_seconds=120)
    assert s["next_tick_delay_seconds"] == 120
    assert s["tick_count_in_state"] == 0


def test_write_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "state.json"
    s = default_state()
    s["state"] = "BUILDING"
    s["intensity"] = 2
    write_state_atomic(path, s)

    loaded = load_state(path)
    assert loaded["state"] == "BUILDING"
    assert loaded["intensity"] == 2


def test_corrupt_file_is_recovered(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    s = load_state(path)
    # Should silently recover to defaults rather than raise.
    assert s["state"] == "BOREDOM"
    # And the corrupt file should be moved aside.
    assert path.with_suffix(".json.corrupt").exists()


def test_update_state_increments_tick_count_on_same_state():
    prev = default_state()
    prev["state"] = "BUILDING"
    prev["tick_count_in_state"] = 3
    new = update_state(prev, "BUILDING", 2, 300, "still working")
    assert new["tick_count_in_state"] == 4
    assert new["since_ts"] == prev["since_ts"]


def test_update_state_resets_on_state_change():
    prev = default_state()
    prev["state"] = "BUILDING"
    prev["tick_count_in_state"] = 5
    # Force an older since_ts so the new timestamp is guaranteed to differ.
    prev["since_ts"] = "2025-01-01T00:00:00+00:00"
    new = update_state(prev, "REFLECTING", 1, 600, "shift")
    assert new["tick_count_in_state"] == 1
    assert new["since_ts"] != prev["since_ts"]


def test_atomic_write_uses_replace(tmp_path: Path):
    path = tmp_path / "state.json"
    write_state_atomic(path, default_state())
    # No leftover .tmp file.
    assert not path.with_suffix(".json.tmp").exists()
    assert json.loads(path.read_text())["state"] == "BOREDOM"
