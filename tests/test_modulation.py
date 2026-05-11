"""Modulation — activity multiplier math."""
import os
import time
from pathlib import Path

from ai_employee.config import ModulationConfig
from ai_employee.heartbeat.modulation import compute_activity_multiplier


def test_modulation_disabled_returns_one(tmp_path):
    cfg = ModulationConfig(enabled=False)
    assert compute_activity_multiplier(tmp_path, cfg) == 1.0


def test_active_workspace_multiplier(tmp_path):
    # Create a fresh file → workspace is "active".
    (tmp_path / "recent.txt").write_text("hello")
    cfg = ModulationConfig(
        enabled=True, signals=["filesystem"],
        active_window_minutes=60, idle_window_minutes=1440,
        active_multiplier=0.5, idle_multiplier=2.0,
    )
    assert compute_activity_multiplier(tmp_path, cfg) == 0.5


def test_idle_workspace_multiplier(tmp_path):
    p = tmp_path / "old.txt"
    p.write_text("hello")
    # Backdate the file by ~2 days.
    two_days_ago = time.time() - 2 * 86400
    os.utime(p, (two_days_ago, two_days_ago))
    cfg = ModulationConfig(
        enabled=True, signals=["filesystem"],
        active_window_minutes=5, idle_window_minutes=60,
        active_multiplier=0.5, idle_multiplier=2.0,
    )
    assert compute_activity_multiplier(tmp_path, cfg) == 2.0
