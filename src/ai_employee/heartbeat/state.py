"""State.json — atomic, crash-safe read/write.

The state file holds everything the heartbeat needs to resume after a restart:
current emotional/cognitive state, intensity, tick counters, last observation,
next-tick delay.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def default_state(default_delay_seconds: int = 300, default_state_name: str = "BOREDOM",
                  default_intensity: int = 1) -> dict:
    return {
        "state": default_state_name,
        "intensity": default_intensity,
        "since_ts": datetime.now(timezone.utc).isoformat(),
        "last_tick_ts": None,
        "tick_count_in_state": 0,
        "next_tick_delay_seconds": default_delay_seconds,
        "last_observation": "first boot",
    }


def load_state(path: Path, default_delay_seconds: int = 300) -> dict:
    """Load state.json, returning a fresh default if it doesn't exist."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default_state(default_delay_seconds=default_delay_seconds)
    except json.JSONDecodeError:
        # Corrupt file — back it up and start fresh rather than crash.
        backup = path.with_suffix(path.suffix + ".corrupt")
        try:
            path.rename(backup)
        except OSError:
            pass
        return default_state(default_delay_seconds=default_delay_seconds)


def write_state_atomic(path: Path, state: dict) -> None:
    """Write state.json atomically: write temp + os.replace. Crash-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def update_state(
    prev: dict,
    new_state: str,
    new_intensity: int,
    next_delay: int,
    observation: str,
) -> dict:
    """Compute the next state dict from the previous one + a parsed update."""
    same_state = (new_state == prev.get("state"))
    return {
        "state": new_state,
        "intensity": new_intensity,
        "since_ts": prev.get("since_ts") if same_state else datetime.now(timezone.utc).isoformat(),
        "last_tick_ts": datetime.now(timezone.utc).isoformat(),
        "tick_count_in_state": (prev.get("tick_count_in_state", 0) + 1) if same_state else 1,
        "next_tick_delay_seconds": next_delay,
        "last_observation": observation,
    }
