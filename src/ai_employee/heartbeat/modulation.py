"""User-activity modulation — make the heartbeat breathe with the user.

Active user (recent file edits, commits) → next tick delay × active_multiplier (faster).
Idle user (no signals for idle_window) → next tick delay × idle_multiplier (slower).

Signals are pluggable; today we support filesystem mtime scans and git log lookups.
Both are best-effort and fail silently — modulation should never crash the loop.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..config import ModulationConfig


def _most_recent_fs_mtime(root: Path, ignore_globs: tuple[str, ...] = ("memory.db*", "state.json*", "budget.json*", "*.log", "*.tmp")) -> datetime | None:
    """Scan the workspace recursively for the most recent file mtime.

    Files written by the agent itself (state.json, memory.db, etc.) are
    excluded so the agent doesn't see its own writes as "user activity."
    """
    if not root.exists():
        return None
    latest = None
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(p.match(g) for g in ignore_globs):
                continue
            try:
                mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if latest is None or mt > latest:
                latest = mt
    except (PermissionError, OSError):
        pass
    return latest


def _most_recent_git_commit(root: Path) -> datetime | None:
    """Look up most recent git commit time, if root is inside a repo."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%cI"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return datetime.fromisoformat(out.stdout.strip().replace("Z", "+00:00"))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def compute_activity_multiplier(workspace: Path, config: ModulationConfig) -> float:
    """Return a multiplier to apply to next_tick_delay based on user activity.

    < 1.0 = tick faster (user is around), > 1.0 = tick slower (user idle).
    Returns 1.0 when modulation is disabled or no signal is available.
    """
    if not config.enabled:
        return 1.0

    most_recent: datetime | None = None
    for signal in config.signals:
        candidate: datetime | None = None
        if signal == "filesystem":
            candidate = _most_recent_fs_mtime(workspace)
        elif signal == "git":
            candidate = _most_recent_git_commit(workspace)
        if candidate and (most_recent is None or candidate > most_recent):
            most_recent = candidate

    if most_recent is None:
        return 1.0

    age_minutes = (datetime.now(timezone.utc) - most_recent).total_seconds() / 60.0
    if age_minutes <= config.active_window_minutes:
        return config.active_multiplier
    if age_minutes >= config.idle_window_minutes:
        return config.idle_multiplier
    return 1.0
