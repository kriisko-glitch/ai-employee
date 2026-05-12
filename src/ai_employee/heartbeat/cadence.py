"""Activity-based cadence — speed up the heartbeat when the user is around.

If the watched user posted in the channel <2 min ago → tick in 15s
                                              <5 min   → 30s
                                              <15 min  → 60s
otherwise use the model-chosen delay (clamped by min/max).

This is the "responsive within the same context window" mechanic. When
you talk to Mike Ross, his next tick fires within seconds — same
conversation, no parallel session.
"""
from __future__ import annotations

from typing import Optional

from ..config import DiscordCadenceConfig


def discord_active_delay(age_seconds: Optional[float],
                         config: DiscordCadenceConfig,
                         requested_delay: int) -> int:
    """Compute next-tick delay given when the watched user last posted.

    Returns the smaller of (cadence-adjusted, requested_delay). When the
    user is active, this returns a much shorter delay; when quiet, returns
    the requested delay unchanged.
    """
    if age_seconds is None:
        return requested_delay

    if age_seconds < config.active_within_minutes * 60:
        return min(requested_delay, config.active_delay_seconds)
    if age_seconds < config.recent_within_minutes * 60:
        return min(requested_delay, config.recent_delay_seconds)
    if age_seconds < config.pulse_within_minutes * 60:
        return min(requested_delay, config.pulse_delay_seconds)
    return requested_delay
