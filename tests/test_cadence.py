"""Activity-based cadence — Discord age → faster tick."""
from ai_employee.config import DiscordCadenceConfig
from ai_employee.heartbeat.cadence import discord_active_delay


def test_age_none_returns_requested():
    cfg = DiscordCadenceConfig()
    assert discord_active_delay(None, cfg, 300) == 300


def test_active_window_floors_to_15():
    cfg = DiscordCadenceConfig()
    # 60 seconds ago = active (< 2 min)
    assert discord_active_delay(60, cfg, 300) == 15


def test_recent_window_floors_to_30():
    cfg = DiscordCadenceConfig()
    # 4 minutes ago = recent (< 5 min)
    assert discord_active_delay(4 * 60, cfg, 300) == 30


def test_pulse_window_floors_to_60():
    cfg = DiscordCadenceConfig()
    # 10 minutes ago = pulse (< 15 min)
    assert discord_active_delay(10 * 60, cfg, 300) == 60


def test_quiet_uses_requested():
    cfg = DiscordCadenceConfig()
    # 30 minutes ago — agent's chosen delay wins
    assert discord_active_delay(30 * 60, cfg, 600) == 600


def test_never_increases_above_requested():
    """Cadence speeds up; it never slows down."""
    cfg = DiscordCadenceConfig()
    # If agent already chose to be fast, don't override upward.
    assert discord_active_delay(60, cfg, 10) == 10  # requested smaller than active_delay
