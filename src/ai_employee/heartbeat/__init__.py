"""Heartbeat — autonomous tick loop. Disabled by default; opt-in via config."""

from .state import load_state, write_state_atomic, default_state
from .tick import tick_once, run_loop
from .budget import BudgetTracker
from .modulation import compute_activity_multiplier

__all__ = [
    "load_state",
    "write_state_atomic",
    "default_state",
    "tick_once",
    "run_loop",
    "BudgetTracker",
    "compute_activity_multiplier",
]
