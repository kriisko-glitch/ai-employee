"""Daily budget tracker — hard kill-switch for runaway heartbeats.

Tracks token usage per day in <workspace>/budget.json. Computes USD spend
from the per-million-token rates in agent.yaml. When today's spend exceeds
`daily_usd_cap`, the tick loop exits cleanly.

Rates are user-supplied because they change frequently. Set them in
`agent.yaml` from your provider's current rate card.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import BudgetConfig


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class BudgetState:
    date: str
    input_tokens: int
    output_tokens: int
    spend_usd: float


class BudgetTracker:
    def __init__(self, budget_file: Path, config: BudgetConfig):
        self.budget_file = budget_file
        self.config = config

    def _load(self) -> BudgetState:
        if not self.budget_file.exists():
            return BudgetState(date=_today(), input_tokens=0, output_tokens=0, spend_usd=0.0)
        try:
            data = json.loads(self.budget_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return BudgetState(date=_today(), input_tokens=0, output_tokens=0, spend_usd=0.0)
        # Reset if it's a new day.
        if data.get("date") != _today():
            return BudgetState(date=_today(), input_tokens=0, output_tokens=0, spend_usd=0.0)
        return BudgetState(
            date=data["date"],
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            spend_usd=float(data.get("spend_usd", 0.0)),
        )

    def _save(self, state: BudgetState) -> None:
        self.budget_file.parent.mkdir(parents=True, exist_ok=True)
        self.budget_file.write_text(
            json.dumps({
                "date": state.date,
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "spend_usd": round(state.spend_usd, 6),
            }, indent=2),
            encoding="utf-8",
        )

    def record(self, input_tokens: int, output_tokens: int) -> BudgetState:
        """Add a tick's token usage and recompute today's spend."""
        s = self._load()
        s.input_tokens += input_tokens
        s.output_tokens += output_tokens
        s.spend_usd = (
            s.input_tokens * self.config.pricing.input_per_1m_usd / 1_000_000.0
            + s.output_tokens * self.config.pricing.output_per_1m_usd / 1_000_000.0
        )
        self._save(s)
        return s

    def exceeded(self) -> bool:
        """True if today's spend has met or exceeded the daily cap."""
        if self.config.daily_usd_cap <= 0:
            return False
        return self._load().spend_usd >= self.config.daily_usd_cap

    def status(self) -> BudgetState:
        return self._load()
