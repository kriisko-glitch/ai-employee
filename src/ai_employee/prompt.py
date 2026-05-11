"""Prompt assembly — combine SOUL + state + memory recall into a single prompt.

Kept deliberately simple. Users who want richer prompt construction can
subclass or replace this with their own builder.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import AgentConfig


SYSTEM_PROMPT_TEMPLATE = """\
You are {name}, an AI agent.

# Identity
{soul}

# Persistent memory (high-level notes you've kept across sessions)
{memory_md}

# Recent recall (valence-weighted; ↑ = hit/learning, ↓ = miss/warning)
{recall_block}

# Current state
state: {state}
intensity: {intensity}
since: {since_ts}
tick: {tick_count}
last_observation: {last_observation}

# Your job this turn
Respond with your inner reasoning and any output. If your emotional/cognitive
state has shifted, include a header on its own line:

[STATE intensity]    (e.g. [BUILDING 2], [STUCK 1], [REFLECTING 3])

Then a body. If you want to schedule the next tick differently, end with:

[next_tick_seconds: N]    (clamped to [{min_s}, {max_s}])
"""


def _read_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def build_prompt(
    config: AgentConfig,
    state: dict,
    recall_chunks: Optional[list[dict]] = None,
    extra_observation: Optional[str] = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_message) for the runner.

    `recall_chunks` is the output of memory.retrieve.retrieve(), already
    valence-weighted and ordered. Each chunk is a dict with at least
    `body` and `valence`.
    """
    soul = _read_or_empty(config.soul_file) or f"(SOUL.md not found — running with no identity declaration)"
    memory_md = _read_or_empty(config.memory_file) or "(empty)"

    recall_block = _format_recall(recall_chunks or [])

    system = SYSTEM_PROMPT_TEMPLATE.format(
        name=config.name,
        soul=soul,
        memory_md=memory_md,
        recall_block=recall_block,
        state=state.get("state", "BOREDOM"),
        intensity=state.get("intensity", 1),
        since_ts=state.get("since_ts", ""),
        tick_count=state.get("tick_count_in_state", 0),
        last_observation=state.get("last_observation", "(none)"),
        min_s=config.heartbeat.cadence.min_seconds,
        max_s=config.heartbeat.cadence.max_seconds,
    )

    user_parts = [
        f"It is {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
    ]
    if extra_observation:
        user_parts.append(f"Observation: {extra_observation}")
    user_parts.append(
        "Take a tick. Reflect, decide, act. Output as instructed in the system prompt."
    )

    return system, "\n\n".join(user_parts)


def _format_recall(chunks: list[dict]) -> str:
    if not chunks:
        return "(no prior memories surfaced)"
    lines = []
    valence_arrow = {
        "hit": "↑",
        "miss": "↓",
        "walkback": "↻",
        "unmarked": "·",
    }
    for c in chunks:
        v = c.get("valence", "unmarked")
        arrow = valence_arrow.get(v, "·")
        body = c.get("body", "").strip().replace("\n", " ")
        if len(body) > 280:
            body = body[:277] + "..."
        lines.append(f"  {arrow} [{v}] {body}")
    return "\n".join(lines)
