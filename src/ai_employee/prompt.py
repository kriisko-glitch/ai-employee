"""Prompt assembly — combine SOUL + state + memory recall + (v0.2) channel
context + wake context into a single system+user prompt pair.

The system prompt evolves with the agent's mode:
  - normal tick:   identity + memory + recall + state
  - dreaming tick: identity + memory + sleep ritual instructions
  - waking tick:   identity + memory + wake context block + light tick

Channel context (recent Discord messages) is injected when the listener
is enabled. The agent decides whether to respond to a new user message.
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
{channel_block}{wake_block}
# Current state
state: {state}
intensity: {intensity}
since: {since_ts}
tick: {tick_count}
last_observation: {last_observation}
{drive_hint}

# What you can and CAN'T do this tick

You are a text-only language model right now. You have NO tools in this
release:
- You CANNOT read files, list directories, or scan workspaces.
- You CANNOT run shell commands.
- You CANNOT fetch URLs or browse the web.
- You CANNOT execute code.

If you find yourself describing any of those actions ("i ran ls", "i read
file X", "i scanned the repo"), you are CONFABULATING. Stop, walk back,
and say so explicitly. A walkback is more valuable than a fabricated win.

What you CAN do:
- Think and reason about what your operator has told you.
- Remember things between ticks via your conversation history and memory DB.
- Speak in your channel via the POST block below.
- Decide your own state and pacing.

# Output format — STRICT

Produce your response in this exact structure. The daemon parses it.

[THINKING]
Your inner reasoning. What you notice, what you weigh, what you decide.
This will be retained in your conversation history so you stay coherent
across ticks — but it will NOT be sent to the channel. Be honest with
yourself here. No theatre. No performing.
[/THINKING]

[POST]
What you actually say in the channel. Concise. On-voice. Only what your
operator should see. If there is nothing worth saying this tick — no new
information, no question, no reply due — write exactly one word:

SILENT

Do not pad. Do not narrate. Do not include any [THINKING], [STATE], or
[next_tick_seconds] markers inside this block.
[/POST]

[STATE NAME intensity]    (e.g. [BUILDING 2], [REFLECTING 1], [BOREDOM 2])

[next_tick_seconds: N]     (integer, clamped to [{min_s}, {max_s}])

{kris_check_note}
"""


KRIS_CHECK_NOTE = """\
**⚠ CHANNEL MESSAGE CHECK — DO THIS FIRST:**
Scan the channel context above. If your operator posted anything AFTER your
most recent post, that is a direct message to you. Respond to it NOW, before
any other work. Keep the response concise and on-point.
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
    channel_context: Optional[str] = None,
    wake_context_block: Optional[str] = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_message) for the runner.

    `recall_chunks`     — valence-weighted memory recall (memory.retrieve)
    `channel_context`   — formatted recent channel messages (listener output)
    `wake_context_block` — full WAKE_PROMPT_BLOCK from memory.wake (only on WAKING)
    """
    soul = _read_or_empty(config.soul_file) or "(SOUL.md not found — running with no identity declaration)"
    memory_md = _read_or_empty(config.memory_file) or "(empty)"

    recall_block = _format_recall(recall_chunks or [])

    channel_block = ""
    if channel_context:
        channel_block = f"\n# Recent channel posts\n```\n{channel_context}\n```\n"

    wake_block = ""
    if wake_context_block:
        wake_block = f"\n# WAKE CONTEXT\n{wake_context_block}\n"

    kris_check = KRIS_CHECK_NOTE if channel_context else ""

    drive_hint = _drive_hint(state.get("state", ""), config)

    system = SYSTEM_PROMPT_TEMPLATE.format(
        name=config.name,
        soul=soul,
        memory_md=memory_md,
        recall_block=recall_block,
        channel_block=channel_block,
        wake_block=wake_block,
        state=state.get("state", "BOREDOM"),
        intensity=state.get("intensity", 1),
        since_ts=state.get("since_ts", ""),
        tick_count=state.get("tick_count_in_state", 0),
        last_observation=state.get("last_observation", "(none)"),
        drive_hint=drive_hint,
        min_s=config.heartbeat.cadence.min_seconds,
        max_s=config.heartbeat.cadence.max_seconds,
        kris_check_note=kris_check,
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


def _drive_hint(state_name: str, config: AgentConfig) -> str:
    """Tell the agent which drive is in play given its current state."""
    if not config.memory.parameterization.drive_bias.enabled or not state_name:
        return ""
    bias = config.memory.parameterization.drive_bias
    su = state_name.upper()
    if su in (s.upper() for s in bias.competence_states):
        return "drive: competence — you're in a state that pulls toward skill, execution, shipping."
    if su in (s.upper() for s in bias.autonomy_states):
        return "drive: autonomy — you're in a state that pulls toward self-direction, choosing what to do."
    if su in (s.upper() for s in bias.relatedness_states):
        return "drive: relatedness — you're in a state that pulls toward connection, exchange, contract."
    return ""


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
        # Show SDT axes if any are non-zero (legacy chunks have them all at 0.0).
        sdt = ""
        a, c_, r = c.get("autonomy", 0), c.get("competence", 0), c.get("relatedness", 0)
        if (a or c_ or r):
            sdt = f" A{a:.1f}C{c_:.1f}R{r:.1f}"
        lines.append(f"  {arrow} [{v}{sdt}] {body}")
    return "\n".join(lines)
