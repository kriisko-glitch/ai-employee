"""WAKING — the one-tick grace state after dreaming.

When the agent wakes, the daemon constructs a "what you carried in" block
from the highest-scoring chunks (valence × SDT × decay) and the latest
sleep_event's carry_forward. This gets injected into the prompt for the
first post-sleep tick.

The agent is told it's WAKING. It reads the wake context, gets oriented,
and produces a brief acknowledgement before resuming normal ticks. This
is the "groggy on waking" period — concise, calibrated, not yet
full-throttle.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..config import AgentConfig, MemoryConfig
from .db import connect, init_schema
from .retrieve import _decay_factor, _valence_weight

log = logging.getLogger(__name__)


WAKE_PROMPT_BLOCK = """\
**You are WAKING.** You just finished a sleep/dreaming consolidation. Your
conversation history is empty. Your identity (SOUL, MEMORY.md) is intact.
You're a little groggy by design — take this tick to read in, get oriented,
and post a short acknowledgement. Don't try to ship anything heavy this
tick. Save that for the next tick when you're fully back online.

**What you carried in from before sleep:**

{wake_context}

This tick: read the above, post a brief on-voice wake message (2-4
sentences). Then set state to the appropriate post-wake state — usually
REFLECTING or BOREDOM, not the same state you slept from. Take a normal
delay (180-300s) so you settle.
"""


def build_wake_context(config: AgentConfig, top_k: int = 8) -> str:
    """Pull the top scored chunks from the most recent sleep_event + the
    written carry_forward summary, format as a markdown block.
    """
    if not config.memory.enabled:
        return "(memory disabled — no wake context)"
    if not config.memory_db_file.exists():
        return "(no memory yet — this is your first wake)"

    conn = connect(config.memory_db_file)
    try:
        init_schema(conn, config.memory.storage.embedding.dim)

        # Latest sleep_event's carry_forward (most recent dream summary).
        carry_row = conn.execute(
            "SELECT carry_forward, ts, moments_stored "
            "FROM sleep_event ORDER BY id DESC LIMIT 1"
        ).fetchone()

        # Top scored sleep_carryforward chunks (the dreamed moments).
        chunks = conn.execute(
            """
            SELECT id, ts, body, valence, autonomy, competence, relatedness
            FROM chunk
            WHERE source = 'sleep_carryforward'
            ORDER BY id DESC
            LIMIT ?
            """,
            (top_k * 3,),
        ).fetchall()
    finally:
        conn.close()

    if not chunks and not carry_row:
        return "(no sleep event recorded yet)"

    # Rank by valence × SDT × decay.
    ranked: list[tuple[float, dict]] = []
    for c in chunks:
        d = _decay_factor(c["ts"], config.memory.parameterization.decay.half_life_days)
        v = _valence_weight(c["valence"], config.memory)
        sdt_avg = (c["autonomy"] + c["competence"] + c["relatedness"]) / 3.0
        # SDT acts as an attention multiplier on what surfaces at wake.
        score = v * d * (1.0 + sdt_avg)
        ranked.append((score, dict(c)))
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:top_k]

    parts: list[str] = []
    if carry_row and carry_row["carry_forward"]:
        parts.append("**Last consolidation summary:**")
        parts.append(carry_row["carry_forward"].strip())
        parts.append("")

    if top:
        parts.append("**Most-load-bearing moments from before sleep:**")
        for _, c in top:
            sdt = f"A={c['autonomy']:.1f} C={c['competence']:.1f} R={c['relatedness']:.1f}"
            body = c["body"].replace("\n", " ").strip()
            if len(body) > 240:
                body = body[:237] + "..."
            parts.append(f"  - [{c['valence']} | {sdt}] {body}")

    return "\n".join(parts)


def build_wake_prompt_block(wake_context: str) -> str:
    return WAKE_PROMPT_BLOCK.format(wake_context=wake_context)
