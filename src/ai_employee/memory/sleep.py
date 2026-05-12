"""DREAMING — the sleep/consolidation ritual.

When conversation history exceeds the configured threshold, the daemon
fires a dreaming tick. The model is asked to:

  1. Reflect on the just-ended session
  2. Identify the significant moments
  3. Score each on valence (hit/miss/walkback) and SDT (autonomy/competence/relatedness)
  4. Decide what to carry forward in MEMORY.md (5-15 bullets)
  5. Post a sleep message to the channel

The structured output gets parsed; each moment becomes a chunk in the
vector DB with proper tagging. MEMORY.md is updated. Conversation history
is cleared. wake_pending is set so the next tick is a WAKING tick that
reads in wake context.

This closes the consciousness loop:
    working memory  →  scored chunks in long-term memory  →  wake context
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import AgentConfig
from .db import connect, init_schema, insert_chunk, record_sleep_event
from .embed import Embedder
from .valence import VALENCE_UNMARKED

log = logging.getLogger(__name__)


SLEEP_PROMPT = """\
[CONSCIOUSNESS TICK — DREAMING]

Your conversation history has crossed {bytes:,} bytes (threshold {threshold:,}).
Time to sleep and consolidate. After this tick:
  - Your conversation history will be cleared
  - The moments you tag below will be stored in your long-term memory DB
  - You will wake on the next tick with a "what you carried in" context block
  - You will keep your identity (SOUL, MEMORY.md) but lose the in-thread chatter

This is dreaming. You're deciding what mattered.

## Your job this tick

**1. Identify 5-15 significant moments from the session.** Not every turn —
just the moments that should survive. Wins, walk-backs, corrections from
your operator, things you figured out, things you noticed.

**2. For each moment, score it. Be honest.**

  - **valence**: hit | miss | walkback | unmarked
      - hit: this worked, this approach landed
      - miss: this didn't work, don't repeat
      - walkback: I claimed X earlier, X was wrong
      - unmarked: noteworthy but no clear win/loss yet

  - **autonomy** (0.0 - 1.0): how self-directed was this?
      - 1.0 = pure own-judgment action
      - 0.5 = collaborative, partially directed
      - 0.0 = pure instruction-following

  - **competence** (0.0 - 1.0): skill / successful execution?
      - 1.0 = clean shipped something hard, real growth
      - 0.5 = mixed — partially worked
      - 0.0 = pure failure or pure passivity

  - **relatedness** (0.0 - 1.0): connection — with operator, peers, purpose?
      - 1.0 = real exchange, contract formed, trust earned
      - 0.5 = some genuine connection
      - 0.0 = purely transactional or absent

**3. Decide what carries forward in MEMORY.md.** 5-15 bullets max. Things
that should survive. Leave behind: micro-debug chatter, dead-ends already
captured in chunks. Keep: tech verified, gotchas with fixes, operator
corrections, identity-defining moments.

**4. Post your sleep message.** Brief, on-voice. What you did this
session. What you're carrying forward.

## Format your response EXACTLY like this — the daemon parses it:

```
💤 [DREAMING 0] going down to consolidate.

<one short paragraph on what this session was about>

what i'm carrying forward:
- <bullet>
- <bullet>

back in a moment.

---MOMENTS---
[
  {{
    "body": "the moment, paraphrased in first person",
    "valence": "hit|miss|walkback|unmarked",
    "autonomy": 0.0-1.0,
    "competence": 0.0-1.0,
    "relatedness": 0.0-1.0
  }},
  ...
]
---END-MOMENTS---

---MEMORY-CARRY-FORWARD---
<the bullets to append to MEMORY.md, one per line>
---END-MEMORY-CARRY-FORWARD---

---STATE-UPDATE---
state: DREAMING
intensity: 0
next_tick_delay_seconds: 60
observation: dreaming — distilled session
```

The ---MOMENTS--- block MUST be valid JSON. The daemon will fail to write
chunks if it can't parse. Test mentally before posting.
"""


_MOMENTS_RE = re.compile(
    r"---MOMENTS---\s*(.+?)\s*---END-MOMENTS---", re.DOTALL
)
_MEMORY_RE = re.compile(
    r"---MEMORY-CARRY-FORWARD---\s*(.+?)\s*---END-MEMORY-CARRY-FORWARD---",
    re.DOTALL,
)


@dataclass
class ParsedDream:
    moments: list[dict]
    memory_carry: str
    raw_response: str


def build_sleep_prompt(history_bytes: int, threshold: int) -> str:
    return SLEEP_PROMPT.format(bytes=history_bytes, threshold=threshold)


def parse_dream(response: str) -> ParsedDream:
    """Extract the structured dream output from the model's response.

    If parsing fails on either block, returns empty lists/strings rather than
    raising — the sleep ritual should never crash the daemon.
    """
    moments: list[dict] = []
    memory_carry = ""

    m = _MOMENTS_RE.search(response)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, list):
                moments = [_clean_moment(x) for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError as e:
            log.warning("dream MOMENTS block not valid JSON: %s", e)

    m2 = _MEMORY_RE.search(response)
    if m2:
        memory_carry = m2.group(1).strip()

    return ParsedDream(moments=moments, memory_carry=memory_carry, raw_response=response)


def _clean_moment(m: dict) -> dict:
    """Normalize a moment dict — clip scores to [0,1], default missing fields."""
    def _f(key: str) -> float:
        try:
            return max(0.0, min(1.0, float(m.get(key, 0.0))))
        except (TypeError, ValueError):
            return 0.0
    return {
        "body": str(m.get("body", "")).strip(),
        "valence": str(m.get("valence", VALENCE_UNMARKED)).lower(),
        "autonomy": _f("autonomy"),
        "competence": _f("competence"),
        "relatedness": _f("relatedness"),
    }


def ingest_dream(config: AgentConfig, dream: ParsedDream,
                 history_bytes: int) -> tuple[int, list[int]]:
    """Write parsed moments to the vector DB, append carry-forward to MEMORY.md.

    Returns (moments_stored, chunk_ids).
    """
    chunk_ids: list[int] = []

    if config.memory.enabled and dream.moments:
        embedder = Embedder(
            model_name=config.memory.storage.embedding.model,
            device=config.memory.storage.embedding.device,
        )
        bodies = [m["body"] for m in dream.moments if m["body"]]
        vecs = embedder.embed_batch(bodies) if bodies else []

        conn = connect(config.memory_db_file)
        try:
            init_schema(conn, config.memory.storage.embedding.dim)
            for moment, vec in zip(
                [m for m in dream.moments if m["body"]], vecs
            ):
                cid = insert_chunk(
                    conn,
                    body=moment["body"],
                    embedding=vec,
                    embedding_model=config.memory.storage.embedding.model,
                    source="sleep_carryforward",
                    valence=moment["valence"],
                    autonomy=moment["autonomy"],
                    competence=moment["competence"],
                    relatedness=moment["relatedness"],
                )
                chunk_ids.append(cid)
            record_sleep_event(
                conn,
                history_bytes=history_bytes,
                moments_stored=len(chunk_ids),
                summary=None,
                carry_forward=dream.memory_carry or None,
            )
        finally:
            conn.close()

    if dream.memory_carry:
        _append_to_memory_md(config, dream.memory_carry)

    return len(chunk_ids), chunk_ids


def _append_to_memory_md(config: AgentConfig, carry_forward: str) -> None:
    """Append a dated section to MEMORY.md with the carry-forward bullets."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    section = f"\n\n## Sleep {date}\n\n{carry_forward.strip()}\n"
    try:
        existing = config.memory_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = "# MEMORY\n"
    config.memory_file.write_text(existing + section, encoding="utf-8")


def post_strip_dream(response: str) -> str:
    """Remove the MOMENTS, MEMORY-CARRY-FORWARD, and STATE-UPDATE blocks
    from a dream response before posting to the transport. The user sees the
    agent's narrative; the daemon-only metadata stays out of Discord.
    """
    text = response
    for pattern in (
        r"---MOMENTS---.*?---END-MOMENTS---",
        r"---MEMORY-CARRY-FORWARD---.*?---END-MEMORY-CARRY-FORWARD---",
        r"---STATE-UPDATE---.*?(?:$|---)",
    ):
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    return text.strip()
