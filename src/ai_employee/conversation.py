"""Conversation history — working memory across ticks.

A JSONL file per agent at <workspace>/conversation.jsonl. Each tick appends
(user, assistant) turns. The model sees the recent N turns in its system
prompt so it doesn't restart cold every tick.

Cleared at sleep/dreaming when the file crosses `compact.threshold_bytes`.
Sleep does NOT delete content — it distills it into SDT-scored chunks in
the vector DB, then clears the JSONL.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Turn:
    role: str           # "user" | "assistant"
    content: str
    ts: str
    source: Optional[str] = None   # "tick" | "discord" | "manual"
    meta: Optional[dict] = None    # arbitrary; e.g. discord_message_id

    def to_jsonl(self) -> str:
        d = {"role": self.role, "content": self.content, "ts": self.ts}
        if self.source:
            d["source"] = self.source
        if self.meta:
            d["meta"] = self.meta
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "Turn":
        d = json.loads(line)
        return cls(
            role=d["role"],
            content=d["content"],
            ts=d["ts"],
            source=d.get("source"),
            meta=d.get("meta"),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_turn(path: Path, role: str, content: str,
                source: Optional[str] = None, meta: Optional[dict] = None) -> Turn:
    """Append a single turn to the JSONL file. Creates the file if missing."""
    turn = Turn(role=role, content=content, ts=_now(), source=source, meta=meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(turn.to_jsonl() + "\n")
    return turn


def load_all(path: Path) -> list[Turn]:
    if not path.exists():
        return []
    out: list[Turn] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Turn.from_jsonl(line))
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def load_recent(path: Path, n: int) -> list[Turn]:
    """Return the most recent N turns. Cheap for small files."""
    turns = load_all(path)
    return turns[-n:] if n > 0 else turns


def size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def clear(path: Path) -> None:
    """Truncate the conversation file. Used after a successful sleep
    consolidation (chunks already written to DB).
    """
    if path.exists():
        # Atomic truncate via rename to a backup that gets deleted, so a
        # crash mid-clear leaves the old file recoverable.
        backup = path.with_suffix(path.suffix + ".pre-sleep")
        try:
            os.replace(path, backup)
            backup.unlink()
        except OSError:
            # Fallback: simple truncate.
            try:
                path.write_text("", encoding="utf-8")
            except OSError:
                pass


def to_message_list(turns: list[Turn]) -> list[dict]:
    """Convert turns to the OpenAI chat-completions messages shape."""
    return [{"role": t.role, "content": t.content} for t in turns]
