"""Parse the structured agent response into its components.

Expected format:

    [THINKING]
    internal reasoning
    [/THINKING]

    [POST]
    what to post in the channel (or SILENT)
    [/POST]

    [STATE NAME intensity]
    [next_tick_seconds: N]

When a block is missing, we degrade gracefully:
  - no THINKING → treat the whole response as POST content (legacy v0.1/early-v0.2)
  - no POST → use the stripped response as POST
  - POST == "SILENT" → don't post anything

The full raw response is always retained — it goes into the conversation
history so the model sees its own prior reasoning on the next tick.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_THINKING_RE = re.compile(r"\[THINKING\]\s*(.*?)\s*\[/THINKING\]", re.DOTALL | re.IGNORECASE)
# Closed POST block (preferred).
_POST_CLOSED_RE = re.compile(r"\[POST\]\s*(.*?)\s*\[/POST\]", re.DOTALL | re.IGNORECASE)
# Unterminated POST: opens with [POST], runs until [STATE], [next_tick…], or EOF.
# Robust against models that forget the closing tag.
_POST_OPEN_RE = re.compile(
    r"\[POST\]\s*(.*?)(?=\[/POST\]|\[[A-Z_]+\s+\d+\]|\[next_tick_seconds:|$)",
    re.DOTALL | re.IGNORECASE,
)
_STATE_HEADER_RE = re.compile(r"\[([A-Z_]+)\s+(\d+)\]")
_NEXT_TICK_RE = re.compile(r"\[next_tick_seconds:\s*(\d+)\]", re.IGNORECASE)
_SILENT_RE = re.compile(r"^\s*SILENT\s*\.?\s*$", re.IGNORECASE)


@dataclass
class ParsedResponse:
    raw: str                 # the full response (goes into conversation history)
    thinking: Optional[str]  # extracted inner monologue (may be None if not present)
    post: Optional[str]      # extracted channel content (None means don't post)
    state: Optional[str]
    intensity: Optional[int]
    next_tick_seconds: Optional[int]
    is_silent: bool          # True if POST == "SILENT"

    @property
    def has_structured_format(self) -> bool:
        return self.thinking is not None or self.post is not None or self.is_silent


def parse_response(text: str) -> ParsedResponse:
    """Pull THINKING / POST / STATE / next_tick out of an agent response."""
    raw = text or ""

    thinking_match = _THINKING_RE.search(raw)
    thinking = thinking_match.group(1).strip() if thinking_match else None

    # Try closed [POST]...[/POST] first; fall back to open-ended [POST]... if
    # the model forgot the closer (a common model failure mode).
    post_match = _POST_CLOSED_RE.search(raw)
    if not post_match:
        post_match = _POST_OPEN_RE.search(raw)

    if post_match:
        post_body = post_match.group(1).strip()
        is_silent = bool(_SILENT_RE.match(post_body))
        post = None if is_silent else post_body
    else:
        # No [POST] marker at all — fall back to whole response minus
        # thinking/state markers (true legacy behavior).
        is_silent = False
        post = _legacy_strip(raw)

    state_match = _STATE_HEADER_RE.search(raw)
    state = state_match.group(1) if state_match else None
    try:
        intensity = int(state_match.group(2)) if state_match else None
    except (TypeError, ValueError):
        intensity = None

    next_match = _NEXT_TICK_RE.search(raw)
    try:
        next_tick = int(next_match.group(1)) if next_match else None
    except (TypeError, ValueError):
        next_tick = None

    return ParsedResponse(
        raw=raw,
        thinking=thinking,
        post=post,
        state=state,
        intensity=intensity,
        next_tick_seconds=next_tick,
        is_silent=is_silent,
    )


def _legacy_strip(text: str) -> str:
    """Strip THINKING blocks and metadata markers from a legacy-format response
    so something postable remains. Used when no POST block is found."""
    cleaned = _THINKING_RE.sub("", text)
    # Remove the structured markers themselves, leave the body text alone.
    cleaned = re.sub(r"\[STATE[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[next_tick_seconds:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[POST\]|\[/POST\]", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()
