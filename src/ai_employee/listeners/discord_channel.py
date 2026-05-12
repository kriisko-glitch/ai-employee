"""Discord channel listener — poll a channel via the bot API.

Stdlib-only (urllib). No discord.py dependency. Reads the most recent N
messages; the daemon decides what to do with them (surface as observation,
adjust cadence, etc.).

This is the inbound twin of `transport/discord.py` (which posts outbound
via webhook).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DISCORD_API = "https://discord.com/api/v10"
USER_AGENT = "ai-employee/0.2 (+https://github.com/kriisko-glitch/ai-employee)"


@dataclass
class DiscordMessage:
    id: str
    author_id: int
    author_name: str
    content: str
    ts_iso: str
    is_bot: bool
    is_webhook: bool


class DiscordChannelListener:
    def __init__(self, channel_id: int, bot_token_env: str = "DISCORD_BOT_TOKEN",
                 user_id_to_watch: int = 0):
        if not channel_id:
            raise ValueError("channel_id is required for DiscordChannelListener")
        self.channel_id = channel_id
        self.user_id_to_watch = user_id_to_watch
        self.token = os.environ.get(bot_token_env, "").strip()
        if not self.token:
            raise RuntimeError(
                f"DiscordChannelListener: env var {bot_token_env!r} is empty"
            )

    def fetch_recent(self, limit: int = 8) -> list[DiscordMessage]:
        """GET the last `limit` messages, newest first."""
        url = f"{DISCORD_API}/channels/{self.channel_id}/messages?limit={limit}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bot {self.token}",
                "User-Agent": USER_AGENT,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            log.warning("Discord fetch_recent failed: HTTP %s", e.code)
            return []
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            log.warning("Discord fetch_recent failed: %s", e)
            return []

        out: list[DiscordMessage] = []
        for m in raw:
            try:
                author = m.get("author") or {}
                out.append(DiscordMessage(
                    id=m["id"],
                    author_id=int(author.get("id", 0)),
                    author_name=author.get("username", "?"),
                    content=m.get("content", ""),
                    ts_iso=m.get("timestamp", ""),
                    is_bot=bool(author.get("bot")),
                    is_webhook=bool(m.get("webhook_id")),
                ))
            except (KeyError, ValueError):
                continue
        return out

    def new_user_messages_since(self, since_id: Optional[str],
                                limit: int = 8) -> list[DiscordMessage]:
        """Return user messages from the watched user newer than `since_id`.

        Filters out:
          - bot posts (Director, Mike Ross's own webhook posts)
          - webhook posts
          - anything from author IDs != user_id_to_watch (when set)
        """
        recent = self.fetch_recent(limit=limit)
        # Discord returns newest first; reverse to chronological for easier comparison.
        recent.reverse()
        filtered: list[DiscordMessage] = []
        for m in recent:
            if m.is_bot or m.is_webhook:
                continue
            if self.user_id_to_watch and m.author_id != self.user_id_to_watch:
                continue
            if not m.content.strip():
                continue
            if since_id and m.id <= since_id:  # Discord IDs are snowflakes — string compare is fine
                continue
            filtered.append(m)
        return filtered

    def format_for_prompt(self, messages: list[DiscordMessage]) -> str:
        """Render channel context for inclusion in the agent's prompt."""
        if not messages:
            return "(channel quiet)"
        lines = []
        for m in messages:
            ts = m.ts_iso[:19] if m.ts_iso else "?"
            tag = "bot" if (m.is_bot or m.is_webhook) else "user"
            lines.append(f"  [{ts}] {tag}:{m.author_name}: {m.content}")
        return "\n".join(lines)


def most_recent_kris_age_seconds(messages: list[DiscordMessage],
                                 user_id: int) -> Optional[float]:
    """How long ago did the watched user last post? None if no post found.

    Used by cadence logic to speed up ticks when user is active.
    """
    for m in messages:
        if user_id and m.author_id != user_id:
            continue
        if m.is_bot or m.is_webhook:
            continue
        try:
            ts = datetime.fromisoformat(m.ts_iso.replace("Z", "+00:00"))
        except ValueError:
            continue
        return (datetime.now(timezone.utc) - ts).total_seconds()
    return None


def load_last_seen(path: Path) -> Optional[str]:
    """Read the most recent processed Discord message ID, if tracked."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("last_message_id")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_last_seen(path: Path, message_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"last_message_id": message_id}, indent=2),
        encoding="utf-8",
    )
