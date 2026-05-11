"""Discord webhook transport.

Uses urllib (stdlib) so we don't pull discord.py for a one-line POST.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

DISCORD_MAX_CONTENT = 2000  # Discord message length cap


class DiscordTransport:
    def __init__(self, webhook_url_env: str = "DISCORD_WEBHOOK_URL",
                 username: Optional[str] = None):
        self.webhook_url = os.environ.get(webhook_url_env, "").strip()
        if not self.webhook_url:
            raise RuntimeError(
                f"Discord transport configured but {webhook_url_env} is not set."
            )
        self.username = username

    def post(self, text: str) -> None:
        # Discord caps at 2000 chars per message; split on paragraph boundaries.
        chunks = _split_for_discord(text, DISCORD_MAX_CONTENT)
        for chunk in chunks:
            payload = {"content": chunk}
            if self.username:
                payload["username"] = self.username
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()


def _split_for_discord(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    remaining = text
    while len(remaining) > limit:
        # Prefer to break at the last paragraph or newline within limit.
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        out.append(remaining[:cut])
        remaining = remaining[cut:].lstrip()
    if remaining:
        out.append(remaining)
    return out
