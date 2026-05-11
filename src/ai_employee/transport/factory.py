"""Build a Transport from TransportConfig."""
from __future__ import annotations

from ..config import TransportConfig
from .base import Transport
from .discord import DiscordTransport
from .stdout import StdoutTransport
from .webhook import WebhookTransport


def build_transport(config: TransportConfig, name: str = "agent") -> Transport:
    kind = config.kind.lower()
    if kind == "stdout":
        return StdoutTransport(name=name)
    if kind == "discord":
        return DiscordTransport(
            webhook_url_env=config.discord.webhook_url_env,
            username=config.discord.username or name,
        )
    if kind == "webhook":
        return WebhookTransport(
            url_env=config.webhook.url_env,
            format=config.webhook.format,
        )
    raise ValueError(f"Unknown transport.kind: {config.kind!r}")
