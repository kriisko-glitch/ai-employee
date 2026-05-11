"""Output transports — where the agent posts its turn."""

from .base import Transport
from .stdout import StdoutTransport
from .discord import DiscordTransport
from .webhook import WebhookTransport
from .factory import build_transport

__all__ = [
    "Transport",
    "StdoutTransport",
    "DiscordTransport",
    "WebhookTransport",
    "build_transport",
]
