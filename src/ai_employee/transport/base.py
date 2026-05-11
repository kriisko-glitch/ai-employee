"""Transport protocol — anything that can deliver an agent's output."""
from __future__ import annotations

from typing import Protocol


class Transport(Protocol):
    def post(self, text: str) -> None:
        ...
