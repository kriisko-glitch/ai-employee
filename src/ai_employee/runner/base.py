"""Runner protocol — what every model adapter must provide.

Swap providers by swapping the runner. Everything downstream (tick loop,
budget tracker, transport) is provider-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RunResult:
    """One round-trip with the model."""
    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    raw: object = None  # provider-specific response; opaque to callers


class Runner(Protocol):
    """A model runner. Implementations: OpenAICompatibleRunner, plus any mocks."""

    def run(self, system: str, user: str) -> RunResult:
        ...
