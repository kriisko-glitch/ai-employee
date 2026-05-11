"""Model runners — abstract over any OpenAI-compatible API."""

from .base import Runner, RunResult
from .openai_compatible import OpenAICompatibleRunner, build_runner

__all__ = ["Runner", "RunResult", "OpenAICompatibleRunner", "build_runner"]
