"""OpenAI-compatible runner — works for OpenAI, Anthropic (via /v1/), DeepSeek,
Groq, Cerebras, OpenRouter, Ollama, LM Studio, llama.cpp server, and anything
else that speaks Chat Completions.

The OpenAI Python SDK accepts a `base_url` override, so a single class covers
the entire ecosystem. Per-provider quirks (if any) are isolated in build_runner.
"""
from __future__ import annotations

from typing import Optional

from openai import OpenAI

from ..config import ModelConfig, get_api_key
from .base import Runner, RunResult


class OpenAICompatibleRunner:
    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def run(self, system: str, user: str) -> RunResult:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        return RunResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            model_id=self.model_id,
            raw=response,
        )


def build_runner(config: ModelConfig) -> Runner:
    """Construct a Runner from a ModelConfig.

    Raises RuntimeError if the required API key env var is missing.
    """
    api_key = get_api_key(config.api_key_env)
    return OpenAICompatibleRunner(
        model_id=config.model_id,
        api_key=api_key,
        base_url=config.resolved_base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=float(config.timeout_seconds),
    )
