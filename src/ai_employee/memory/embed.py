"""Sentence-transformers wrapper.

Lazy-loaded: the model only loads when first needed. This keeps `aie` startup
fast for stateless one-shot ticks (`aie tick`) where memory isn't touched.
"""
from __future__ import annotations

from typing import Optional


class Embedder:
    """Lazy wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch  # noqa: F401
            if hasattr(__import__("torch").backends, "mps") and __import__("torch").backends.mps.is_available():
                return "mps"
            if __import__("torch").cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32
        )
        return [v.tolist() for v in vecs]
