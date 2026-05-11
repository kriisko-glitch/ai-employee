"""Valence-weighted, decay-adjusted retrieval over the chunk store."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import MemoryConfig
from .db import connect, init_schema, vec_search, mark_recalled
from .embed import Embedder
from .valence import (
    VALENCE_HIT, VALENCE_MISS, VALENCE_WALKBACK, VALENCE_UNMARKED,
)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _decay_factor(chunk_ts: Optional[str], half_life_days: float) -> float:
    """Exponential decay: weight halves every `half_life_days`."""
    dt = _parse_iso(chunk_ts)
    if dt is None or half_life_days <= 0:
        return 1.0
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / half_life_days)


def _valence_weight(valence: str, config: MemoryConfig) -> float:
    p = config.parameterization.retrieval
    return {
        VALENCE_HIT: p.boost_hits,
        VALENCE_MISS: p.suppress_misses,
        VALENCE_WALKBACK: p.walkback_weight,
        VALENCE_UNMARKED: 1.0,
    }.get(valence, 1.0)


def retrieve(
    db_file: Path,
    query: str,
    config: MemoryConfig,
) -> list[dict]:
    """Search memory; return top-K chunks ordered by valence-and-decay-weighted score.

    Score formula: sim × valence_weight × decay_factor, where
    sim = max(0, 1 - vec_distance) (sqlite-vec returns L2 on normalized vecs).
    """
    if not db_file.exists():
        return []

    embedder = Embedder(
        model_name=config.storage.embedding.model,
        device=config.storage.embedding.device,
    )
    qvec = embedder.embed(query)

    conn = connect(db_file)
    try:
        init_schema(conn, config.storage.embedding.dim)
        # Pull a wider net than top_k so re-ranking has material to work with.
        candidates = vec_search(conn, qvec, top_k=max(config.parameterization.retrieval.top_k * 3, 20))

        scored: list[tuple[float, dict]] = []
        for row in candidates:
            sim = max(0.0, 1.0 - float(row["distance"]))
            decay = (
                _decay_factor(row["ts"], config.parameterization.decay.half_life_days)
                if config.parameterization.decay.enabled
                else 1.0
            )
            vw = _valence_weight(row["valence"], config)
            score = sim * vw * decay
            scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: config.parameterization.retrieval.top_k]

        for _, chunk in top:
            mark_recalled(conn, chunk["id"])

        return [{**chunk, "score": s} for s, chunk in top]
    finally:
        conn.close()
