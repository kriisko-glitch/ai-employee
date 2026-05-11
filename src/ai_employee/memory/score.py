"""Ingest helper — write a chunk to memory with auto-valence tagging.

Convenience wrapper around db.insert_chunk + embed + valence.tag_valence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import MemoryConfig
from .db import connect, init_schema, insert_chunk, record_attempt
from .embed import Embedder
from .valence import tag_valence, VALENCE_UNMARKED


def remember(
    db_file: Path,
    body: str,
    config: MemoryConfig,
    source: Optional[str] = None,
    valence: Optional[str] = None,
) -> int:
    """Embed and store a chunk. If `valence` is None and auto_tag is on, run the
    heuristic tagger. Returns the chunk id.
    """
    if valence is None:
        valence = (
            tag_valence(body)
            if config.parameterization.auto_tag.enabled
            else VALENCE_UNMARKED
        )

    embedder = Embedder(
        model_name=config.storage.embedding.model,
        device=config.storage.embedding.device,
    )
    vec = embedder.embed(body)

    conn = connect(db_file)
    try:
        init_schema(conn, config.storage.embedding.dim)
        return insert_chunk(
            conn,
            body=body,
            embedding=vec,
            embedding_model=config.storage.embedding.model,
            source=source,
            valence=valence,
        )
    finally:
        conn.close()


def log_attempt(
    db_file: Path,
    task: str,
    approach: str,
    outcome: str,
    config: MemoryConfig,
    lesson: Optional[str] = None,
) -> int:
    """Record a solution_attempt row + a paired chunk linking to it.

    The chunk's body summarizes task+approach+outcome+lesson for vector recall;
    the structured row supports exact queries like "all misses on task X".
    """
    body = f"Task: {task}\nApproach: {approach}\nOutcome: {outcome}"
    if lesson:
        body += f"\nLesson: {lesson}"

    embedder = Embedder(
        model_name=config.storage.embedding.model,
        device=config.storage.embedding.device,
    )
    vec = embedder.embed(body)

    conn = connect(db_file)
    try:
        init_schema(conn, config.storage.embedding.dim)
        chunk_id = insert_chunk(
            conn,
            body=body,
            embedding=vec,
            embedding_model=config.storage.embedding.model,
            source="solution_attempt",
            valence=outcome,  # outcome literally is the valence label
        )
        record_attempt(conn, task=task, approach=approach, outcome=outcome,
                       lesson=lesson, chunk_id=chunk_id)
        return chunk_id
    finally:
        conn.close()
