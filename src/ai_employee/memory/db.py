"""SQLite + sqlite-vec memory store.

One DB file per agent at <workspace>/memory.db. Schema:

    chunk              — every memory unit (free-form notes, observations, posts)
    chunk_vec          — virtual table; sqlite-vec embeddings
    solution_attempt   — explicit task→approach→outcome rows (the structured
                          half of "hit the mark vs. not")
    valence_override   — manual overrides to auto-tagged valence

The embedding model name is stored per chunk so future migrations can re-embed
selectively.
"""
from __future__ import annotations

import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .valence import VALENCE_UNMARKED


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pack_vec(vec: list[float]) -> bytes:
    """sqlite-vec stores vectors as packed float32 bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def connect(db_file: Path) -> sqlite3.Connection:
    """Open the agent DB with sqlite-vec loaded."""
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection, embedding_dim: int) -> None:
    """Create tables if they don't exist."""
    conn.executescript(f"""
    CREATE TABLE IF NOT EXISTS chunk (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ts              TEXT NOT NULL,
        ingested_ts     TEXT NOT NULL,
        source          TEXT,
        body            TEXT NOT NULL,
        valence         TEXT NOT NULL DEFAULT '{VALENCE_UNMARKED}',
        weight          REAL NOT NULL DEFAULT 0.5,
        embedding_model TEXT NOT NULL,
        last_recalled_ts TEXT,
        recall_count    INTEGER NOT NULL DEFAULT 0,
        notes           TEXT
    );
    CREATE INDEX IF NOT EXISTS chunk_ts_idx ON chunk(ts DESC);
    CREATE INDEX IF NOT EXISTS chunk_valence_idx ON chunk(valence);

    CREATE TABLE IF NOT EXISTS solution_attempt (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT NOT NULL,
        task        TEXT NOT NULL,
        approach    TEXT NOT NULL,
        outcome     TEXT NOT NULL,           -- hit | miss | walkback
        lesson      TEXT,
        chunk_id    INTEGER REFERENCES chunk(id)
    );
    CREATE INDEX IF NOT EXISTS attempt_ts_idx ON solution_attempt(ts DESC);
    CREATE INDEX IF NOT EXISTS attempt_outcome_idx ON solution_attempt(outcome);

    CREATE TABLE IF NOT EXISTS valence_override (
        chunk_id    INTEGER PRIMARY KEY REFERENCES chunk(id),
        valence     TEXT NOT NULL,
        ts          TEXT NOT NULL,
        reason      TEXT
    );
    """)
    # sqlite-vec virtual table — separate because it needs the extension loaded.
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
            embedding float[{embedding_dim}]
        )
    """)
    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    body: str,
    embedding: list[float],
    embedding_model: str,
    source: Optional[str] = None,
    valence: str = VALENCE_UNMARKED,
    weight: float = 0.5,
    ts: Optional[str] = None,
) -> int:
    """Insert a chunk and its embedding. Returns the chunk id."""
    ts = ts or _now()
    cur = conn.execute(
        """
        INSERT INTO chunk (ts, ingested_ts, source, body, valence, weight, embedding_model)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, _now(), source, body, valence, weight, embedding_model),
    )
    chunk_id = cur.lastrowid
    conn.execute(
        "INSERT INTO chunk_vec(rowid, embedding) VALUES (?, ?)",
        (chunk_id, _pack_vec(embedding)),
    )
    conn.commit()
    return chunk_id


def set_valence(conn: sqlite3.Connection, chunk_id: int, valence: str,
                reason: Optional[str] = None) -> None:
    """Manually override a chunk's valence."""
    conn.execute("UPDATE chunk SET valence = ? WHERE id = ?", (valence, chunk_id))
    conn.execute(
        """
        INSERT INTO valence_override (chunk_id, valence, ts, reason)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            valence = excluded.valence, ts = excluded.ts, reason = excluded.reason
        """,
        (chunk_id, valence, _now(), reason),
    )
    conn.commit()


def record_attempt(
    conn: sqlite3.Connection,
    task: str,
    approach: str,
    outcome: str,
    lesson: Optional[str] = None,
    chunk_id: Optional[int] = None,
) -> int:
    """Record an explicit solution attempt — the structured half of the
    'hit the mark' model. Use this when you want a queryable, labeled record
    of how a particular problem was solved (or wasn't).
    """
    cur = conn.execute(
        """
        INSERT INTO solution_attempt (ts, task, approach, outcome, lesson, chunk_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (_now(), task, approach, outcome, lesson, chunk_id),
    )
    conn.commit()
    return cur.lastrowid


def vec_search(conn: sqlite3.Connection, query_vec: list[float],
               top_k: int = 20) -> list[sqlite3.Row]:
    """kNN over chunk_vec. Returns chunk rows joined with their cosine distance.

    Note: sqlite-vec returns L2 distance by default on normalized vectors,
    which is monotone with (1 - cosine) — small distance = more similar.
    """
    rows = conn.execute(
        """
        SELECT c.*, v.distance
        FROM chunk_vec v
        JOIN chunk c ON c.id = v.rowid
        WHERE v.embedding MATCH ?
        ORDER BY v.distance
        LIMIT ?
        """,
        (_pack_vec(query_vec), top_k),
    ).fetchall()
    return rows


def mark_recalled(conn: sqlite3.Connection, chunk_id: int) -> None:
    """Bump recall_count and last_recalled_ts for a chunk."""
    conn.execute(
        """
        UPDATE chunk
        SET recall_count = recall_count + 1, last_recalled_ts = ?
        WHERE id = ?
        """,
        (_now(), chunk_id),
    )
    conn.commit()
