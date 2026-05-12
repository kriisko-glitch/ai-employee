"""Memory DB — schema init + vec_search regression.

Avoids the sentence-transformers dependency by inserting hand-crafted vectors
directly. This is the test that would have caught the sqlite-vec 0.1.9 kNN bug.
"""
import importlib.util
from pathlib import Path

import pytest

# Skip entire module if sqlite-vec isn't installed (core-only install).
if importlib.util.find_spec("sqlite_vec") is None:
    pytest.skip("sqlite-vec not installed", allow_module_level=True)

from ai_employee.memory.db import (
    connect, init_schema, insert_chunk, vec_search, set_valence,
)


DIM = 4  # tiny vectors keep the test readable


def _vec(*values: float) -> list[float]:
    assert len(values) == DIM
    return list(values)


def test_init_schema_creates_tables(tmp_path: Path):
    conn = connect(tmp_path / "m.db")
    try:
        init_schema(conn, DIM)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "chunk" in names
        assert "solution_attempt" in names
        assert "valence_override" in names
    finally:
        conn.close()


def test_insert_and_vec_search_roundtrip(tmp_path: Path):
    conn = connect(tmp_path / "m.db")
    try:
        init_schema(conn, DIM)

        # Three chunks, distinct embeddings.
        id_a = insert_chunk(conn, "alpha body", _vec(1, 0, 0, 0), "test", valence="hit")
        id_b = insert_chunk(conn, "beta body",  _vec(0, 1, 0, 0), "test", valence="miss")
        id_c = insert_chunk(conn, "gamma body", _vec(0, 0, 1, 0), "test", valence="unmarked")

        # Query closest to alpha.
        rows = vec_search(conn, _vec(0.99, 0.01, 0, 0), top_k=3)
        assert len(rows) == 3
        # Nearest result should be alpha (smallest distance).
        assert rows[0]["id"] == id_a
        assert rows[0]["body"] == "alpha body"

        # top_k=1 must work (the regression case — sqlite-vec 0.1.9 requires
        # LIMIT inside the vec0 scan, not after a join).
        rows = vec_search(conn, _vec(0, 1, 0, 0), top_k=1)
        assert len(rows) == 1
        assert rows[0]["id"] == id_b
    finally:
        conn.close()


def test_set_valence_override(tmp_path: Path):
    conn = connect(tmp_path / "m.db")
    try:
        init_schema(conn, DIM)
        cid = insert_chunk(conn, "body", _vec(1, 0, 0, 0), "test", valence="unmarked")
        set_valence(conn, cid, "hit", reason="manual test")

        row = conn.execute("SELECT valence FROM chunk WHERE id=?", (cid,)).fetchone()
        assert row["valence"] == "hit"

        override = conn.execute(
            "SELECT valence, reason FROM valence_override WHERE chunk_id=?", (cid,)
        ).fetchone()
        assert override["valence"] == "hit"
        assert override["reason"] == "manual test"
    finally:
        conn.close()
