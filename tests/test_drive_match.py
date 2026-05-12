"""Drive-matched retrieval — current state biases recall toward matching SDT axis."""
import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("sqlite_vec") is None:
    pytest.skip("sqlite-vec not installed", allow_module_level=True)

import sqlite3
import sqlite_vec

from ai_employee.config import MemoryConfig
from ai_employee.memory.db import connect, init_schema, insert_chunk
from ai_employee.memory.retrieve import _drive_match


DIM = 4


def _vec(*v):
    assert len(v) == DIM
    return list(v)


def _make_chunk_row(autonomy=0.0, competence=0.0, relatedness=0.0):
    """Build a sqlite3.Row-like dict for drive_match testing."""
    return {
        "autonomy": autonomy,
        "competence": competence,
        "relatedness": relatedness,
        "keys": lambda: ["autonomy", "competence", "relatedness"],
    }


class FakeRow:
    """Behaves like sqlite3.Row enough for _drive_match (has `.keys()` + __getitem__)."""
    def __init__(self, **kw):
        self.data = kw
    def __getitem__(self, k):
        return self.data[k]
    def keys(self):
        return list(self.data.keys())


def test_drive_match_disabled_returns_one():
    cfg = MemoryConfig()
    cfg.parameterization.drive_bias.enabled = False
    row = FakeRow(autonomy=1.0, competence=1.0, relatedness=1.0)
    assert _drive_match(row, "BUILDING", cfg) == 1.0


def test_drive_match_building_boosts_competence():
    cfg = MemoryConfig()
    row = FakeRow(autonomy=0.0, competence=1.0, relatedness=0.0)
    # bias_strength=0.5, competence=1.0 → multiplier = 1.0 + 0.5*1.0 = 1.5
    assert _drive_match(row, "BUILDING", cfg) == pytest.approx(1.5)


def test_drive_match_boredom_boosts_autonomy():
    cfg = MemoryConfig()
    row = FakeRow(autonomy=0.8, competence=0.2, relatedness=0.1)
    # bias_strength=0.5, autonomy=0.8 → 1.0 + 0.5*0.8 = 1.4
    assert _drive_match(row, "BOREDOM", cfg) == pytest.approx(1.4)


def test_drive_match_waking_boosts_relatedness():
    cfg = MemoryConfig()
    row = FakeRow(autonomy=0.1, competence=0.2, relatedness=0.9)
    # bias_strength=0.5, relatedness=0.9 → 1.0 + 0.5*0.9 = 1.45
    assert _drive_match(row, "WAKING", cfg) == pytest.approx(1.45)


def test_drive_match_unknown_state_returns_one():
    cfg = MemoryConfig()
    row = FakeRow(autonomy=1.0, competence=1.0, relatedness=1.0)
    assert _drive_match(row, "WEIRD_STATE", cfg) == 1.0


def test_drive_match_case_insensitive():
    cfg = MemoryConfig()
    row = FakeRow(autonomy=0.0, competence=1.0, relatedness=0.0)
    assert _drive_match(row, "building", cfg) == pytest.approx(1.5)
    assert _drive_match(row, "Building", cfg) == pytest.approx(1.5)


def test_schema_migration_adds_sdt_columns(tmp_path: Path):
    """Existing v0.1 chunk tables should be migrated to add SDT columns."""
    db_file = tmp_path / "old.db"
    # Build a v0.1-style table manually.
    raw = sqlite3.connect(str(db_file))
    raw.enable_load_extension(True); sqlite_vec.load(raw); raw.enable_load_extension(False)
    raw.execute("""
        CREATE TABLE chunk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, ingested_ts TEXT NOT NULL,
            source TEXT, body TEXT NOT NULL,
            valence TEXT NOT NULL DEFAULT 'unmarked',
            weight REAL NOT NULL DEFAULT 0.5,
            embedding_model TEXT NOT NULL,
            last_recalled_ts TEXT, recall_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        )
    """)
    raw.execute("""
        INSERT INTO chunk (ts, ingested_ts, body, embedding_model)
        VALUES ('2026-01-01', '2026-01-01', 'old chunk', 'BAAI/bge-small-en-v1.5')
    """)
    raw.commit()
    raw.close()

    # Now open with the new init_schema — should add the SDT cols.
    conn = connect(db_file)
    init_schema(conn, DIM)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunk)").fetchall()}
    assert "autonomy" in cols
    assert "competence" in cols
    assert "relatedness" in cols
    # Old row should default to 0.0 on the new columns.
    row = conn.execute("SELECT autonomy, competence, relatedness FROM chunk").fetchone()
    assert row["autonomy"] == 0.0
    conn.close()


def test_insert_chunk_with_sdt(tmp_path: Path):
    db_file = tmp_path / "m.db"
    conn = connect(db_file)
    init_schema(conn, DIM)
    cid = insert_chunk(
        conn, body="growth moment", embedding=_vec(0.1, 0.2, 0.3, 0.4),
        embedding_model="test", valence="hit",
        autonomy=0.7, competence=0.9, relatedness=0.4,
    )
    row = conn.execute(
        "SELECT autonomy, competence, relatedness FROM chunk WHERE id=?", (cid,)
    ).fetchone()
    assert row["autonomy"] == pytest.approx(0.7)
    assert row["competence"] == pytest.approx(0.9)
    assert row["relatedness"] == pytest.approx(0.4)
    conn.close()
