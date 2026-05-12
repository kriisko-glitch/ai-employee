"""DREAMING ritual — prompt format, parse, ingest, post-strip."""
import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("sqlite_vec") is None:
    pytest.skip("sqlite-vec not installed", allow_module_level=True)

from ai_employee.config import load_agent_config
from ai_employee.memory.sleep import (
    build_sleep_prompt, parse_dream, post_strip_dream,
)


def _sample_dream_response() -> str:
    return """\
💤 [DREAMING 0] going down to consolidate.

session was about wiring up first deepseek tick and finding rough edges.

what i'm carrying forward:
- deepseek usage fields parse cleanly
- vec_search needs LIMIT inside vec0 scan, not after JOIN
- cp1252 console crashes on unicode dashes — utf-8 reconfigure fixes it

back in a moment.

---MOMENTS---
[
  {"body": "vec_search fix landed — CTE pattern works", "valence": "hit",
   "autonomy": 0.7, "competence": 0.9, "relatedness": 0.3},
  {"body": "first attempt at aie init left name as 'example'", "valence": "miss",
   "autonomy": 0.5, "competence": 0.3, "relatedness": 0.2},
  {"body": "i'd assumed the bare pip install path worked — wrong",
   "valence": "walkback", "autonomy": 0.4, "competence": 0.5, "relatedness": 0.6}
]
---END-MOMENTS---

---MEMORY-CARRY-FORWARD---
- deepseek usage fields parse cleanly
- vec_search needs LIMIT inside vec0 scan
- cp1252 console requires utf-8 reconfigure on Windows
---END-MEMORY-CARRY-FORWARD---

---STATE-UPDATE---
state: DREAMING
intensity: 0
next_tick_delay_seconds: 60
observation: dreaming — distilled session
"""


def test_build_sleep_prompt_includes_thresholds():
    p = build_sleep_prompt(1_500_000, 1_000_000)
    assert "1,500,000" in p
    assert "1,000,000" in p
    assert "MOMENTS" in p


def test_parse_dream_extracts_moments():
    d = parse_dream(_sample_dream_response())
    assert len(d.moments) == 3
    assert d.moments[0]["valence"] == "hit"
    assert d.moments[0]["competence"] == 0.9
    assert d.moments[1]["valence"] == "miss"
    assert d.moments[2]["valence"] == "walkback"


def test_parse_dream_extracts_memory_carry():
    d = parse_dream(_sample_dream_response())
    assert "deepseek usage fields" in d.memory_carry
    assert "vec_search" in d.memory_carry


def test_parse_dream_clips_sdt_scores_to_unit_range():
    bad = """\
junk
---MOMENTS---
[{"body":"x","valence":"hit","autonomy":1.7,"competence":-0.3,"relatedness":2.5}]
---END-MOMENTS---
"""
    d = parse_dream(bad)
    m = d.moments[0]
    assert 0.0 <= m["autonomy"] <= 1.0
    assert 0.0 <= m["competence"] <= 1.0
    assert 0.0 <= m["relatedness"] <= 1.0


def test_parse_dream_handles_garbage():
    d = parse_dream("no structured blocks at all here")
    assert d.moments == []
    assert d.memory_carry == ""


def test_post_strip_dream_removes_blocks():
    stripped = post_strip_dream(_sample_dream_response())
    assert "---MOMENTS---" not in stripped
    assert "---MEMORY-CARRY-FORWARD---" not in stripped
    assert "---STATE-UPDATE---" not in stripped
    # The narrative body survives.
    assert "going down to consolidate" in stripped


def _build_agent_with_memory(tmp_path: Path):
    repo = tmp_path
    (repo / "pyproject.toml").touch()
    agent_dir = repo / "agents" / "dreamer"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("test")
    (agent_dir / "MEMORY.md").write_text("# MEMORY\n")
    (agent_dir / "agent.yaml").write_text("""\
name: dreamer
workspace: agents/dreamer
model:
  provider: openai
  model_id: test
  api_key_env: FAKE
heartbeat:
  enabled: false
memory:
  enabled: true
  storage:
    backend: sqlite_vec
    embedding:
      model: BAAI/bge-small-en-v1.5
      dim: 4
""")
    return load_agent_config(agent_dir / "agent.yaml", repo_root=repo)


def test_ingest_dream_writes_chunks_and_updates_memory_md(tmp_path: Path,
                                                          monkeypatch):
    cfg = _build_agent_with_memory(tmp_path)

    # Stub Embedder to skip the real model load.
    from ai_employee.memory import sleep as sleep_mod
    class FakeEmbedder:
        def __init__(self, *a, **kw): pass
        def embed_batch(self, texts): return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    monkeypatch.setattr(sleep_mod, "Embedder", FakeEmbedder)

    d = parse_dream(_sample_dream_response())
    moments, ids = sleep_mod.ingest_dream(cfg, d, history_bytes=1_100_000)

    assert moments == 3
    assert len(ids) == 3

    # MEMORY.md should now contain the carry-forward bullets.
    body = cfg.memory_file.read_text()
    assert "vec_search needs LIMIT" in body
    assert "## Sleep" in body

    # Verify the chunks have SDT scores set.
    import sqlite3, sqlite_vec
    conn = sqlite3.connect(cfg.memory_db_file)
    conn.enable_load_extension(True); sqlite_vec.load(conn); conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT body, valence, autonomy, competence, relatedness FROM chunk"
    ).fetchall()
    conn.close()
    assert len(rows) == 3
    by_valence = {r["valence"]: r for r in rows}
    assert by_valence["hit"]["competence"] == pytest.approx(0.9)
    assert by_valence["miss"]["competence"] == pytest.approx(0.3)

    # Sleep event recorded.
    conn = sqlite3.connect(cfg.memory_db_file)
    cnt = conn.execute("SELECT COUNT(*) FROM sleep_event").fetchone()[0]
    conn.close()
    assert cnt == 1
