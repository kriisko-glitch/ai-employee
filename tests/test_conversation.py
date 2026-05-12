"""Working-memory JSONL — append, load, clear, conversion."""
from pathlib import Path

from ai_employee.conversation import (
    Turn, append_turn, load_all, load_recent, size_bytes, clear, to_message_list,
)


def test_turn_roundtrip():
    t = Turn(role="user", content="hello world", ts="2026-05-12T12:00:00+00:00")
    line = t.to_jsonl()
    parsed = Turn.from_jsonl(line)
    assert parsed.role == "user"
    assert parsed.content == "hello world"


def test_append_and_load(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    append_turn(p, "user", "first")
    append_turn(p, "assistant", "reply 1")
    append_turn(p, "user", "second")

    turns = load_all(p)
    assert len(turns) == 3
    assert turns[0].content == "first"
    assert turns[-1].content == "second"


def test_load_recent_returns_tail(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    for i in range(10):
        append_turn(p, "user" if i % 2 == 0 else "assistant", f"msg {i}")
    recent = load_recent(p, 4)
    assert len(recent) == 4
    assert recent[0].content == "msg 6"
    assert recent[-1].content == "msg 9"


def test_clear_truncates(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    append_turn(p, "user", "x" * 1000)
    assert size_bytes(p) > 100
    clear(p)
    assert size_bytes(p) == 0
    assert load_all(p) == []


def test_to_message_list_shape():
    turns = [
        Turn(role="user", content="a", ts="t1"),
        Turn(role="assistant", content="b", ts="t2"),
    ]
    msgs = to_message_list(turns)
    assert msgs == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_corrupt_lines_skipped(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        '{"role":"user","content":"good","ts":"t"}\n'
        'not json at all\n'
        '{"role":"assistant","content":"also good","ts":"t"}\n',
        encoding="utf-8",
    )
    turns = load_all(p)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"
