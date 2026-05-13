"""Structured response parser — THINKING / POST / STATE / next_tick separation."""
import pytest

from ai_employee.response_parse import parse_response


def test_full_structured_response():
    text = """\
[THINKING]
kris just messaged. i should answer briefly.
[/THINKING]

[POST]
yes — what's up?
[/POST]

[REFLECTING 2]
[next_tick_seconds: 60]
"""
    p = parse_response(text)
    assert p.thinking == "kris just messaged. i should answer briefly."
    assert p.post == "yes — what's up?"
    assert p.state == "REFLECTING"
    assert p.intensity == 2
    assert p.next_tick_seconds == 60
    assert p.is_silent is False
    assert p.has_structured_format is True


def test_silent_post():
    text = """\
[THINKING]
nothing to say this tick.
[/THINKING]

[POST]
SILENT
[/POST]

[BOREDOM 1]
[next_tick_seconds: 600]
"""
    p = parse_response(text)
    assert p.is_silent is True
    assert p.post is None
    # Thinking is still retained — model still sees its own reasoning.
    assert "nothing to say" in p.thinking


def test_silent_case_insensitive():
    p = parse_response("[POST]silent[/POST]")
    assert p.is_silent is True
    p = parse_response("[POST]  Silent.  [/POST]")
    assert p.is_silent is True


def test_no_thinking_block_falls_back():
    """Legacy or malformed responses with no THINKING block should still
    produce a postable result rather than blowing up."""
    text = "Just a plain response with no markers."
    p = parse_response(text)
    assert p.thinking is None
    assert p.post == "Just a plain response with no markers."
    assert p.is_silent is False


def test_no_post_block_strips_thinking_for_legacy_fallback():
    """If THINKING is present but POST is not, the fallback strips THINKING
    so we don't post the inner monologue by accident."""
    text = """\
[THINKING]
my secret reasoning
[/THINKING]

Some loose body text.

[REFLECTING 1]
[next_tick_seconds: 120]
"""
    p = parse_response(text)
    assert p.thinking == "my secret reasoning"
    assert "secret reasoning" not in p.post
    assert "Some loose body text" in p.post


def test_state_and_next_tick_extracted_independently():
    text = "[POST]hello[/POST]\n[BUILDING 3]\n[next_tick_seconds: 90]"
    p = parse_response(text)
    assert p.state == "BUILDING"
    assert p.intensity == 3
    assert p.next_tick_seconds == 90


def test_missing_state_returns_none():
    text = "[POST]just a post[/POST]"
    p = parse_response(text)
    assert p.state is None
    assert p.intensity is None
    assert p.next_tick_seconds is None


def test_raw_preserved_for_history():
    """The full response — INCLUDING thinking — must be preserved so the
    model sees its own reasoning continuity in conversation history."""
    text = """\
[THINKING]
this is my reasoning
[/THINKING]

[POST]
short reply
[/POST]
"""
    p = parse_response(text)
    assert p.raw == text
    assert "this is my reasoning" in p.raw  # thinking lives on in raw → history


def test_multiline_post_preserved():
    text = """\
[POST]
line one
line two

line four after blank
[/POST]
"""
    p = parse_response(text)
    assert "line one" in p.post
    assert "line four" in p.post


def test_empty_response():
    p = parse_response("")
    assert p.post == ""
    assert p.thinking is None
    assert p.is_silent is False
