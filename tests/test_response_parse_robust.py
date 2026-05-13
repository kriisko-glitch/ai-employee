"""Resilience tests for the response parser — handles the common
model-output failure modes (forgotten closing tags, mixed casing, etc).
"""
from ai_employee.response_parse import parse_response


def test_unterminated_post_block_stops_at_state_marker():
    """Model forgot [/POST]. Parser should stop at the [STATE...] marker
    rather than pulling everything to EOF."""
    text = """\
[THINKING]
weighing options
[/THINKING]

[POST]
short clean reply

[REFLECTING 2]
[next_tick_seconds: 90]
"""
    p = parse_response(text)
    assert p.thinking == "weighing options"
    assert p.post is not None
    assert "short clean reply" in p.post
    # The state marker must NOT have leaked into the post.
    assert "[REFLECTING" not in p.post
    assert "[next_tick_seconds" not in p.post


def test_unterminated_post_stops_at_next_tick_marker():
    text = """\
[POST]
hello world
[next_tick_seconds: 120]
"""
    p = parse_response(text)
    assert "hello world" in p.post
    assert "[next_tick_seconds" not in p.post


def test_unterminated_post_to_eof_when_no_markers_follow():
    """If nothing follows [POST] except plain text, take it all (until EOF)."""
    text = """\
[POST]
this is the whole post
no closing tag and no markers after
"""
    p = parse_response(text)
    assert "this is the whole post" in p.post
    assert "no closing tag" in p.post


def test_thinking_does_not_leak_into_unterminated_post():
    """The bug we fixed: when POST is unterminated, the parser must NOT also
    pull in the THINKING block that came before."""
    text = """\
[THINKING]
secret reasoning that should not be visible
[/THINKING]

[POST]
the actual reply

[REFLECTING 1]
[next_tick_seconds: 60]
"""
    p = parse_response(text)
    assert "secret reasoning" not in p.post
    assert "the actual reply" in p.post


def test_closed_post_still_preferred_over_open():
    """When both close-tag and downstream markers exist, the closed match wins."""
    text = """\
[POST]
inner content
[/POST]

some loose text after

[REFLECTING 1]
"""
    p = parse_response(text)
    assert p.post == "inner content"
    assert "loose text" not in p.post
