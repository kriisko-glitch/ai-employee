"""Valence tagger — pattern matching."""
import pytest

from ai_employee.memory.valence import (
    tag_valence,
    VALENCE_HIT, VALENCE_MISS, VALENCE_WALKBACK, VALENCE_UNMARKED,
)


@pytest.mark.parametrize("text", [
    "I was wrong about that.",
    "I'm walking back my earlier claim.",
    "On reflection, my reading was off.",
    "Postscript: it turned out to be wrong.",
    "Retraction: my approach didn't hold up.",
])
def test_walkback(text):
    assert tag_valence(text) == VALENCE_WALKBACK


@pytest.mark.parametrize("text", [
    "The tests failed in CI.",
    "That approach failed.",
    "I'm suspicious of myself here.",
    "I shouldn't have committed that.",
])
def test_miss(text):
    assert tag_valence(text) == VALENCE_MISS


@pytest.mark.parametrize("text", [
    "That was right — landed cleanly.",
    "The approach worked as expected.",
    "Tests passed.",
    "Confirmed correct.",
])
def test_hit(text):
    assert tag_valence(text) == VALENCE_HIT


def test_unmarked():
    assert tag_valence("Just a neutral observation about the weather.") == VALENCE_UNMARKED
    assert tag_valence("") == VALENCE_UNMARKED


def test_walkback_takes_precedence_over_miss():
    # Contains both miss-language and walkback-language; walkback wins.
    assert tag_valence("My approach failed — I was wrong.") == VALENCE_WALKBACK
