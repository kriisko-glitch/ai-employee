"""Heuristic valence tagger.

Tags a chunk of text as one of: hit, miss, walkback, unmarked.

- **hit**     — solution worked, approach validated, claim held up
- **miss**    — approach failed, claim turned out wrong
- **walkback** — earlier claim explicitly retracted; calibration moment
- **unmarked** — default; no strong signal

Regex-based. Conservative — when uncertain, returns 'unmarked' so the chunk
surfaces without bias. Users can re-tag explicitly via `aie tag`.
"""
from __future__ import annotations

import re

VALENCE_HIT = "hit"
VALENCE_MISS = "miss"
VALENCE_WALKBACK = "walkback"
VALENCE_UNMARKED = "unmarked"


# Walk-backs — first-person retraction of an earlier claim.
WALKBACK_RE = re.compile(
    r"("
    r"\bi was wrong\b|\bi'?ve been wrong\b|\bi miscall(?:ed)?\b|"
    r"\bi over-?(?:call|claim|read|reach)(?:ed)?\b|"
    r"\bi(?:'?m| am)? walking(?: this)? back\b|"
    r"\bi (?:walk|need to walk)(?: this)? back\b|"
    r"\bwalk(?:ing)? back (?:my|the|a) (?:claim|view|stance|position|reading)\b|"
    r"\bretract(?:ion|ing)?\b|\bcorrected picture\b|\bamend(?:ing|ment)\b|"
    r"\bon (?:re-?read|reflection|second-?look)\b|"
    r"\b(?:turn(?:ed|s)) out (?:to be wrong|i was wrong)\b|"
    r"\bmy reading was off\b"
    r")",
    re.IGNORECASE,
)


# Hits — validated outcomes, things that landed.
HIT_RE = re.compile(
    r"\b("
    r"that (?:was|is) right|landed cleanly|worked as expected|"
    r"approach (?:held|worked|succeeded)|test(?:s)? pass(?:ed|ing)|"
    r"this is what i keep|confirmed correct|"
    r"validated|nailed it|exactly right|"
    r"shipped (?:and|with) (?:passing|green)"
    r")\b",
    re.IGNORECASE,
)


# Misses — things that failed without an attached retraction.
MISS_RE = re.compile(
    r"\b("
    r"that didn'?t work|approach failed|"
    r"test(?:s)? fail(?:ed|ing)|broke (?:in|on) production|"
    r"this was wrong|didn'?t hold up|"
    r"i'?m suspicious of myself|i may be wrong|"
    r"shouldn'?t have|that was a mistake"
    r")\b",
    re.IGNORECASE,
)


def tag_valence(text: str) -> str:
    """Return one of: hit | miss | walkback | unmarked.

    Order of precedence: walkback > miss > hit > unmarked. A walk-back is the
    most informative signal (calibration), then a miss (warning), then a hit
    (confirmed approach).
    """
    if not text:
        return VALENCE_UNMARKED
    if WALKBACK_RE.search(text):
        return VALENCE_WALKBACK
    if MISS_RE.search(text):
        return VALENCE_MISS
    if HIT_RE.search(text):
        return VALENCE_HIT
    return VALENCE_UNMARKED
