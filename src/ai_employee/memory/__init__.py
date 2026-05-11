"""Long-term memory — sqlite + sqlite-vec, valence-tagged, decay-weighted."""

from .valence import tag_valence, VALENCE_HIT, VALENCE_MISS, VALENCE_WALKBACK, VALENCE_UNMARKED

__all__ = [
    "tag_valence",
    "VALENCE_HIT",
    "VALENCE_MISS",
    "VALENCE_WALKBACK",
    "VALENCE_UNMARKED",
]
