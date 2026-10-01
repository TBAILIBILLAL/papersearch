"""Turning text into index terms.

Text goes through three steps: split into lowercase words, drop stopwords,
and reduce plurals to their singular form so that "network" and "networks"
are the same term.
"""

from __future__ import annotations

import re

WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be been between both but by can could
    do does each for from had has have how if in into is it its may more most no not of
    on one only or other our out over so some such than that the their them then there
    these they this those through to under up use used using very was we were what when
    where which while who will with would
    """.split()
)

# Words that end in "s" but are not plurals, and plurals the rules below get wrong.
_UNCHANGED = frozenset({"bias", "alias", "lens", "canvas", "atlas", "chaos", "always", "perhaps", "whereas"})
_IRREGULAR = {
    "biases": "bias",
    "aliases": "alias",
    "lenses": "lens",
    "analyses": "analysis",
    "hypotheses": "hypothesis",
    "indices": "index",
    "matrices": "matrix",
    "vertices": "vertex",
    "movies": "movie",
    "cookies": "cookie",
}

_stem_cache: dict[str, str] = {}


def words(text: str) -> list[str]:
    """All lowercase words in the text, stopwords included."""
    return WORD.findall(text.lower())


def stem(word: str) -> str:
    """Reduce a plural to its singular form with a few suffix rules."""
    cached = _stem_cache.get(word)
    if cached is not None:
        return cached

    if len(word) <= 3 or word in _UNCHANGED:
        result = word
    elif word in _IRREGULAR:
        result = _IRREGULAR[word]
    elif word.endswith("ies") and len(word) > 4:
        result = word[:-3] + "y"  # queries -> query
    elif word.endswith(("sses", "xes", "ches", "shes", "zes")):
        result = word[:-2]  # classes -> class, approaches -> approach
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        result = word[:-1]  # networks -> network
    else:
        result = word

    _stem_cache[word] = result
    return result


def is_content_word(word: str) -> bool:
    """True for words worth indexing: longer than one character and not a stopword."""
    return len(word) > 1 and word not in STOPWORDS


def terms(text: str) -> list[str]:
    """The index terms of a text, in order, with repeats."""
    return [stem(w) for w in WORD.findall(text.lower()) if len(w) > 1 and w not in STOPWORDS]
