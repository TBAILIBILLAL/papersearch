"""Spelling correction for query words."""

from __future__ import annotations

from .trie import Trie


def edit_distance(a: str, b: str) -> int:
    """The number of single-character edits needed to turn `a` into `b`.

    An edit is an insertion, a deletion, a replacement, or a swap of two
    neighbouring characters (so "nueral" is one edit away from "neural").
    """
    previous: list[int] | None = None
    current = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        before, previous, current = previous, current, [i]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if before is not None and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                value = min(value, before[j - 2] + 1)
            current.append(value)
    return current[-1]


class SpellChecker:
    """Suggests the most likely intended word for a word that is not in the collection.

    The trie holds the known words, weighted by how many documents contain each.
    """

    def __init__(self, vocabulary: Trie):
        self.vocabulary = vocabulary

    def correct(self, word: str, max_distance: int | None = None) -> str | None:
        """The closest known word, or None if the word is known or nothing is close.

        Short words allow one edit and longer words two, unless max_distance says otherwise.
        """
        if len(word) < 3 or word in self.vocabulary:
            return None
        if max_distance is None:
            max_distance = 1 if len(word) <= 4 else 2
        candidates = self.vocabulary.fuzzy(word, max_distance)
        if not candidates:
            return None
        # fewest edits first; among those, the word seen in the most documents
        best = min(candidates, key=lambda c: (c[1], -c[2], c[0]))
        return best[0]
