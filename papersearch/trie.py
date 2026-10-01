"""A weighted trie (prefix tree).

It answers two questions quickly:

* complete(prefix): which stored keys start with this prefix, best first?
* fuzzy(word): which stored keys are within a few typing mistakes of this word?

Each node remembers the highest weight found anywhere below it. That lets
complete() walk towards the best keys first and stop early, instead of
collecting every key under the prefix and sorting them.
"""

from __future__ import annotations

import heapq
from itertools import count


class _Node:
    __slots__ = ("children", "weight", "best")

    def __init__(self) -> None:
        self.children: dict[str, _Node] | None = None  # created on first use; leaves never need one
        self.weight = 0  # weight of the key that ends here; 0 means no key ends here
        self.best = 0  # highest weight in this subtree


class Trie:
    def __init__(self) -> None:
        self.root = _Node()
        self.size = 0

    def insert(self, key: str, weight: int) -> None:
        """Store a key with a positive weight. Inserting it again replaces the weight."""
        if not key or weight <= 0:
            raise ValueError("key must be non-empty and weight positive")
        node = self.root
        path = [node]
        for ch in key:
            if node.children is None:
                node.children = {}
            child = node.children.get(ch)
            if child is None:
                child = node.children[ch] = _Node()
            node = child
            path.append(node)
        if node.weight == 0:
            self.size += 1
        node.weight = weight
        for visited in path:
            if weight > visited.best:
                visited.best = weight

    def weight(self, key: str) -> int:
        """The weight of a key, or 0 if it is not stored."""
        node = self._find(key)
        return node.weight if node else 0

    def __contains__(self, key: str) -> bool:
        return self.weight(key) > 0

    def __len__(self) -> int:
        return self.size

    def _find(self, prefix: str) -> _Node | None:
        node = self.root
        for ch in prefix:
            node = node.children.get(ch) if node.children else None
            if node is None:
                return None
        return node

    # ------------------------------------------------------------------ autocomplete

    def complete(self, prefix: str, limit: int = 8) -> list[tuple[str, int]]:
        """The `limit` heaviest keys that start with `prefix`, heaviest first."""
        start = self._find(prefix)
        if start is None or limit <= 0:
            return []

        results: list[tuple[str, int]] = []
        tie = count()  # keeps heap entries comparable when weights are equal
        # entries: (-weight, tie, text, node, is_key). A subtree entry is ordered by its best weight.
        heap: list = [(-start.best, next(tie), prefix, start, False)]
        while heap and len(results) < limit:
            negative, _, text, node, is_key = heapq.heappop(heap)
            if is_key:
                results.append((text, -negative))
                continue
            if node.weight:
                heapq.heappush(heap, (-node.weight, next(tie), text, node, True))
            if node.children:
                for ch, child in node.children.items():
                    heapq.heappush(heap, (-child.best, next(tie), text + ch, child, False))
        return results

    # ------------------------------------------------------------------ fuzzy search

    def fuzzy(self, word: str, max_distance: int) -> list[tuple[str, int, int]]:
        """All keys within `max_distance` edits of `word`, as (key, distance, weight).

        One edit is inserting, deleting or replacing a character, or swapping two
        neighbouring characters. The trie is walked once while one row of the
        edit-distance table is carried along per node; a branch is abandoned as
        soon as no key below it can still be close enough.
        """
        results: list[tuple[str, int, int]] = []
        first_row = list(range(len(word) + 1))
        for ch, child in (self.root.children or {}).items():
            self._fuzzy(child, ch, "", word, first_row, None, ch, max_distance, results)
        return results

    def _fuzzy(self, node, ch, previous_ch, word, previous_row, row_before, text, max_distance, results) -> None:
        row = [previous_row[0] + 1]
        for i in range(1, len(word) + 1):
            cost = 0 if word[i - 1] == ch else 1
            value = min(row[i - 1] + 1, previous_row[i] + 1, previous_row[i - 1] + cost)
            if row_before is not None and i > 1 and ch == word[i - 2] and previous_ch == word[i - 1]:
                value = min(value, row_before[i - 2] + 1)  # two neighbouring characters swapped
            row.append(value)

        if node.weight and row[-1] <= max_distance:
            results.append((text, row[-1], node.weight))

        # a swap reaches back two rows, so a branch is only dead when both rows are too far
        if node.children and (min(row) <= max_distance or min(previous_row) < max_distance):
            for next_ch, child in node.children.items():
                self._fuzzy(child, next_ch, ch, word, row, previous_row, text + next_ch, max_distance, results)
