import random

import pytest

from papersearch.spelling import SpellChecker, edit_distance
from papersearch.trie import Trie


def make_trie(entries):
    trie = Trie()
    for key, weight in entries.items():
        trie.insert(key, weight)
    return trie


def test_insert_and_lookup():
    trie = make_trie({"graph": 5, "graphs": 2, "grid": 9})
    assert len(trie) == 3
    assert "graph" in trie and "gra" not in trie and "tree" not in trie
    assert trie.weight("grid") == 9
    trie.insert("graph", 7)  # replacing a weight does not add a key
    assert len(trie) == 3 and trie.weight("graph") == 7


def test_invalid_inserts_are_rejected():
    trie = Trie()
    with pytest.raises(ValueError):
        trie.insert("", 1)
    with pytest.raises(ValueError):
        trie.insert("word", 0)


def test_complete_returns_heaviest_keys_first():
    trie = make_trie({"neural": 50, "network": 80, "networks": 30, "new": 10, "graph": 99})
    assert trie.complete("ne", 3) == [("network", 80), ("neural", 50), ("networks", 30)]
    assert trie.complete("net") == [("network", 80), ("networks", 30)]
    assert trie.complete("x") == []
    assert trie.complete("ne", 0) == []


def test_complete_agrees_with_sorting_everything():
    generator = random.Random(1)
    entries = {"".join(generator.choice("abc") for _ in range(generator.randint(1, 6))): generator.randint(1, 1000)
               for _ in range(400)}
    trie = make_trie(entries)
    for prefix in ("", "a", "ab", "abc", "cb"):
        expected = sorted((w for k, w in entries.items() if k.startswith(prefix)), reverse=True)[:5]
        assert [weight for _, weight in trie.complete(prefix, 5)] == expected


def test_edit_distance():
    assert edit_distance("neural", "neural") == 0
    assert edit_distance("nueral", "neural") == 1  # swapped neighbours
    assert edit_distance("netwrk", "network") == 1  # missing letter
    assert edit_distance("graph", "grape") == 1  # wrong letter
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3


def test_fuzzy_finds_keys_within_the_distance():
    trie = make_trie({"neural": 5, "neutral": 3, "natural": 2, "network": 9})
    found = {key: distance for key, distance, _ in trie.fuzzy("nueral", 2)}
    assert found == {"neural": 1, "neutral": 2}


def test_fuzzy_agrees_with_edit_distance_on_random_words():
    generator = random.Random(2)
    keys = {"".join(generator.choice("abcd") for _ in range(generator.randint(2, 7))) for _ in range(300)}
    trie = make_trie({key: 1 for key in keys})
    for _ in range(40):
        word = "".join(generator.choice("abcd") for _ in range(generator.randint(2, 7)))
        expected = {key for key in keys if edit_distance(word, key) <= 2}
        assert {key for key, _, _ in trie.fuzzy(word, 2)} == expected


def test_spell_checker_prefers_the_closest_then_the_most_common_word():
    checker = SpellChecker(make_trie({"learning": 900, "leaning": 4, "network": 500, "networks": 300, "graph": 50}))
    assert checker.correct("lerning") == "learning"
    assert checker.correct("netwrok") == "network"
    assert checker.correct("network") is None  # already a known word
    assert checker.correct("zzzzzz") is None  # nothing is close
    assert checker.correct("gr") is None  # too short to guess
