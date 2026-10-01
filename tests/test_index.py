import math

import pytest

from papersearch.index import InvertedIndex


def build(*documents, **options):
    index = InvertedIndex(**options)
    for title, abstract in documents:
        index.add(title, abstract)
    index.finish()
    return index


def test_postings_list_the_documents_that_contain_a_term():
    index = build(("", "graph search"), ("", "tree search"), ("", "graph graph theory"))
    documents, frequencies = index.postings["graph"]
    assert list(documents) == [0, 2]
    assert list(frequencies) == [1, 2]
    assert index.df("search") == 2
    assert index.df("missing") == 0


def test_title_words_count_more_than_abstract_words():
    index = build(("graph", "graph theory"), title_boost=3)
    assert list(index.postings["graph"][1]) == [4]  # 1 in the abstract + 3 for the title
    assert index.doc_len[0] == 5


def test_rare_terms_have_higher_idf():
    index = build(("", "common rare"), ("", "common"), ("", "common"))
    assert index.idf("rare") > index.idf("common")
    assert index.idf("common") == pytest.approx(math.log(1 + 0.5 / 3.5))


def test_bm25_matches_the_formula():
    index = build(("", "alpha beta beta"), ("", "alpha gamma delta epsilon zeta"))
    scores = index.score(["beta"])
    tf, length, average = 2, 3, 4.0
    expected = index.idf("beta") * tf * (index.k1 + 1) / (tf + index.k1 * (1 - index.b + index.b * length / average))
    assert scores == {0: pytest.approx(expected, rel=1e-5)}


def test_document_matching_more_query_terms_ranks_first():
    index = build(("", "graph neural network"), ("", "graph database"), ("", "neural network training"))
    scores = index.score(["graph", "neural", "network"])
    assert max(scores, key=scores.get) == 0


def test_shorter_document_ranks_above_longer_one_with_the_same_count():
    index = build(("", "search engine"), ("", "search " + "filler " * 40))
    scores = index.score(["search"])
    assert scores[0] > scores[1]


def test_repeated_query_terms_are_counted_once():
    index = build(("", "graph theory"))
    assert index.score(["graph", "graph"]) == index.score(["graph"])


def test_baseline_modes():
    index = build(("", "graph graph tree"), ("", "graph"))
    assert index.score(["graph", "tree"], "tf") == {0: 3.0, 1: 1.0}
    assert index.score(["graph", "tree"], "match") == {0: 2.0, 1: 1.0}
    with pytest.raises(ValueError):
        index.score(["graph"], "unknown")


def test_state_round_trip_gives_the_same_scores():
    index = build(("graph search", "graph neural network"), ("tree", "tree search and graph"))
    restored = InvertedIndex.from_state(index.to_state())
    assert restored.num_docs == 2
    assert restored.score(["graph", "search"]) == index.score(["graph", "search"])
