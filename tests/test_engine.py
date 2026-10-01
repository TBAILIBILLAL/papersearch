from papersearch.engine import SearchEngine, highlight, title_phrases

from .conftest import PAPERS


def titles(response):
    return ["".join(piece["text"] for piece in result["title"]) for result in response["results"]]


def marked(pieces):
    return [piece["text"] for piece in pieces if piece["hit"]]


def test_search_ranks_the_best_match_first(engine):
    response = engine.search("graph neural networks traffic")
    assert titles(response)[0] == "Graph Neural Networks for Traffic Forecasting"
    assert response["total"] == 3  # every paper that contains at least one of the words
    assert response["corrected_query"] is None
    assert response["took_ms"] >= 0


def test_singular_query_finds_plural_text(engine):
    assert titles(engine.search("transformer"))[0] == "Efficient Transformers for Long Documents"
    assert titles(engine.search("database query"))[0] == "Query Optimization in Distributed Databases"


def test_matches_are_marked_in_title_and_snippet(engine):
    result = engine.search("traffic signals")["results"][0]
    assert marked(result["title"]) == ["Traffic", "Signal"]
    assert "traffic" in [text.lower() for text in marked(result["snippet"])]
    assert result["url"] == "https://arxiv.org/abs/" + result["arxiv_id"]
    assert result["published"] == "2024-03"


def test_misspelled_words_are_corrected(engine):
    response = engine.search("reinforcment lerning")
    assert response["corrected_query"] == "reinforcement learning"
    assert response["total"] == 2
    assert "Reinforcement Learning" in titles(response)[0]


def test_correction_keeps_the_other_words_of_the_query(engine):
    assert engine.search("the trafic of networks")["corrected_query"] == "the traffic of networks"


def test_correction_can_be_switched_off(engine):
    response = engine.search("reinforcment lerning", correct=False)
    assert response["corrected_query"] is None
    assert response["total"] == 0


def test_rare_typo_that_exists_in_a_paper_is_still_corrected(storage, monkeypatch):
    # real collections contain misspellings, so a typo can match a paper or two
    typo_paper = {"id": "2404.00007", "title": "Notes on Road Trafic", "abstract": "A short note.",
                  "authors": ["Ana Ruiz"], "categories": "cs.CY"}
    storage.add_documents(PAPERS + [typo_paper])
    engine = SearchEngine.build(storage)
    monkeypatch.setattr("papersearch.engine.RARE_WORD_FACTOR", 2)

    assert engine.search("trafic")["corrected_query"] == "traffic"
    assert engine.search("trafic", correct=False)["total"] == 1
    # a rare word with no much more common neighbour is left alone
    assert engine.search("ruiz")["corrected_query"] is None


def test_category_filter(engine):
    everything = engine.search("learning")
    only_robotics = engine.search("learning", category="cs.RO")
    assert everything["total"] == 2
    assert only_robotics["total"] == 1
    assert only_robotics["results"][0]["category"] == "cs.RO"
    assert engine.search("learning", category="cs.XX")["total"] == 0


def test_paging_does_not_repeat_results(engine):
    first = engine.search("networks learning traffic", limit=2, offset=0)
    second = engine.search("networks learning traffic", limit=2, offset=2)
    assert first["total"] == second["total"] == 4
    assert len(first["results"]) == 2 and len(second["results"]) == 2
    assert not set(titles(first)) & set(titles(second))


def test_queries_without_content_return_nothing(engine):
    assert engine.search("the of and")["total"] == 0
    assert engine.search("!!!")["results"] == []
    assert engine.search("zzzzzzzz")["total"] == 0


def test_suggest_completes_words_and_phrases(engine):
    assert "traffic" in engine.suggest("traf")
    assert "reinforcement learning" in engine.suggest("reinf")
    assert engine.suggest("neural net")[0] == "neural networks"
    assert "neural networks" in engine.suggest("neural ")  # a finished word is completed with phrases
    assert engine.suggest("") == []
    assert engine.suggest("qqq") == []


def test_title_phrases_start_and_end_with_content_words():
    assert title_phrases(["internet", "of", "things"]) == ["internet of things"]
    assert title_phrases(["a", "survey", "of", "learning"]) == ["survey of learning"]
    assert "neural networks" in title_phrases(["graph", "neural", "networks"])
    assert title_phrases(["learning", "via", "novel", "graphs"]) == []  # weak first or last words


def test_highlight_picks_the_window_with_the_most_matches():
    text = "alpha " * 30 + "graph neural network " + "omega " * 30
    pieces = highlight(text, {"graph", "neural", "network"}, max_words=10)
    assert marked(pieces) == ["graph", "neural", "network"]
    assert pieces[0]["text"] == "... " and pieces[-1]["text"] == " ..."
    assert highlight("", {"graph"}) == []
    assert highlight("No match here.", {"graph"}) == [{"text": "No match here.", "hit": False}]


def test_category_counts_and_stats(engine):
    counts = dict(engine.category_counts())
    assert counts["cs.LG"] == 1 and counts["cs.AI"] == 1 and len(counts) == 6
    stats = engine.stats()
    assert stats["documents"] == 6
    assert stats["postings"] >= stats["terms"] > 0


def test_saved_index_gives_the_same_results(engine, storage, tmp_path):
    path = str(tmp_path / "index.pkl")
    engine.save(path)
    restored = SearchEngine.load(path, storage)
    for query in ("graph neural networks", "reinforcment lerning", "databases"):
        assert titles(restored.search(query)) == titles(engine.search(query))
    assert restored.suggest("traf") == engine.suggest("traf")
    assert restored.stats()["postings"] == engine.stats()["postings"]
