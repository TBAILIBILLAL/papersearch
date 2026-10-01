import json

from papersearch.storage import clean, normalise, published_from_id, read_jsonl

from .conftest import PAPERS


def test_published_month_comes_from_the_identifier():
    assert published_from_id("2405.01234") == "2024-05"
    assert published_from_id("0704.0001") == "2007-04"
    assert published_from_id("cs/0501006") == "2005-01"
    assert published_from_id("hep-th/9901001") == "1999-01"
    assert published_from_id("math.GT/0309136") == "2003-09"
    assert published_from_id("not-an-id") == "0000-00"


def test_normalise_cleans_whitespace_and_picks_the_primary_category():
    row = normalise({"id": "2401.00001", "title": "A  title\n on two lines ", "abstract": " Text. ",
                     "authors": ["Ada Chen", "Omar Diallo"], "categories": "cs.LG stat.ML"})
    assert row["title"] == "A title on two lines"
    assert row["abstract"] == "Text."
    assert row["authors"] == "Ada Chen, Omar Diallo"
    assert row["primary_category"] == "cs.LG"
    assert row["published"] == "2024-01"
    assert clean(" a \t b\n") == "a b"


def test_add_count_and_read_back(storage):
    assert storage.count() == 0
    assert storage.add_documents(PAPERS) == len(PAPERS)
    assert storage.count() == len(PAPERS)

    rows = list(storage.iter_documents(batch_size=4))  # more than one batch
    assert [row["arxiv_id"] for row in rows] == [paper["id"] for paper in PAPERS]
    assert rows[0]["primary_category"] == "cs.LG"

    wanted = [rows[1]["id"], rows[3]["id"]]
    fetched = storage.get_documents(wanted)
    assert set(fetched) == set(wanted)
    assert fetched[wanted[0]]["title"] == PAPERS[1]["title"]
    assert storage.get_documents([]) == {}


def test_papers_already_stored_are_skipped(storage):
    storage.add_documents(PAPERS[:3])
    assert storage.add_documents(PAPERS) == len(PAPERS) - 3
    assert storage.add_documents(PAPERS) == 0
    assert storage.count() == len(PAPERS)


def test_records_without_title_or_abstract_are_skipped(storage):
    broken = [{"id": "2401.99999", "title": "", "abstract": "text", "authors": [], "categories": "cs.LG"},
              {"id": "2401.99998", "title": "Title", "abstract": "", "authors": [], "categories": "cs.LG"}]
    assert storage.add_documents(broken) == 0


def test_naive_search_requires_every_word(storage):
    storage.add_documents(PAPERS)
    rows = list(storage.iter_documents())
    traffic = storage.naive_search(["traffic"])
    assert traffic == [rows[0]["id"], rows[3]["id"]]
    assert storage.naive_search(["traffic", "graph"]) == [rows[0]["id"]]
    assert storage.naive_search(["nothingmatches"]) == []
    assert storage.naive_search([]) == []


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "papers.jsonl"
    path.write_text(json.dumps(PAPERS[0]) + "\n\n" + json.dumps(PAPERS[1]) + "\n", encoding="utf-8")
    assert [record["id"] for record in read_jsonl(str(path))] == ["2401.00001", "2401.00002"]
