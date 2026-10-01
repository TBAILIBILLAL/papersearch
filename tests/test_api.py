import pytest
from fastapi.testclient import TestClient

from papersearch.api import create_app
from papersearch.bootstrap import prepare
from papersearch.config import Settings

from .conftest import PAPERS


@pytest.fixture()
def client(engine):
    with TestClient(create_app(engine)) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "documents": len(PAPERS)}


def test_search_endpoint(client):
    body = client.get("/api/search", params={"q": "graph neural networks"}).json()
    assert body["total"] == 2
    assert body["results"][0]["arxiv_id"] in {"2401.00001", "cs/0501006"}
    assert {"title", "snippet", "authors", "category", "published", "score", "url"} <= set(body["results"][0])


def test_search_with_category_and_paging(client):
    body = client.get("/api/search", params={"q": "learning", "category": "cs.RO", "limit": 1}).json()
    assert body["total"] == 1 and len(body["results"]) == 1
    assert client.get("/api/search", params={"q": "learning", "offset": 5}).json()["results"] == []


def test_spelling_correction_through_the_api(client):
    corrected = client.get("/api/search", params={"q": "trafic"}).json()
    assert corrected["corrected_query"] == "traffic" and corrected["total"] == 2
    exact = client.get("/api/search", params={"q": "trafic", "correct": "false"}).json()
    assert exact["corrected_query"] is None and exact["total"] == 0


def test_invalid_requests_are_rejected(client):
    assert client.get("/api/search").status_code == 422  # the query is required
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "limit": 500}).status_code == 422
    assert client.get("/api/search", params={"q": "x" * 300}).status_code == 422


def test_suggest_categories_and_stats(client):
    assert "traffic" in client.get("/api/suggest", params={"q": "traf"}).json()["suggestions"]
    categories = client.get("/api/categories").json()["categories"]
    assert {"name": "cs.DB", "papers": 1} in categories
    assert client.get("/api/stats").json()["documents"] == len(PAPERS)


def test_search_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "PaperSearch" in response.text


def test_prepare_loads_the_sample_then_reuses_the_saved_index(tmp_path):
    sample = tmp_path / "sample.jsonl"
    import json
    sample.write_text("\n".join(json.dumps(paper) for paper in PAPERS), encoding="utf-8")
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'papers.db'}",
        index_path=str(tmp_path / "index.pkl"),
        sample_path=str(sample),
    )
    messages: list[str] = []
    first = prepare(settings, log=messages.append)
    assert first.index.num_docs == len(PAPERS)
    assert any("sample papers" in message for message in messages)

    messages.clear()
    second = prepare(settings, log=messages.append)
    assert second.index.num_docs == len(PAPERS)
    assert any("Loaded the index" in message for message in messages)
    first.storage.engine.dispose()
    second.storage.engine.dispose()
