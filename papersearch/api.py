"""The web API and the search page."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from . import __version__
from .bootstrap import prepare
from .config import load_settings
from .engine import SearchEngine

STATIC = Path(__file__).parent / "static"


def create_app(engine: SearchEngine | None = None) -> FastAPI:
    """Build the application. Pass an engine to use it as is (the tests do); otherwise one is prepared at start-up."""
    state: dict = {"engine": engine}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["engine"] is None:
            state["engine"] = prepare(load_settings())
        yield

    app = FastAPI(
        title="PaperSearch",
        version=__version__,
        description="Search computer science papers with a from-scratch inverted index and BM25 ranking.",
        lifespan=lifespan,
    )

    def current() -> SearchEngine:
        return state["engine"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": current().index.num_docs}

    @app.get("/api/search")
    def search(
        q: str = Query(..., min_length=1, max_length=200, description="The search query"),
        category: str | None = Query(None, max_length=32, description="Only papers with this primary category"),
        limit: int = Query(10, ge=1, le=50),
        offset: int = Query(0, ge=0, le=1000),
        correct: bool = Query(True, description="Fix words that look misspelled"),
    ) -> dict:
        return current().search(q, category or None, limit, offset, correct)

    @app.get("/api/suggest")
    def suggest(q: str = Query(..., min_length=1, max_length=100), limit: int = Query(8, ge=1, le=20)) -> dict:
        return {"query": q, "suggestions": current().suggest(q, limit)}

    @app.get("/api/categories")
    def categories() -> dict:
        return {"categories": [{"name": name, "papers": count} for name, count in current().category_counts()]}

    @app.get("/api/stats")
    def stats() -> dict:
        return current().stats()

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
