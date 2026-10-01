"""Measuring ranking quality and speed.

Quality is measured with known-item retrieval: only the abstracts are indexed,
and each sampled paper's title is used as a query. A good ranking puts that
paper's own abstract at or near the top. No hand-made judgements are needed,
so anyone can reproduce the numbers.
"""

from __future__ import annotations

import os
import random
import statistics
import time

from .engine import SearchEngine
from .index import InvertedIndex
from .storage import Storage
from .tokenizer import is_content_word, terms, words

MODES = {
    "bm25": "BM25 (used by the engine)",
    "tf": "Raw term frequency",
    "match": "Number of matching query words",
}


def evaluate_ranking(storage: Storage, sample: int = 1000, seed: int = 7) -> dict:
    """Recall@1, Recall@10 and MRR@10 of each scoring mode on title-to-abstract retrieval."""
    index = InvertedIndex(title_boost=0)
    titles: list[str] = []
    for doc in storage.iter_documents():
        index.add_terms([], terms(doc["abstract"]))
        titles.append(doc["title"])
    index.finish()

    chosen = random.Random(seed).sample(range(index.num_docs), min(sample, index.num_docs))
    results = {}
    for mode in MODES:
        top1 = top10 = 0
        reciprocal = 0.0
        for doc in chosen:
            scores = index.score(terms(titles[doc]), mode)
            own = scores.get(doc)
            if own is None:
                continue
            higher = sum(1 for value in scores.values() if value > own)
            ties = sum(1 for value in scores.values() if value == own) - 1
            rank = higher + 1 + ties / 2  # documents with the same score share the average position
            if rank <= 1:
                top1 += 1
            if rank <= 10:
                top10 += 1
                reciprocal += 1 / rank
        count = len(chosen)
        results[mode] = {"recall_at_1": top1 / count, "recall_at_10": top10 / count, "mrr_at_10": reciprocal / count}
    return {"documents": index.num_docs, "queries": len(chosen), "modes": results}


def sample_queries(engine: SearchEngine, count: int, seed: int = 7) -> list[str]:
    """Realistic queries: two or three consecutive content words taken from random titles."""
    generator = random.Random(seed)
    doc_numbers = generator.sample(range(engine.index.num_docs), min(count * 2, engine.index.num_docs))
    rows = engine.storage.get_documents([engine.db_ids[doc] for doc in doc_numbers])
    queries = []
    for row in rows.values():
        content = [w for w in words(row["title"]) if is_content_word(w)]
        if len(content) < 2:
            continue
        size = min(len(content), generator.choice((2, 3)))
        start = generator.randrange(len(content) - size + 1)
        queries.append(" ".join(content[start : start + size]))
        if len(queries) == count:
            break
    return queries


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def benchmark(engine: SearchEngine, index_path: str | None, queries: int = 300, baseline_queries: int = 30) -> dict:
    """Time full searches through the index, and the same queries as a LIKE scan in the database."""
    query_list = sample_queries(engine, queries)

    engine_ms = []
    for query in query_list:
        started = time.perf_counter()
        engine.search(query)
        engine_ms.append((time.perf_counter() - started) * 1000)

    baseline_ms = []
    for query in query_list[:baseline_queries]:
        query_words = [w for w in words(query) if is_content_word(w)]
        started = time.perf_counter()
        engine.storage.naive_search(query_words, 10)
        baseline_ms.append((time.perf_counter() - started) * 1000)

    stats = engine.stats()
    return {
        "documents": stats["documents"],
        "terms": stats["terms"],
        "postings": stats["postings"],
        "build_seconds": stats["build_seconds"],
        "index_megabytes": round(os.path.getsize(index_path) / 1_000_000, 1) if index_path and os.path.exists(index_path) else None,
        "queries": len(engine_ms),
        "engine_ms": summarise(engine_ms),
        "baseline_queries": len(baseline_ms),
        "baseline_ms": summarise(baseline_ms),
        "database": engine.storage.engine.dialect.name,
    }


def summarise(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p95": None, "p99": None, "mean": None}
    return {
        "median": round(statistics.median(values), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "mean": round(statistics.fmean(values), 2),
    }
