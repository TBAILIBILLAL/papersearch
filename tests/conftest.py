"""Shared fixtures: a tiny collection of papers and an engine built on it."""

from __future__ import annotations

import os

import pytest

from papersearch.engine import SearchEngine
from papersearch.storage import Storage, metadata

PAPERS = [
    {
        "id": "2401.00001",
        "title": "Graph Neural Networks for Traffic Forecasting",
        "abstract": "We study graph neural networks that predict road traffic. The networks learn "
                    "spatial structure from the road graph and temporal patterns from sensor data.",
        "authors": ["Ada Chen", "Omar Diallo"],
        "categories": "cs.LG cs.AI",
    },
    {
        "id": "2401.00002",
        "title": "A Survey of Reinforcement Learning in Robotics",
        "abstract": "Reinforcement learning lets robots learn control policies from trial and error. "
                    "This survey reviews algorithms, simulators and open problems.",
        "authors": ["Lena Fischer"],
        "categories": "cs.RO cs.LG",
    },
    {
        "id": "2402.00003",
        "title": "Efficient Transformers for Long Documents",
        "abstract": "Attention in transformers grows quadratically with length. We propose sparse attention "
                    "that keeps accuracy on long documents while reducing memory.",
        "authors": ["Wei Zhang", "Sara Lopez", "Tom Baker"],
        "categories": "cs.CL",
    },
    {
        "id": "2403.00004",
        "title": "Traffic Signal Control with Deep Reinforcement Learning",
        "abstract": "A deep reinforcement learning agent controls traffic signals at an intersection. "
                    "The agent reduces waiting time compared with fixed timing plans.",
        "authors": ["Omar Diallo"],
        "categories": "cs.AI",
    },
    {
        "id": "2403.00005",
        "title": "Query Optimization in Distributed Databases",
        "abstract": "We describe a cost model for queries over distributed databases and an optimizer "
                    "that chooses join orders. Queries run faster on standard benchmarks.",
        "authors": ["Priya Raman", "Jon Park"],
        "categories": "cs.DB",
    },
    {
        "id": "cs/0501006",
        "title": "Neural Networks and Graph Theory",
        "abstract": "This note connects neural networks with classical results from graph theory.",
        "authors": ["Erik Olsen"],
        "categories": "cs.DM",
    },
]


@pytest.fixture()
def storage(tmp_path):
    """An empty document store: SQLite in a temporary folder, or the database named by DATABASE_URL."""
    url = os.environ.get("DATABASE_URL") or f"sqlite:///{tmp_path / 'test.db'}"
    store = Storage(url)
    metadata.drop_all(store.engine)
    metadata.create_all(store.engine)
    yield store
    store.engine.dispose()


@pytest.fixture()
def engine(storage):
    storage.add_documents(PAPERS)
    return SearchEngine.build(storage)
