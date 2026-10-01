"""Getting from an empty folder to a running engine."""

from __future__ import annotations

import os

from .config import Settings
from .engine import SearchEngine
from .storage import Storage, read_jsonl


def prepare(settings: Settings, log=print) -> SearchEngine:
    """Open the document store, load the sample data if it is empty, and load or build the index."""
    storage = Storage(settings.database_url)

    if storage.count() == 0 and os.path.exists(settings.sample_path):
        added = storage.add_documents(read_jsonl(settings.sample_path))
        log(f"Loaded {added} sample papers from {settings.sample_path}")

    documents = storage.count()
    if os.path.exists(settings.index_path):
        try:
            engine = SearchEngine.load(settings.index_path, storage)
            if engine.index.num_docs == documents:
                log(f"Loaded the index for {documents} papers from {settings.index_path}")
                return engine
            log("The index is out of date; rebuilding it")
        except (ValueError, KeyError, EOFError) as error:
            log(f"The index file could not be used ({error}); rebuilding it")

    engine = SearchEngine.build(storage)
    engine.save(settings.index_path)
    log(f"Indexed {documents} papers in {engine.build_seconds:.1f} s")
    return engine
