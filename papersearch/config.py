"""Settings, read from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    index_path: str
    sample_path: str


def load_settings() -> Settings:
    return Settings(
        database_url=os.environ.get("DATABASE_URL", "sqlite:///data/papersearch.db"),
        index_path=os.environ.get("INDEX_PATH", "data/index.pkl"),
        sample_path=os.environ.get("SAMPLE_PATH", "data/sample.jsonl"),
    )
