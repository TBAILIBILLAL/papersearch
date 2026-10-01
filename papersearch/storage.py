"""The document store: paper metadata in a relational database.

The same code runs on SQLite (the default, no setup) and on PostgreSQL
(set DATABASE_URL). The search index is built from this table; the database
is also what a result page is read from once the index has chosen the papers.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable, Iterator

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, and_, create_engine, func, insert, or_, select

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("arxiv_id", String(32), nullable=False, unique=True),
    Column("title", Text, nullable=False),
    Column("abstract", Text, nullable=False),
    Column("authors", Text, nullable=False),
    Column("categories", Text, nullable=False),
    Column("primary_category", String(32), nullable=False, index=True),
    Column("published", String(7), nullable=False),  # year and month, e.g. 2024-05
)

_SPACES = re.compile(r"\s+")
_NEW_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
_OLD_ID = re.compile(r"^[a-z\-]+(?:\.[A-Z]{2})?/(\d{2})(\d{2})\d{3}")


def published_from_id(arxiv_id: str) -> str:
    """arXiv identifiers start with the year and month of submission (2405.01234 -> 2024-05)."""
    match = _NEW_ID.match(arxiv_id) or _OLD_ID.match(arxiv_id)
    if not match:
        return "0000-00"
    year, month = int(match.group(1)), match.group(2)
    century = 1900 if year >= 91 else 2000
    return f"{century + year}-{month}"


def clean(text: str) -> str:
    """Collapse line breaks and repeated spaces into single spaces."""
    return _SPACES.sub(" ", text).strip()


def normalise(record: dict) -> dict:
    """Turn one harvested record into a row for the documents table."""
    arxiv_id = record["id"]
    categories = clean(record.get("categories", ""))
    authors = record.get("authors", [])
    if isinstance(authors, list):
        authors = ", ".join(authors)
    return {
        "arxiv_id": arxiv_id,
        "title": clean(record["title"]),
        "abstract": clean(record["abstract"]),
        "authors": clean(authors),
        "categories": categories,
        "primary_category": categories.split(" ")[0] if categories else "unknown",
        "published": published_from_id(arxiv_id),
    }


def read_jsonl(path: str) -> Iterator[dict]:
    """Read records from a file with one JSON object per line."""
    with open(path, encoding="utf-8-sig") as handle:  # also accepts files saved with a byte order mark
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class Storage:
    def __init__(self, url: str):
        if url.startswith("sqlite:///"):
            folder = os.path.dirname(url[len("sqlite:///"):])
            if folder:
                os.makedirs(folder, exist_ok=True)
        self.engine = create_engine(url)
        metadata.create_all(self.engine)

    def count(self) -> int:
        with self.engine.connect() as con:
            return con.execute(select(func.count()).select_from(documents)).scalar_one()

    def add_documents(self, records: Iterable[dict], batch_size: int = 2000) -> int:
        """Insert harvested records, skipping papers that are already stored. Returns how many were added."""
        with self.engine.connect() as con:
            known = set(con.execute(select(documents.c.arxiv_id)).scalars())

        added = 0
        batch: list[dict] = []
        with self.engine.begin() as con:
            for record in records:
                if not record.get("title") or not record.get("abstract") or record["id"] in known:
                    continue
                known.add(record["id"])
                batch.append(normalise(record))
                if len(batch) >= batch_size:
                    con.execute(insert(documents), batch)
                    added += len(batch)
                    batch = []
            if batch:
                con.execute(insert(documents), batch)
                added += len(batch)
        return added

    def iter_documents(self, batch_size: int = 5000) -> Iterator[dict]:
        """All documents in id order, read in batches so memory use stays flat."""
        last_id = 0
        while True:
            query = select(documents).where(documents.c.id > last_id).order_by(documents.c.id).limit(batch_size)
            with self.engine.connect() as con:
                rows = con.execute(query).mappings().all()
            if not rows:
                return
            for row in rows:
                yield dict(row)
            last_id = rows[-1]["id"]

    def get_documents(self, ids: list[int]) -> dict[int, dict]:
        """The documents with the given database ids, keyed by id."""
        if not ids:
            return {}
        with self.engine.connect() as con:
            rows = con.execute(select(documents).where(documents.c.id.in_(ids))).mappings().all()
        return {row["id"]: dict(row) for row in rows}

    def naive_search(self, query_words: list[str], limit: int = 10) -> list[int]:
        """The baseline the index is measured against: scan every row with LIKE.

        Returns ids of documents whose title or abstract contains every query word.
        There is no ranking, and the database has to read the whole table.
        """
        if not query_words:
            return []
        conditions = [
            or_(func.lower(documents.c.title).like(f"%{word}%"), func.lower(documents.c.abstract).like(f"%{word}%"))
            for word in query_words
        ]
        query = select(documents.c.id).where(and_(*conditions)).order_by(documents.c.id).limit(limit)
        with self.engine.connect() as con:
            return list(con.execute(query).scalars())
