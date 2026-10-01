"""The inverted index and BM25 ranking.

An inverted index maps each term to the documents that contain it. Instead of
scanning every document for a query word, the engine looks the word up once
and gets the list of matching documents directly.

For each term the index keeps a posting list: two parallel arrays holding the
document numbers and how often the term occurs in each of them.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter

from .tokenizer import terms

TITLE_BOOST = 3  # a word in the title counts as much as three in the abstract
MAX_TF = 65535  # term frequencies are stored in 16 bits


class InvertedIndex:
    def __init__(self, k1: float = 1.2, b: float = 0.75, title_boost: int = TITLE_BOOST):
        self.k1 = k1
        self.b = b
        self.title_boost = title_boost
        self.postings: dict[str, tuple[array, array]] = {}
        self.doc_len = array("I")
        self.num_docs = 0
        self.avg_len = 0.0
        self._norm = array("f")

    # ------------------------------------------------------------------ building

    def add(self, title: str, abstract: str) -> int:
        """Index one document and return its number (0, 1, 2, ...)."""
        return self.add_terms(terms(title), terms(abstract))

    def add_terms(self, title_terms: list[str], abstract_terms: list[str]) -> int:
        """Index one document whose text has already been turned into terms."""
        counts = Counter(abstract_terms)
        length = len(abstract_terms)
        if self.title_boost:
            for term in title_terms:
                counts[term] += self.title_boost
                length += self.title_boost

        doc = self.num_docs
        postings = self.postings
        for term, tf in counts.items():
            entry = postings.get(term)
            if entry is None:
                entry = postings[term] = (array("I"), array("H"))
            entry[0].append(doc)
            entry[1].append(tf if tf < MAX_TF else MAX_TF)

        self.doc_len.append(length)
        self.num_docs += 1
        return doc

    def finish(self) -> None:
        """Compute the values that depend on the whole collection. Call once after adding."""
        total = sum(self.doc_len)
        self.avg_len = total / self.num_docs if self.num_docs else 0.0
        k1, b, avg = self.k1, self.b, self.avg_len or 1.0
        # the length part of the BM25 formula is the same for every query, so it is stored per document
        self._norm = array("f", (k1 * (1.0 - b + b * length / avg) for length in self.doc_len))

    # ------------------------------------------------------------------ querying

    def df(self, term: str) -> int:
        """Document frequency: the number of documents that contain the term."""
        entry = self.postings.get(term)
        return len(entry[0]) if entry else 0

    def idf(self, term: str) -> float:
        """Inverse document frequency: rare terms get a high value, common terms a low one."""
        df = self.df(term)
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], mode: str = "bm25") -> dict[int, float]:
        """Score every document that contains at least one query term.

        mode "bm25" is the ranking used by the engine. "tf" (raw term counts) and
        "match" (number of query terms present) exist as baselines for evaluation.
        """
        scores: dict[int, float] = {}
        get = scores.get
        k1_plus_1 = self.k1 + 1.0
        norm = self._norm

        for term in set(query_terms):
            entry = self.postings.get(term)
            if entry is None:
                continue
            docs, tfs = entry
            if mode == "bm25":
                weight = self.idf(term) * k1_plus_1
                for doc, tf in zip(docs, tfs):
                    scores[doc] = get(doc, 0.0) + weight * tf / (tf + norm[doc])
            elif mode == "tf":
                for doc, tf in zip(docs, tfs):
                    scores[doc] = get(doc, 0.0) + tf
            elif mode == "match":
                for doc in docs:
                    scores[doc] = get(doc, 0.0) + 1.0
            else:
                raise ValueError(f"unknown scoring mode: {mode}")
        return scores

    # ------------------------------------------------------------------ persistence

    def to_state(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "title_boost": self.title_boost,
            "postings": self.postings,
            "doc_len": self.doc_len,
        }

    @classmethod
    def from_state(cls, state: dict) -> "InvertedIndex":
        index = cls(state["k1"], state["b"], state["title_boost"])
        index.postings = state["postings"]
        index.doc_len = state["doc_len"]
        index.num_docs = len(index.doc_len)
        index.finish()
        return index
