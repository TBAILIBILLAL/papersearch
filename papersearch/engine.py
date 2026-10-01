"""The search engine: index, ranking, spelling, suggestions and snippets working together."""

from __future__ import annotations

import heapq
import os
import pickle
import re
import time
from array import array
from collections import Counter

from .index import InvertedIndex
from .spelling import SpellChecker
from .storage import Storage
from .tokenizer import STOPWORDS, is_content_word, stem, words
from .trie import Trie

STATE_VERSION = 1
SNIPPET_WORDS = 45
RARE_WORD_FACTOR = 100  # a typo candidate must be this many times more common than the rare word
_ANY_WORD = re.compile(r"[A-Za-z0-9]+")
# words that make a poor first or last word of a suggested phrase ("learning via", "novel method")
_WEAK_EDGES = frozenset({"via", "based", "toward", "towards", "new", "novel", "without", "beyond", "across"})


def _good_edge(word: str) -> bool:
    return is_content_word(word) and word not in _WEAK_EDGES


def title_phrases(title_words: list[str]) -> list[str]:
    """Two- and three-word phrases from a title that start and end with a meaningful word."""
    phrases = []
    for size in (2, 3):
        for i in range(len(title_words) - size + 1):
            chunk = title_words[i : i + size]
            if _good_edge(chunk[0]) and _good_edge(chunk[-1]) and all(len(w) > 1 for w in chunk):
                phrases.append(" ".join(chunk))
    return phrases


def highlight(text: str, query_terms: set[str], max_words: int | None = None) -> list[dict]:
    """Split text into pieces, marking the words that match the query.

    With max_words, only the stretch of text that contains the most matches is
    returned. The result is a list of {"text": ..., "hit": bool}, so the page can
    mark matches without building HTML from strings.
    """
    matches = list(_ANY_WORD.finditer(text))
    if not matches:
        return [{"text": text, "hit": False}] if text else []
    hits = [stem(m.group().lower()) in query_terms for m in matches]

    first, last = 0, len(matches)
    if max_words and len(matches) > max_words:
        current = best = sum(hits[:max_words])
        for start in range(1, len(matches) - max_words + 1):
            current += hits[start + max_words - 1] - hits[start - 1]
            if current > best:
                best, first = current, start
        last = first + max_words

    begin = matches[first].start() if first > 0 else 0
    end = matches[last - 1].end() if last < len(matches) else len(text)

    pieces: list[dict] = []
    position = begin
    for match, hit in zip(matches[first:last], hits[first:last]):
        if hit:
            if match.start() > position:
                pieces.append({"text": text[position : match.start()], "hit": False})
            pieces.append({"text": match.group(), "hit": True})
            position = match.end()
    if position < end:
        pieces.append({"text": text[position:end], "hit": False})

    if begin > 0:
        pieces.insert(0, {"text": "... ", "hit": False})
    if end < len(text):
        pieces.append({"text": " ...", "hit": False})
    return pieces


class SearchEngine:
    def __init__(self) -> None:
        self.index = InvertedIndex()
        self.db_ids = array("I")  # document number -> id in the database
        self.categories: list[str] = []
        self.doc_category = array("H")  # document number -> position in self.categories
        self.word_counts: dict[str, int] = {}
        self.phrase_counts: dict[str, int] = {}
        self.words = Trie()
        self.phrases = Trie()
        self.speller = SpellChecker(self.words)
        self.storage: Storage | None = None
        self.build_seconds = 0.0
        self._stats: dict | None = None

    # ------------------------------------------------------------------ building

    @classmethod
    def build(cls, storage: Storage) -> "SearchEngine":
        """Read every document from the store and build all search structures."""
        started = time.perf_counter()
        engine = cls()
        engine.storage = storage
        category_ids: dict[str, int] = {}
        word_counts: Counter = Counter()
        phrase_counts: Counter = Counter()

        for doc in storage.iter_documents():
            title_words = words(doc["title"])
            abstract_words = words(doc["abstract"])
            title_content = [w for w in title_words if len(w) > 1 and w not in STOPWORDS]
            abstract_content = [w for w in abstract_words if len(w) > 1 and w not in STOPWORDS]

            engine.index.add_terms([stem(w) for w in title_content], [stem(w) for w in abstract_content])
            engine.db_ids.append(doc["id"])
            category = doc["primary_category"]
            if category not in category_ids:
                category_ids[category] = len(engine.categories)
                engine.categories.append(category)
            engine.doc_category.append(category_ids[category])

            word_counts.update(set(title_content))
            word_counts.update(set(abstract_content) - set(title_content))
            phrase_counts.update(title_phrases(title_words))

        engine.index.finish()

        # words and phrases seen only a handful of times are mostly noise for suggestions
        minimum = max(2, engine.index.num_docs // 20000)
        engine.word_counts = {w: n for w, n in word_counts.items() if n >= minimum}
        engine.phrase_counts = {p: n * len(p.split(" ")) for p, n in phrase_counts.items() if n >= minimum}
        engine._build_tries()
        engine.build_seconds = time.perf_counter() - started
        return engine

    def _build_tries(self) -> None:
        self.words = Trie()
        for word, count in self.word_counts.items():
            self.words.insert(word, count)
        self.phrases = Trie()
        for phrase, weight in self.phrase_counts.items():
            self.phrases.insert(phrase, weight)
        self.speller = SpellChecker(self.words)

    # ------------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        """Write the index to disk so the next start does not have to rebuild it."""
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        state = {
            "version": STATE_VERSION,
            "index": self.index.to_state(),
            "db_ids": self.db_ids,
            "categories": self.categories,
            "doc_category": self.doc_category,
            "word_counts": self.word_counts,
            "phrase_counts": self.phrase_counts,
            "build_seconds": self.build_seconds,
        }
        temporary = path + ".tmp"
        with open(temporary, "wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str, storage: Storage) -> "SearchEngine":
        """Read an index written by save(). Only load index files you created yourself."""
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        if state.get("version") != STATE_VERSION:
            raise ValueError("the index file was written by a different version; rebuild it")
        engine = cls()
        engine.storage = storage
        engine.index = InvertedIndex.from_state(state["index"])
        engine.db_ids = state["db_ids"]
        engine.categories = state["categories"]
        engine.doc_category = state["doc_category"]
        engine.word_counts = state["word_counts"]
        engine.phrase_counts = state["phrase_counts"]
        engine.build_seconds = state["build_seconds"]
        engine._build_tries()
        return engine

    # ------------------------------------------------------------------ searching

    def correct_word(self, word: str) -> str | None:
        """A replacement for a query word that looks misspelled, or None to keep the word."""
        found = self.index.df(stem(word))
        if found == 0:
            return self.speller.correct(word)
        if word in self.words:
            return None
        # The word is very rare. Real collections contain typos, so a rare word that is one
        # edit away from a far more common word is treated as a typo of that word.
        candidate = self.speller.correct(word, max_distance=1)
        if candidate and self.words.weight(candidate) >= RARE_WORD_FACTOR * found:
            return candidate
        return None

    def correct_query(self, query: str, correct: bool = True) -> tuple[list[str], str | None]:
        """Returns the content words of the query (misspellings fixed) and the corrected query, if it changed."""
        output: list[str] = []
        content: list[str] = []
        changed = False
        for word in words(query):
            if is_content_word(word):
                replacement = self.correct_word(word) if correct else None
                if replacement:
                    word, changed = replacement, True
                content.append(word)
            output.append(word)
        return content, (" ".join(output) if changed else None)

    def rank(self, query_terms: list[str], category: str | None = None, limit: int = 10, offset: int = 0):
        """Score the matching documents and return (total matches, top (document, score) pairs)."""
        scores = self.index.score(query_terms)
        if category:
            if category not in self.categories:
                return 0, []
            wanted = self.categories.index(category)
            doc_category = self.doc_category
            scores = {doc: score for doc, score in scores.items() if doc_category[doc] == wanted}
        # highest score first; equal scores keep a stable order by document number
        top = heapq.nlargest(offset + limit, scores.items(), key=lambda item: (item[1], -item[0]))
        return len(scores), top[offset:]

    def search(self, query: str, category: str | None = None, limit: int = 10, offset: int = 0,
               correct: bool = True) -> dict:
        started = time.perf_counter()
        content_words, corrected = self.correct_query(query, correct)
        query_terms = [stem(w) for w in content_words]
        total, top = self.rank(query_terms, category, limit, offset)

        rows = self.storage.get_documents([self.db_ids[doc] for doc, _ in top])
        term_set = set(query_terms)
        results = []
        for doc, score in top:
            row = rows[self.db_ids[doc]]
            results.append(
                {
                    "arxiv_id": row["arxiv_id"],
                    "url": f"https://arxiv.org/abs/{row['arxiv_id']}",
                    "title": highlight(row["title"], term_set),
                    "snippet": highlight(row["abstract"], term_set, SNIPPET_WORDS),
                    "authors": row["authors"],
                    "category": row["primary_category"],
                    "published": row["published"],
                    "score": round(score, 3),
                }
            )
        return {
            "query": query,
            "corrected_query": corrected,
            "total": total,
            "offset": offset,
            "took_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": results,
        }

    # ------------------------------------------------------------------ suggestions

    def suggest(self, text: str, limit: int = 8) -> list[str]:
        """Complete what the user is typing with frequent words and title phrases."""
        typed = " ".join(words(text))
        if not typed:
            return []
        if text[-1:].isspace():
            typed += " "  # the user finished a word and is starting the next one

        suggestions: list[str] = []

        def add(candidates: list[str]) -> None:
            for candidate in candidates:
                if candidate != typed.strip() and candidate not in suggestions:
                    suggestions.append(candidate)

        head, _, last = typed.rpartition(" ")
        if not head:
            # one word typed: the most common words first, with room left for phrases
            phrases = [key for key, _ in self.phrases.complete(typed, limit)]
            completed = [key for key, _ in self.words.complete(last, limit)]
            add(completed[: limit - min(len(phrases), limit // 2)] + phrases)
            return suggestions[:limit]

        # Several words typed. Phrases are at most three words long, so try to continue the
        # last three, two and one words in turn, keeping everything typed before them.
        typed_words = typed.split(" ")  # the last element is "" when the text ends with a space
        for size in (3, 2, 1):
            if size > len(typed_words):
                continue
            kept = " ".join(typed_words[:-size])
            context = " ".join(typed_words[-size:])
            if not context.strip():
                continue
            found = [key for key, _ in self.phrases.complete(context, limit)]
            add([f"{kept} {key}".strip() for key in found])
            if len(suggestions) >= limit // 2:
                return suggestions[:limit]

        # no phrase continues the text: complete the last word on its own
        if last:
            add([f"{head} {word}" for word, _ in self.words.complete(last, limit)])
        return suggestions[:limit]

    # ------------------------------------------------------------------ facts

    def category_counts(self) -> list[tuple[str, int]]:
        """Categories with the number of papers in each, largest first."""
        counts = Counter(self.doc_category)
        return sorted(((self.categories[i], n) for i, n in counts.items()), key=lambda item: (-item[1], item[0]))

    def stats(self) -> dict:
        if self._stats is None:
            self._stats = {
                "documents": self.index.num_docs,
                "terms": len(self.index.postings),
                "postings": sum(len(entry[0]) for entry in self.index.postings.values()),
                "average_length": round(self.index.avg_len, 1),
                "suggestion_words": len(self.words),
                "suggestion_phrases": len(self.phrases),
                "categories": len(self.categories),
                "build_seconds": round(self.build_seconds, 2),
            }
        return self._stats
