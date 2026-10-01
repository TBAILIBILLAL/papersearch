# PaperSearch

[![CI](https://github.com/billy001-11/papersearch/actions/workflows/ci.yml/badge.svg)](https://github.com/billy001-11/papersearch/actions/workflows/ci.yml)

A search engine for computer science papers, built from scratch in Python. It indexes
arXiv titles and abstracts, ranks results with BM25, completes queries as you type and
corrects misspelled words.

The search core uses no search library. The inverted index, the ranking formula, the
autocomplete trie and the spelling correction are all written here, in about 600 lines.

![PaperSearch finding papers for a misspelled query](docs/screenshot.png)

## Features

- **Ranked full-text search** over titles and abstracts with BM25; words in the title count more.
- **Autocomplete** with frequent words and title phrases, most common first.
- **Spelling correction**: "nueral" finds "neural", with a link to search for the original words.
- **Highlighted snippets** showing the part of the abstract that matches best.
- **Category filter** and paging.
- **REST API** with generated documentation at `/docs`, and a single-page web interface.
- **SQLite or PostgreSQL** for the document store, switched with one environment variable.

## Results

Measured on 100,000 computer science papers from arXiv (127,806 distinct terms,
9.1 million postings), on a laptop with an Intel Core i5-1145G7, Python 3.12 and SQLite.

**Speed**

| | Median | 95th percentile |
|---|---|---|
| Search through the index (300 queries) | 6.7 ms | 20.0 ms |
| `LIKE` scan in the database (30 of the same queries) | 105 ms | 507 ms |

The index search is the whole request: ranking every matching paper, reading the top
ten from the database and building the snippets. The `LIKE` scan only returns the first
ten matching rows, unranked, and it is fast only when it finds ten matches early.

Building the index takes 17 seconds and the saved index is 62 MB.

**Ranking quality**

To measure ranking without hand-made judgements, only the abstracts are indexed and
each paper's title is used as a query. A good ranking puts that paper's own abstract
first. Results for 1,000 randomly chosen papers among the 100,000:

| Ranking | Right paper first | Right paper in top 10 | MRR@10 |
|---|---|---|---|
| BM25 (used by the engine) | 85.1% | 94.8% | 0.885 |
| Number of matching query words | 65.6% | 87.5% | 0.752 |
| Raw term frequency | 24.1% | 52.6% | 0.331 |

Reproduce both tables with `papersearch bench` and `papersearch evaluate` after loading
the same amount of data (see [Loading more papers](#loading-more-papers)).

## How it works

```mermaid
flowchart LR
    A[arXiv metadata] -->|ingest| B[(Document store<br>SQLite or PostgreSQL)]
    B -->|build| C[Inverted index]
    B -->|build| D[Word trie and phrase trie]
    Q[Query] --> T[Tokenizer]
    T --> S[Spelling correction]
    D --> S
    S --> R[BM25 ranking]
    C --> R
    R -->|top results| P[Snippets and highlights]
    B --> P
```

1. **Tokenizer** (`tokenizer.py`). Text is lowercased and split into words, stopwords
   are dropped, and plurals are reduced to the singular so "network" matches "networks".
2. **Inverted index** (`index.py`). For every term the index stores a posting list: the
   documents that contain the term and how often. Posting lists are compact typed
   arrays, not Python lists, which keeps 9.1 million postings in a 62 MB file.
3. **BM25** (`index.py`). A query term adds to a document's score according to how often
   it occurs there, how rare the term is in the collection, and how long the document
   is. The length part of the formula is precomputed per document.
4. **Trie** (`trie.py`). Every node stores the highest weight below it, so autocomplete
   walks towards the best completions first and stops after the few it needs.
5. **Spelling correction** (`trie.py`, `spelling.py`). The same trie is searched for
   words within one or two edits of the typed word, carrying one row of the
   edit-distance table per node and abandoning branches that are already too far.
   Swapping two neighbouring letters counts as one edit.
6. **Document store** (`storage.py`). Papers live in a relational table. The index only
   decides which papers to show; their text is then read from the database.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/billy001-11/papersearch.git
cd papersearch
python -m venv .venv
.venv\Scripts\activate          # on macOS or Linux: source .venv/bin/activate
pip install -e .
papersearch serve
```

Open http://127.0.0.1:8000. The first start loads the bundled sample of 2,000 papers
into a SQLite file and builds the index, which takes a second or two.

### With Docker and PostgreSQL

```bash
docker compose up --build
```

This starts PostgreSQL and the application, loads the sample and serves it on
http://localhost:8000.

## Loading more papers

`scripts/fetch_arxiv.py` downloads metadata from arXiv's OAI-PMH service. It needs only
the standard library and waits between requests, as arXiv asks of automated clients.
Downloading 100,000 papers takes 20 to 30 minutes.

```bash
python scripts/fetch_arxiv.py --out data/arxiv-cs.jsonl --from 2026-03-01 --max 100000
papersearch ingest data/arxiv-cs.jsonl
papersearch index
papersearch serve
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/search?q=...&category=...&limit=10&offset=0&correct=true` | Ranked results with highlighted title and snippet |
| `GET /api/suggest?q=...` | Completions for a partly typed query |
| `GET /api/categories` | Categories with paper counts |
| `GET /api/stats` | Size of the index |
| `GET /health` | Liveness check |

```bash
curl "http://127.0.0.1:8000/api/search?q=nueral+networks&limit=1"
```

```json
{
  "query": "nueral networks",
  "corrected_query": "neural networks",
  "total": 24517,
  "took_ms": 31.2,
  "results": [
    {
      "arxiv_id": "...",
      "url": "https://arxiv.org/abs/...",
      "title": [{"text": "Neural", "hit": true}, {"text": " ", "hit": false}, {"text": "Networks", "hit": true}],
      "snippet": [{"text": "... ", "hit": false}],
      "authors": "...",
      "category": "cs.LG",
      "published": "2025-11",
      "score": 9.87
    }
  ]
}
```

Titles and snippets arrive as pieces marked `hit` or not, so the page marks matches
without ever inserting data as HTML. The values above are illustrative; totals and
scores depend on the papers you have loaded.

## Command line

```bash
papersearch search "graph neural networks" --category cs.LG
papersearch suggest "reinf"
papersearch stats
papersearch bench        # speed, against a LIKE scan
papersearch evaluate     # ranking quality
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///data/papersearch.db` | Document store. For PostgreSQL: `postgresql+psycopg://user:password@host/database` (install with `pip install -e ".[postgres]"`) |
| `INDEX_PATH` | `data/index.pkl` | Where the built index is saved |
| `SAMPLE_PATH` | `data/sample.jsonl` | Papers loaded on first start if the store is empty |

## Project layout

```
papersearch/
  tokenizer.py     words, stopwords, plural stemming
  index.py         inverted index and BM25
  trie.py          weighted trie: autocomplete and fuzzy search
  spelling.py      edit distance and the spell checker
  engine.py        search, suggestions, snippets, saving and loading
  storage.py       document store (SQLAlchemy Core)
  evaluation.py    benchmark and ranking evaluation
  bootstrap.py     first-start setup
  api.py           FastAPI application
  cli.py           command line
  static/          the search page
scripts/fetch_arxiv.py   arXiv downloader
tests/                   53 tests
data/sample.jsonl        2,000 papers for a quick start
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests cover the tokenizer, index, ranking, trie, spelling, engine, storage and API.
The trie's autocomplete and fuzzy search are also checked against brute-force
implementations on random data. On every push, GitHub Actions runs the tests on
Python 3.10 and 3.12 with SQLite, again against a real PostgreSQL server, and starts the
Docker Compose setup to search through the running API.

## Design decisions

- **Typed arrays for posting lists.** A Python list of integers costs about 36 bytes
  per number; an `array` costs at most 4. With 9.1 million postings this is the difference
  between fitting comfortably in memory and not.
- **Title boost inside the term frequency.** A title word counts as three abstract
  words. This gives most of the benefit of field-aware ranking with a single posting
  list per term.
- **Best-first search in the trie.** Storing the best weight per subtree makes
  autocomplete cost depend on the number of suggestions wanted, not on the number of
  words under the prefix.
- **Only rare words are corrected.** A word is replaced only if it matches no paper,
  or matches almost none while a word one edit away is at least a hundred times more
  common. Real collections contain typos, so "matches nothing" alone would miss them.
- **A light stemmer.** Only plurals are reduced. A full stemmer would merge more word
  forms, but also unrelated words, and it would make suggestions harder to read.

## Limitations

- No phrase or proximity search: the index stores no word positions.
- The index lives in the memory of one process; it is rebuilt, not updated, when papers are added.
- English only, and the stemmer handles plurals but not verb forms.
- The saved index is a pickle file. Load only index files you created yourself.

## Data and licence

The code is released under the [MIT License](LICENSE).

Paper metadata comes from arXiv, which makes its metadata available under the CC0 1.0
public domain dedication; the papers themselves keep their own licences. Thank you to
arXiv for use of its open access interoperability.
