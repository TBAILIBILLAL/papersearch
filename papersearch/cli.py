"""Command line interface: python -m papersearch <command>."""

from __future__ import annotations

import argparse
import json
import sys

from .bootstrap import prepare
from .config import load_settings
from .engine import SearchEngine
from .evaluation import MODES, benchmark, evaluate_ranking
from .storage import Storage, read_jsonl


def command_ingest(args) -> None:
    storage = Storage(load_settings().database_url)
    added = storage.add_documents(read_jsonl(args.path))
    print(f"Added {added} papers. The store now holds {storage.count()}.")


def command_index(args) -> None:
    settings = load_settings()
    engine = SearchEngine.build(Storage(settings.database_url))
    engine.save(settings.index_path)
    stats = engine.stats()
    print(f"Indexed {stats['documents']} papers in {stats['build_seconds']} s: "
          f"{stats['terms']} terms, {stats['postings']} postings. Saved to {settings.index_path}.")


def command_search(args) -> None:
    engine = prepare(load_settings(), log=lambda message: None)
    response = engine.search(args.query, args.category, args.limit)
    if response["corrected_query"]:
        print(f"Showing results for: {response['corrected_query']}")
    print(f"{response['total']} matches in {response['took_ms']} ms\n")
    for position, result in enumerate(response["results"], 1):
        title = "".join(piece["text"] for piece in result["title"])
        print(f"{position:2}. {title}")
        print(f"    {result['arxiv_id']}  {result['category']}  {result['published']}  score {result['score']}")


def command_suggest(args) -> None:
    engine = prepare(load_settings(), log=lambda message: None)
    for suggestion in engine.suggest(args.text):
        print(suggestion)


def command_stats(args) -> None:
    engine = prepare(load_settings(), log=lambda message: None)
    print(json.dumps(engine.stats(), indent=2))


def command_serve(args) -> None:
    import uvicorn

    from .api import create_app

    engine = prepare(load_settings())
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


def command_bench(args) -> None:
    settings = load_settings()
    engine = prepare(settings, log=lambda message: None)
    result = benchmark(engine, settings.index_path, args.queries, args.baseline_queries)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    fast, slow = result["engine_ms"], result["baseline_ms"]
    print(f"Collection: {result['documents']} papers, {result['terms']} terms, {result['postings']} postings")
    print(f"Index build: {result['build_seconds']} s, {result['index_megabytes']} MB on disk")
    print(f"Search through the index ({result['queries']} queries): "
          f"median {fast['median']} ms, p95 {fast['p95']} ms, p99 {fast['p99']} ms")
    print(f"LIKE scan in {result['database']} ({result['baseline_queries']} queries): "
          f"median {slow['median']} ms, p95 {slow['p95']} ms")


def command_evaluate(args) -> None:
    result = evaluate_ranking(Storage(load_settings().database_url), args.sample)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Title-to-abstract retrieval: {result['queries']} queries over {result['documents']} abstracts\n")
    print(f"{'Ranking':34} {'Recall@1':>9} {'Recall@10':>10} {'MRR@10':>8}")
    for mode, label in MODES.items():
        scores = result["modes"][mode]
        print(f"{label:34} {scores['recall_at_1']:>9.3f} {scores['recall_at_10']:>10.3f} {scores['mrr_at_10']:>8.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="papersearch", description="A search engine for computer science papers.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="load papers from a JSON-lines file into the document store")
    ingest.add_argument("path")
    ingest.set_defaults(run=command_ingest)

    index = commands.add_parser("index", help="build the search index from the document store")
    index.set_defaults(run=command_index)

    search = commands.add_parser("search", help="search from the terminal")
    search.add_argument("query")
    search.add_argument("--category")
    search.add_argument("--limit", type=int, default=10)
    search.set_defaults(run=command_search)

    suggest = commands.add_parser("suggest", help="show completions for a partly typed query")
    suggest.add_argument("text")
    suggest.set_defaults(run=command_suggest)

    stats = commands.add_parser("stats", help="show facts about the index")
    stats.set_defaults(run=command_stats)

    serve = commands.add_parser("serve", help="start the web server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(run=command_serve)

    bench = commands.add_parser("bench", help="measure search speed against a database scan")
    bench.add_argument("--queries", type=int, default=300)
    bench.add_argument("--baseline-queries", type=int, default=30)
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(run=command_bench)

    evaluate = commands.add_parser("evaluate", help="measure ranking quality")
    evaluate.add_argument("--sample", type=int, default=1000)
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(run=command_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
