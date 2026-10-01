"""Download paper metadata from arXiv into a JSON-lines file.

arXiv offers its metadata (titles, abstracts, authors, categories) through the
OAI-PMH protocol. This script asks for one set and one date range, follows the
"resumption tokens" from page to page, and writes one JSON object per paper.

    python scripts/fetch_arxiv.py --out data/arxiv-cs.jsonl --from 2026-01-01 --max 100000

It waits between requests, as arXiv asks of automated clients, and uses only
the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree

ENDPOINT = "https://oaipmh.arxiv.org/oai"
OAI = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV = "{http://arxiv.org/OAI/arXiv/}"
PAUSE_SECONDS = 3.0


def fetch(url: str, attempts: int = 6) -> bytes:
    """Download one page, waiting and retrying when the server is busy."""
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "papersearch/1.0 (metadata harvester)"})
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts:
                raise
            wait = float(error.headers.get("Retry-After") or 10 * attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts:
                raise
            wait = 10 * attempt
        print(f"  server busy, retrying in {wait:.0f} s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError("unreachable")


def text_of(element, tag: str) -> str:
    child = element.find(ARXIV + tag)
    return child.text or "" if child is not None else ""


def parse_page(xml: bytes) -> tuple[list[dict], str | None]:
    """Return the records on one page and the token for the next page, if there is one."""
    root = ElementTree.fromstring(xml)
    error = root.find(OAI + "error")
    if error is not None:
        if error.get("code") == "noRecordsMatch":
            return [], None
        raise RuntimeError(f"arXiv returned an error: {error.get('code')}: {error.text}")

    records = []
    for record in root.iter(OAI + "record"):
        meta = record.find(f"{OAI}metadata/{ARXIV}arXiv")
        if meta is None:
            continue  # a deleted record has a header but no metadata
        authors = []
        for author in meta.iter(ARXIV + "author"):
            name = f"{text_of(author, 'forenames')} {text_of(author, 'keyname')}".strip()
            if name:
                authors.append(name)
        records.append(
            {
                "id": text_of(meta, "id"),
                "title": " ".join(text_of(meta, "title").split()),
                "abstract": " ".join(text_of(meta, "abstract").split()),
                "authors": authors,
                "categories": text_of(meta, "categories"),
            }
        )

    token = root.find(f"{OAI}ListRecords/{OAI}resumptionToken")
    return records, (token.text if token is not None and token.text else None)


def harvest(out_path: str, set_name: str, date_from: str | None, date_until: str | None, maximum: int) -> int:
    query = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_name}
    if date_from:
        query["from"] = date_from
    if date_until:
        query["until"] = date_until
    url = ENDPOINT + "?" + urllib.parse.urlencode(query)

    seen: set[str] = set()
    with open(out_path, "w", encoding="utf-8") as out:
        while url and len(seen) < maximum:
            records, token = parse_page(fetch(url))
            for record in records:
                if record["id"] and record["id"] not in seen and record["title"] and record["abstract"]:
                    seen.add(record["id"])
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if len(seen) >= maximum:
                        break
            print(f"  {len(seen)} papers so far", file=sys.stderr)
            url = ENDPOINT + "?" + urllib.parse.urlencode({"verb": "ListRecords", "resumptionToken": token}) if token else None
            if url:
                time.sleep(PAUSE_SECONDS)
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv paper metadata with OAI-PMH.")
    parser.add_argument("--out", required=True, help="output file, one JSON object per line")
    parser.add_argument("--set", default="cs", dest="set_name", help="arXiv set, for example cs or math")
    parser.add_argument("--from", dest="date_from", help="first modification date, YYYY-MM-DD")
    parser.add_argument("--until", dest="date_until", help="last modification date, YYYY-MM-DD")
    parser.add_argument("--max", type=int, default=100000, dest="maximum", help="stop after this many papers")
    args = parser.parse_args()

    count = harvest(args.out, args.set_name, args.date_from, args.date_until, args.maximum)
    print(f"Wrote {count} papers to {args.out}")


if __name__ == "__main__":
    main()
