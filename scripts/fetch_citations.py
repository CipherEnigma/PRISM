"""Fetch citation counts for the corpus from OpenAlex into data/citations.jsonl (resumable, cached).

    python scripts/fetch_citations.py --dry-run                 how many requests would be made, no network
    python scripts/fetch_citations.py --limit 200               a small trial first
    OPENALEX_API_KEY=... python scripts/fetch_citations.py      the full run (a free key gives a larger daily budget)

Then put the counts into the index without rebuilding it:
    python scripts/patch_index_metadata.py --src index/ --out index/ --citations data/citations.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism.citations import BudgetExhausted, fetch_citations, load_cache, plan   # noqa: E402
from prism.config import DATA_DIR                                                 # noqa: E402
from prism.loader import iter_records                                             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch OpenAlex citation counts for the PRISM corpus.")
    ap.add_argument("--cache", type=Path, default=DATA_DIR / "citations.jsonl")
    ap.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY", ""), help="or set OPENALEX_API_KEY")
    ap.add_argument("--limit", type=int, default=None, help="look up at most this many papers (for a trial)")
    ap.add_argument("--pause", type=float, default=0.1, help="seconds between requests")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and make no requests")
    args = ap.parse_args()

    records = list(iter_records())
    cache = load_cache(args.cache)
    p = plan(records, cache)
    print(f"{len(records):,} papers, {len(cache):,} already cached; to look up: {p['papers_to_look_up']:,} papers, "
          f"{p['distinct_dois']:,} DOIs ({p['doi_requests']:,} requests), "
          f"up to {p['max_pmid_requests'] - p['doi_requests']:,} more requests for PubMed IDs")
    print("API key:", "set" if args.api_key else "not set (the free no-key budget is much smaller)")
    if args.dry_run:
        return 0
    try:
        stats = fetch_citations(records, args.cache, api_key=args.api_key, pause=args.pause, limit=args.limit)
    except BudgetExhausted as err:
        print(f"\nstopped: {err}. Everything fetched so far is saved in {args.cache}; run again later to resume.")
        return 1
    done = load_cache(args.cache)
    have = sum(1 for v in done.values() if v.get("citations") is not None)
    print(f"requests {stats['requests']:,}; found by DOI {stats['found_by_doi']:,}, by PubMed ID {stats['found_by_pmid']:,}, "
          f"not found {stats['not_found']:,}")
    print(f"coverage: {have:,} of {len(records):,} papers have a citation count ({have / len(records):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
