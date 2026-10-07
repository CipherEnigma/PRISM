"""Search the local PRISM index from the command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism.candidates import describe
from prism.config import INDEX_DIR, VARIANTS
from prism.explain import explain
from prism.index import Index
from prism.parser import QuerySyntaxError, parse
from prism.search import search, set_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the PRISM paper index.")
    parser.add_argument("query", nargs="+", help="query text; quote phrases and the whole query as needed")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="V2")
    parser.add_argument("-k", type=int, default=10, help="number of results to show")
    parser.add_argument("--explain", action="store_true", help="show per-term score details")
    parser.add_argument("--plan", action="store_true", help="show query candidate plan")
    args = parser.parse_args(argv)
    query = " ".join(args.query)
    if args.k < 0:
        parser.error("-k must be nonnegative")
    try:
        index = Index.load(INDEX_DIR)
        set_index(index)
        parsed = parse(query)
        if args.plan:
            print(describe(parsed, index, use_champions=VARIANTS[args.variant].champions))
        results = search(query, k=args.k, variant=args.variant)
    except QuerySyntaxError as exc:
        print(f"Query error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for rank, result in enumerate(results, 1):
        print(f"{rank:>3}. {result.score:.8f}  {result.doc_id}  {result.title}")
        if args.explain:
            print(explain(result, parsed, index, variant=args.variant))
    if not results:
        print("No matching papers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
