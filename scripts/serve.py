"""Local web front end for PRISM: python scripts/serve.py [--index index/] [--port 8000]

Loads the index once, serves web/index.html and answers GET /api/search with the same
search(), describe() and explain() output the CLI prints. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prism.candidates import describe  # noqa: E402
from prism.config import INDEX_DIR, VARIANTS  # noqa: E402
from prism.explain import explain  # noqa: E402
from prism.index import Index  # noqa: E402
from prism.parser import QuerySyntaxError, parse  # noqa: E402
from prism.search import search, set_index  # noqa: E402

PAGE = ROOT / "web" / "index.html"
DEFAULT_VARIANT = "V2"
VARIANT_LABELS = {
    "V0": "Flat tf-idf",
    "V1": "Zone-weighted tf-idf",
    "V2": "Zones + authority",
    "V3": "Champion lists",
    "V4": "Phrase boost",
    "V5": "Age-normalized authority",
    "V6": "Query-gated authority",
    "V7": "Adaptive zone weights",
    "R": "BM25 reference",
}
MAX_K = 50


def run_query(index: Index, query: str, variant: str, k: int) -> dict:
    started = time.perf_counter()
    pq = parse(query)
    results = search(query, k=k, variant=variant)
    elapsed_ms = (time.perf_counter() - started) * 1000
    plan = describe(pq, index, use_champions=VARIANTS[variant].champions)
    rows = []
    for rank, r in enumerate(results, 1):
        doc = index.internal_id(r.doc_id)
        year = index.field_value(doc, "year")
        rows.append({
            "rank": rank,
            "doc_id": r.doc_id,
            "title": r.title,
            "score": r.score,
            "zone_scores": r.zone_scores,
            "authority": r.authority,
            "beta": r.beta,
            "year": year or None,
            "journal": index.field_value(doc, "journal"),
            "explain": explain(r, pq, index, variant=variant),
        })
    return {
        "query": query,
        "variant": variant,
        "terms": pq.terms,
        "restricted": not plan.splitlines()[-1].startswith("No restrictions"),
        "plan": plan,
        "ms": round(elapsed_ms, 1),
        "results": rows,
    }


def make_handler(index: Index):
    info = {
        "n_docs": index.n_docs,
        "default_variant": DEFAULT_VARIANT,
        "variants": [{"id": v, "label": VARIANT_LABELS.get(v, v)} for v in VARIANTS],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/info":
                self._json(200, info)
            elif url.path == "/api/search":
                self._search(parse_qs(url.query))
            else:
                self._json(404, {"error": "Not found"})

        def _search(self, params):
            query = params.get("q", [""])[0].strip()
            variant = params.get("variant", [DEFAULT_VARIANT])[0]
            try:
                k = max(1, min(MAX_K, int(params.get("k", ["10"])[0])))
            except ValueError:
                k = 10
            if not query:
                return self._json(400, {"error": "Type a query to search."})
            if variant not in VARIANTS:
                return self._json(400, {"error": f"Unknown variant {variant!r}."})
            try:
                self._json(200, run_query(index, query, variant, k))
            except QuerySyntaxError as exc:
                self._json(400, {"error": str(exc)})

        def _json(self, status, payload):
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            if "/api/search" in self.path:
                sys.stderr.write(f"{self.command} {self.path}\n")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the PRISM web front end.")
    parser.add_argument("--index", type=Path, default=INDEX_DIR, help="index directory or .pkl file")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading index from {args.index} ...", flush=True)
    try:
        index = Index.load(args.index)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print("Build one with scripts/build_index.py, or point at an existing one with "
              "--index, e.g. --index index/sub30k", file=sys.stderr)
        return 2
    set_index(index)
    print(f"Loaded {index.n_docs:,} papers. Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(index))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
