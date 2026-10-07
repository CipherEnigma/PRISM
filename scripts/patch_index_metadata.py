"""D: recompute an index's metadata fields and authority arrays from the right metadata.csv, without a rebuild.

    python scripts/patch_index_metadata.py --src Downloads/index/index/index.pkl --out index/
    python scripts/patch_index_metadata.py --src index/ --out index/ --citations data/citations.jsonl

Use when an index was built from a wrong or partial metadata.csv, or to add citation counts later.
Only numpy arrays and plain Python types are read from --src, so the file cannot run code while loading.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.index_patch import (                                         # noqa: E402
    coverage, load_citations, load_payload, metadata_arrays, patch_payload, write_payload,
)
from prism.loader import iter_records                                   # noqa: E402


def _show(label: str, cov: dict) -> None:
    print(f"  {label:<7} " + "  ".join(f"{k} {v:.1%}" for k, v in cov.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch the metadata part of a PRISM index.")
    ap.add_argument("--src", type=Path, required=True, help="index.pkl or the folder that holds it")
    ap.add_argument("--out", type=Path, required=True, help="folder (or .pkl path) to write the patched index to")
    ap.add_argument("--data-dir", type=Path, default=None, help="folder with corpus.jsonl and metadata.csv (default: data/)")
    ap.add_argument("--citations", type=Path, default=None, help="optional JSONL of citation counts per paper")
    args = ap.parse_args()

    start = time.perf_counter()
    payload = load_payload(args.src)
    print(f"loaded {payload['n_docs']:,} papers from {args.src} in {time.perf_counter() - start:.1f}s")
    records = list(iter_records(args.data_dir))
    citations = load_citations(args.citations) if args.citations else None
    patched = patch_payload(payload, records, citations)
    print("metadata coverage:")
    _show("before", coverage({k: payload[k] for k in ("years", "journals", "sources", "publish_months",
                                                      "pubmed_ids", "dois", "authority_raw", "authority_cohort")}))
    _show("after", coverage(patched))
    if citations is not None:
        print(f"  citation counts for {sum(1 for r in records if r.doc_id in citations):,} of {len(records):,} papers")
    target = write_payload(patched, args.out)
    print(f"wrote {target} ({target.stat().st_size / 2**20:.0f} MB) in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
