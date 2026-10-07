"""A: python scripts/build_index.py --out index/ [--subset N]  (subset keeps all judged docs + seeded sample)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from prism.indexer import build_index
from prism.loader import iter_records


def _find_qrel_ids(data_dir: Path) -> set[str]:
    """Judged doc ids from every qrels file under data_dir.

    Accepts BEIR's qrels/<split>.tsv ("query-id corpus-id score", with a header row) and
    TREC-style files named qrels* ("qid iteration docid rel").
    """
    qrel_paths = {p for p in data_dir.rglob("*") if p.is_file()
                  and (p.parent.name == "qrels" or p.name.lower().startswith("qrels"))}
    seen: set[str] = set()
    for qrel_path in sorted(qrel_paths):
        with qrel_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                pieces = line.strip().split()
                if len(pieces) == 3 and pieces[2].lstrip("-").isdigit():
                    seen.add(pieces[1])          # BEIR: query-id corpus-id score
                elif len(pieces) == 4 and pieces[3].lstrip("-").isdigit():
                    seen.add(pieces[2])          # TREC: qid iteration docid rel
    return seen


def _subset_records(records, limit: int | None):
    if limit is None or limit <= 0:
        return list(records)
    entries = list(records)
    if not entries:
        return []
    qrel_ids = _find_qrel_ids(Path(__file__).resolve().parents[1] / "data")
    keep = set(qrel_ids)
    remaining = [record for record in entries if record.doc_id not in keep]
    rng = random.Random(42)
    if len(keep) < limit:
        sample_size = min(limit - len(keep), len(remaining))
        keep.update(record.doc_id for record in rng.sample(remaining, sample_size))
    return [record for record in entries if record.doc_id in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PRISM inverted index.")
    parser.add_argument("--out", type=Path, required=True, help="Directory to write the index to.")
    parser.add_argument("--subset", type=int, default=None, help="Optional subset size; includes all qrel docs plus a seeded sample.")
    args = parser.parse_args()

    records = list(iter_records())
    if args.subset is not None:
        records = _subset_records(records, args.subset)
    build_index(records, args.out)
    print(f"Built index for {len(records)} records at {args.out}")


if __name__ == "__main__":
    main()
