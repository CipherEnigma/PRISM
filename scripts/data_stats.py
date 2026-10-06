"""A: doc count, empty-abstract share, share with year, share with DOI/PubMed ID, year histogram."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from prism.loader import iter_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Print dataset statistics for the PRISM corpus.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional dataset directory override.")
    args = parser.parse_args()

    records = list(iter_records(data_dir=args.data_dir))
    total = len(records)
    empty_abstract = sum(1 for record in records if not (record.abstract or "").strip())
    with_year = sum(1 for record in records if record.year is not None)
    with_doc_meta = sum(1 for record in records if (record.doi or record.pubmed_id))
    year_hist = Counter(record.year for record in records if record.year is not None)

    print(f"document_count={total}")
    print(f"empty_abstract_share={empty_abstract / total if total else 0.0:.6f}")
    print(f"with_year_share={with_year / total if total else 0.0:.6f}")
    print(f"with_doi_or_pubmed_share={with_doc_meta / total if total else 0.0:.6f}")
    print("year_histogram")
    for year in sorted(year_hist):
        print(f"{year}: {year_hist[year]}")


if __name__ == "__main__":
    main()
