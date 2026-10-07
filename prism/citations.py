"""Citation counts from OpenAlex, cached to disk (authority source for N1 and N2).

Papers are looked up in batches of up to 100 by DOI, then by PubMed ID for those not found. Every result is
appended to a JSONL cache, one line per paper, so a run can be stopped and resumed and no call is repeated:
    {"doc_id": "ug7v899j", "citations": 12, "matched_by": "doi"}      found
    {"doc_id": "abc12345", "citations": null, "matched_by": null}      looked up, not found or no identifier
The cache is read by eval.index_patch.load_citations (`--citations` of scripts/patch_index_metadata.py).
The network call is a parameter (`fetch_fn`), so tests never touch the network.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from prism.schema import Record

API_URL = "https://api.openalex.org/works"
BATCH = 100                                       # OpenAlex allows up to 100 OR values per filter
FetchFn = Callable[[str, str], Mapping]           # (filter expression, api_key) -> parsed JSON response
SleepFn = Callable[[float], None]


class BudgetExhausted(RuntimeError):
    """OpenAlex kept answering 429 (daily budget used up). The cache is saved; resume later."""


def normalize_doi(value: str | None) -> str | None:
    """Lower-case bare DOI ("10.1/abc"), or None if the value is not a DOI."""
    if not value:
        return None
    v = value.strip().lower()
    v = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", v)
    return v if v.startswith("10.") and " " not in v else None


def normalize_pmid(value: str | None) -> str | None:
    """Digits only, or None."""
    if not value:
        return None
    v = re.sub(r"^https?://pubmed\.ncbi\.nlm\.nih\.gov/", "", value.strip().rstrip("/"))
    return v if v.isdigit() else None


def load_cache(path: str | Path) -> dict[str, dict]:
    """doc_id -> cached line. Missing file means an empty cache; a half-written last line is ignored."""
    path = Path(path)
    cache: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[str(obj["doc_id"])] = obj
    return cache


def _append(path: Path, lines: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for obj in lines:
            handle.write(json.dumps(obj) + "\n")


def build_filter(kind: str, keys: Sequence[str]) -> str:
    """OpenAlex filter for a batch: `doi:https://doi.org/10.x|...` or `pmid:123|...`."""
    if kind == "doi":
        return "doi:" + "|".join(f"https://doi.org/{k}" for k in keys)
    if kind == "pmid":
        return "pmid:" + "|".join(keys)
    raise ValueError(f"unknown identifier kind {kind!r}")


def parse_works(response: Mapping) -> dict[tuple[str, str], int]:
    """{("doi", key): count, ("pmid", key): count} for every work in a response."""
    found: dict[tuple[str, str], int] = {}
    for work in response.get("results", []):
        count = work.get("cited_by_count")
        if count is None:
            continue
        doi = normalize_doi(work.get("doi") or (work.get("ids") or {}).get("doi"))
        pmid = normalize_pmid((work.get("ids") or {}).get("pmid"))
        if doi:
            found[("doi", doi)] = int(count)
        if pmid:
            found[("pmid", pmid)] = int(count)
    return found


def http_fetch(filter_expr: str, api_key: str = "") -> Mapping:
    """One real call (stdlib only). Raises urllib.error.HTTPError on failure."""
    params = {"filter": filter_expr, "per_page": str(BATCH), "select": "id,doi,ids,cited_by_count"}
    if api_key:
        params["api_key"] = api_key
    url = f"{API_URL}?{urllib.parse.urlencode(params, safe=':/|,')}"
    request = urllib.request.Request(url, headers={"User-Agent": "PRISM-course-project (student IR project)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _call(fetch_fn: FetchFn, filter_expr: str, api_key: str, sleep: SleepFn, retries: int = 5) -> Mapping:
    """fetch_fn with exponential backoff on 429 and 5xx; BudgetExhausted when 429 will not clear."""
    delay = 2.0
    for attempt in range(retries):
        try:
            return fetch_fn(filter_expr, api_key)
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == retries - 1:
                if err.code == 429:
                    raise BudgetExhausted("OpenAlex returned 429 repeatedly: the daily budget is probably used up") from err
                raise
            sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def plan(records: Sequence[Record], cache: Mapping[str, dict]) -> dict[str, int]:
    """What a run would do, without any network: papers to look up and requests needed."""
    todo = [r for r in records if r.doc_id not in cache]
    dois = {normalize_doi(r.doi) for r in todo} - {None}
    pmids = {normalize_pmid(r.pubmed_id) for r in todo if not normalize_doi(r.doi)} - {None}
    return {"papers_to_look_up": len(todo), "distinct_dois": len(dois), "distinct_pmids_without_doi": len(pmids),
            "doi_requests": -(-len(dois) // BATCH), "max_pmid_requests": -(-len(pmids) // BATCH) + -(-len(dois) // BATCH)}


def fetch_citations(records: Sequence[Record], cache_path: str | Path, fetch_fn: FetchFn = http_fetch,
                    api_key: str = "", pause: float = 0.1, sleep: SleepFn = time.sleep,
                    limit: int | None = None) -> dict[str, int]:
    """Look up every paper not already in the cache; returns counts of requests and outcomes.

    Pass 1 batches papers by DOI. Papers not found there (or without a DOI) that have a PubMed ID are
    batched in pass 2. Results are appended to the cache after every request."""
    cache_path = Path(cache_path)
    cache = load_cache(cache_path)
    todo = [r for r in records if r.doc_id not in cache]
    if limit is not None:
        todo = todo[:limit]
    stats = {"requests": 0, "found_by_doi": 0, "found_by_pmid": 0, "not_found": 0, "already_cached": len(records) - len(todo)}
    resolved: set[str] = set()

    def run_pass(kind: str, key_of: Callable[[Record], str | None], pool: Sequence[Record]) -> None:
        by_key: dict[str, list[Record]] = {}
        for r in pool:
            key = key_of(r)
            if key:
                by_key.setdefault(key, []).append(r)
        keys = sorted(by_key)
        for i in range(0, len(keys), BATCH):
            batch = keys[i:i + BATCH]
            found = parse_works(_call(fetch_fn, build_filter(kind, batch), api_key, sleep))
            stats["requests"] += 1
            lines = []
            for key in batch:
                if (kind, key) in found:
                    for r in by_key[key]:
                        lines.append({"doc_id": r.doc_id, "citations": found[(kind, key)], "matched_by": kind})
                        resolved.add(r.doc_id)
                        stats[f"found_by_{kind}"] += 1
            _append(cache_path, lines)
            sleep(pause)

    run_pass("doi", lambda r: normalize_doi(r.doi), todo)
    leftover = [r for r in todo if r.doc_id not in resolved]
    run_pass("pmid", lambda r: normalize_pmid(r.pubmed_id), leftover)
    misses = [r for r in todo if r.doc_id not in resolved]
    _append(cache_path, ({"doc_id": r.doc_id, "citations": None, "matched_by": None} for r in misses))
    stats["not_found"] = len(misses)
    return stats
