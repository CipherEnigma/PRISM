"""OpenAlex citation fetching with a fake network: batching, matching, caching, resume, backoff."""
import json
import urllib.error

import pytest

from prism.citations import (
    BATCH, BudgetExhausted, build_filter, fetch_citations, load_cache, normalize_doi, normalize_pmid,
    parse_works, plan,
)
from prism.schema import Record


def _recs(n, with_doi=True):
    return [Record(f"p{i}", f"t{i}", "a", doi=f"10.1/{i}" if with_doi else None, pubmed_id=str(1000 + i)) for i in range(n)]


class FakeApi:
    """Knows a count for some DOIs and PMIDs; records every filter it receives."""

    def __init__(self, dois=None, pmids=None):
        self.dois, self.pmids, self.calls = dois or {}, pmids or {}, []

    def __call__(self, filter_expr, api_key):
        self.calls.append(filter_expr)
        kind, _, rest = filter_expr.partition(":")
        results = []
        for v in rest.split("|"):
            if kind == "doi" and v.removeprefix("https://doi.org/") in self.dois:
                d = v.removeprefix("https://doi.org/")
                results.append({"doi": f"https://doi.org/{d.upper()}", "cited_by_count": self.dois[d], "ids": {}})
            if kind == "pmid" and v in self.pmids:
                results.append({"doi": None, "cited_by_count": self.pmids[v],
                                "ids": {"pmid": f"https://pubmed.ncbi.nlm.nih.gov/{v}"}})
        return {"results": results}


def test_identifier_normalization():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("doi: 10.1/x") == "10.1/x" and normalize_doi(" 10.1/x ") == "10.1/x"
    assert normalize_doi("not a doi") is None and normalize_doi("") is None and normalize_doi(None) is None
    assert normalize_pmid("https://pubmed.ncbi.nlm.nih.gov/123/") == "123" and normalize_pmid("456") == "456"
    assert normalize_pmid("PMC123") is None and normalize_pmid(None) is None


def test_filter_expressions():
    assert build_filter("doi", ["10.1/a", "10.1/b"]) == "doi:https://doi.org/10.1/a|https://doi.org/10.1/b"
    assert build_filter("pmid", ["1", "2"]) == "pmid:1|2"
    with pytest.raises(ValueError):
        build_filter("isbn", ["1"])


def test_parse_works_matches_case_insensitively():
    found = parse_works({"results": [{"doi": "https://doi.org/10.1/ABC", "cited_by_count": 7,
                                      "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/42"}},
                                     {"doi": None, "cited_by_count": None, "ids": {}}]})
    assert found == {("doi", "10.1/abc"): 7, ("pmid", "42"): 7}


def test_batches_never_exceed_100_and_everything_is_found(tmp_path):
    recs = _recs(250)
    api = FakeApi(dois={f"10.1/{i}": i for i in range(250)})
    stats = fetch_citations(recs, tmp_path / "c.jsonl", fetch_fn=api, sleep=lambda s: None)
    assert [c.count("|") + 1 for c in api.calls] == [100, 100, 50] and stats["requests"] == 3
    cache = load_cache(tmp_path / "c.jsonl")
    assert len(cache) == 250 and cache["p7"]["citations"] == 7 and cache["p7"]["matched_by"] == "doi"
    assert stats["found_by_doi"] == 250 and stats["not_found"] == 0


def test_pubmed_id_is_the_fallback_and_misses_are_cached(tmp_path):
    recs = [Record("a", "t", "x", doi="10.1/a", pubmed_id="11"), Record("b", "t", "x", doi="10.1/zzz", pubmed_id="22"),
            Record("c", "t", "x", pubmed_id="33"), Record("d", "t", "x")]
    api = FakeApi(dois={"10.1/a": 5}, pmids={"22": 9})
    stats = fetch_citations(recs, tmp_path / "c.jsonl", fetch_fn=api, sleep=lambda s: None)
    cache = load_cache(tmp_path / "c.jsonl")
    assert (cache["a"]["citations"], cache["a"]["matched_by"]) == (5, "doi")
    assert (cache["b"]["citations"], cache["b"]["matched_by"]) == (9, "pmid")
    assert cache["c"]["citations"] is None and cache["d"]["citations"] is None      # looked up, nothing found
    assert stats["found_by_doi"] == 1 and stats["found_by_pmid"] == 1 and stats["not_found"] == 2
    assert "pmid:22|33" in api.calls


def test_shared_doi_gives_every_paper_the_count(tmp_path):
    recs = [Record("a", "t", "x", doi="10.1/same"), Record("b", "t", "x", doi="10.1/SAME")]
    api = FakeApi(dois={"10.1/same": 3})
    fetch_citations(recs, tmp_path / "c.jsonl", fetch_fn=api, sleep=lambda s: None)
    assert len(api.calls) == 1 and api.calls[0].count("|") == 0
    assert {k: v["citations"] for k, v in load_cache(tmp_path / "c.jsonl").items()} == {"a": 3, "b": 3}


def test_resume_makes_no_repeat_calls(tmp_path):
    recs = _recs(30)
    api = FakeApi(dois={f"10.1/{i}": i for i in range(30)})
    fetch_citations(recs[:10], tmp_path / "c.jsonl", fetch_fn=api, sleep=lambda s: None)
    first_calls = len(api.calls)
    stats = fetch_citations(recs, tmp_path / "c.jsonl", fetch_fn=api, sleep=lambda s: None)
    assert stats["already_cached"] == 10 and len(api.calls) == first_calls + 1      # only the 20 new papers
    again = fetch_citations(recs, tmp_path / "c.jsonl", fetch_fn=api, sleep=lambda s: None)
    assert again["requests"] == 0 and len(api.calls) == first_calls + 1


def test_a_torn_last_line_is_ignored(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"doc_id": "a", "citations": 1, "matched_by": "doi"}) + "\n" + '{"doc_id": "b", "citat')
    assert set(load_cache(p)) == {"a"}


def test_limit_and_plan(tmp_path):
    recs = _recs(250)
    assert plan(recs, {})["doi_requests"] == 3 and plan(recs, {})["distinct_dois"] == 250
    assert plan(recs, {r.doc_id: {} for r in recs[:200]})["papers_to_look_up"] == 50
    api = FakeApi(dois={f"10.1/{i}": i for i in range(250)})
    fetch_citations(recs, tmp_path / "c.jsonl", fetch_fn=api, sleep=lambda s: None, limit=120)
    assert len(load_cache(tmp_path / "c.jsonl")) == 120


def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


def test_backoff_then_success(tmp_path):
    state = {"n": 0}
    api = FakeApi(dois={"10.1/0": 4})

    def flaky(f, k):
        state["n"] += 1
        if state["n"] < 3:
            raise _http_error(429)
        return api(f, k)

    sleeps = []
    fetch_citations(_recs(1), tmp_path / "c.jsonl", fetch_fn=flaky, sleep=sleeps.append)
    assert load_cache(tmp_path / "c.jsonl")["p0"]["citations"] == 4
    assert sleeps[:2] == [2.0, 4.0]                                  # exponential backoff


def test_budget_exhausted_keeps_what_was_fetched(tmp_path):
    api = FakeApi(dois={f"10.1/{i}": i for i in range(150)})
    state = {"n": 0}

    def second_batch_blocked(f, k):
        state["n"] += 1
        if state["n"] >= 2:
            raise _http_error(429)
        return api(f, k)

    with pytest.raises(BudgetExhausted):
        fetch_citations(_recs(150), tmp_path / "c.jsonl", fetch_fn=second_batch_blocked, sleep=lambda s: None)
    assert len(load_cache(tmp_path / "c.jsonl")) == BATCH               # the first batch was saved
    resumed = fetch_citations(_recs(150), tmp_path / "c.jsonl", fetch_fn=api, sleep=lambda s: None)
    assert resumed["already_cached"] == BATCH and len(load_cache(tmp_path / "c.jsonl")) == 150


def test_other_http_errors_are_not_retried(tmp_path):
    def broken(f, k):
        raise _http_error(400)

    with pytest.raises(urllib.error.HTTPError):
        fetch_citations(_recs(1), tmp_path / "c.jsonl", fetch_fn=broken, sleep=lambda s: None)


def test_cache_feeds_the_index_patch(tmp_path):
    from eval.index_patch import load_citations
    fetch_citations(_recs(3), tmp_path / "c.jsonl", fetch_fn=FakeApi(dois={"10.1/1": 8}), sleep=lambda s: None)
    assert load_citations(tmp_path / "c.jsonl") == {"p1": 8}          # not-found papers carry no count
