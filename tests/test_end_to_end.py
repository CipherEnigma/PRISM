"""End to end on a handful of hand-written papers: real analyzer, indexer, parser, search() and the eval harness.

Each query has an obvious best paper, worked out by reading the corpus below.
"""
import pytest

from eval.runner import read_per_topic_csv, read_run_file, run_and_save, run_variant
from prism import search as search_module
from prism.config import VARIANTS
from prism.index import Index
from prism.indexer import build_index
from prism.schema import Record

DOCS = [
    Record("p1", "Remdesivir trial in covid patients",
           "A randomized remdesivir trial reduced recovery time in hospitalized covid patients.",
           year=2020, publish_month="2020-05"),
    Record("p2", "Mask wearing and viral transmission",
           "Community mask wearing reduced transmission of the respiratory virus.",
           year=2020, publish_month="2020-04"),
    Record("p3", "Vaccine efficacy against covid",
           "The mrna vaccine showed high efficacy in a large phase three trial.",
           year=2021, publish_month="2021-03"),
    Record("p4", "Digital tools for outbreak control",
           "Mobile contact tracing apps were used for covid outbreak control.",
           year=2020, publish_month="2020-06"),
    Record("p5", "Hydroxychloroquine in covid",
           "A trial of hydroxychloroquine showed no benefit in covid patients.",
           year=2020, publish_month="2020-07"),
    Record("p6", "Contact patterns in schools",
           "Students make contact in crowded classrooms, and tracing of cases was slow.",
           year=2021, publish_month="2021-01"),
    Record("p7", "Sleep and stress in healthcare workers",
           "A survey of nurses reported poor sleep and high stress.",
           year=2019, publish_month="2019-11"),
    Record("p8", "Ventilator allocation in intensive care",
           "Guidelines for allocating ventilators during a surge.",
           year=2020, publish_month="2020-03"),
]
QUERIES = {
    "1": "remdesivir trial",                 # p1 matches in both zones
    "2": "mask transmission",
    "3": "vaccine efficacy",
    "4": '"contact tracing"',                # phrase: p4 only, p6 has both words but not adjacent
    "5": "title:hydroxychloroquine",
    "6": "covid year>=2021",                 # filter keeps only p3
}
EXPECTED_TOP = {"1": "p1", "2": "p2", "3": "p3", "4": "p4", "5": "p5", "6": "p3"}
QRELS = {q: {EXPECTED_TOP[q]: 2} for q in QUERIES}


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    out = tmp_path_factory.mktemp("index")
    build_index(DOCS, out)
    return Index.load(out)


@pytest.fixture(autouse=True)
def use_tiny_index(index, monkeypatch):
    monkeypatch.setattr(search_module, "_INDEX", index)


@pytest.mark.parametrize("variant", ["V0", "V1", "V2", "R"])
def test_every_query_finds_its_best_paper_first(variant):
    for qid, query in QUERIES.items():
        hits = search_module.search(query, k=5, variant=variant)
        assert hits, f"{variant} returned nothing for {query!r}"
        assert hits[0].doc_id == EXPECTED_TOP[qid], f"{variant}: {query!r}"


def test_phrase_excludes_the_non_adjacent_paper():
    ids = [h.doc_id for h in search_module.search('"contact tracing"', k=5, variant="V0")]
    assert ids == ["p4"]


def test_filter_restricts_to_matching_years():
    ids = [h.doc_id for h in search_module.search("covid year>=2021", k=5, variant="V0")]
    assert ids == ["p3"]


def test_results_are_deterministic_including_ties():
    for variant in VARIANTS:
        a = [(h.doc_id, h.score) for h in search_module.search("covid trial", k=8, variant=variant)]
        b = [(h.doc_id, h.score) for h in search_module.search("covid trial", k=8, variant=variant)]
        assert a == b, variant


def test_authority_never_adds_a_non_matching_paper():
    ids = {h.doc_id for h in search_module.search("remdesivir", k=8, variant="V2")}
    assert ids == {"p1"}


def test_harness_runs_the_real_engine_and_scores_it(tmp_path):
    s = run_and_save("V2", QUERIES, QRELS, k=100, runs_dir=tmp_path / "runs", results_dir=tmp_path / "res")
    per = read_per_topic_csv(tmp_path / "res" / "per_topic_V2.csv")
    assert all(per[q]["nDCG@10"] == pytest.approx(1.0) for q in QUERIES)
    assert s["mean"]["Recall@100"] == pytest.approx(1.0)
    run = read_run_file(tmp_path / "runs" / "V2.run")
    assert run["4"][0][0] == "p4" and set(run) == set(QUERIES)
    # an identical second run gives identical rankings and scores (scores are saved to 8 decimals)
    again = run_variant("V2", QUERIES, k=100)
    assert again.ranked() == {q: [d for d, _ in v] for q, v in run.items()}
    for q, hits in again.run.items():
        assert [round(s, 8) for _, s in hits] == [s for _, s in run[q]]


def test_champion_fn_keeps_the_best_documents_by_tf_over_length(index):
    from eval.tune import make_champion_fn
    # "covid" is in p1, p3, p4, p5 only through the title/abstract; keep the top 2 by tf / length
    docs, tfs = index.postings("covid", "all")
    assert len(docs) >= 4
    champ = make_champion_fn(index, 2)("covid", "all")
    lengths = index.lengths("all")[docs]
    ratio = tfs / lengths
    best = set(docs[sorted(range(len(docs)), key=lambda i: (-ratio[i], docs[i]))[:2]].tolist())
    assert champ.tolist() == sorted(best)
    assert make_champion_fn(index, 100)("covid", "all") is None      # short lists use full postings
    assert make_champion_fn(index, 2)("zzzunseen", "all") is None


@pytest.mark.parametrize("variant", ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"])
def test_explainer_net_score_equals_the_search_score(index, variant):
    """The explainer must reproduce every result's score (spec: explainer consistency)."""
    import re
    from prism.explain import explain
    from prism.parser import parse
    for query in ("covid trial", '"contact tracing"', "title:remdesivir year>=2020", "vaccine efficacy"):
        pq = parse(query)
        for hit in search_module.search(query, k=5, variant=variant):
            text = explain(hit, pq, index, variant=variant)
            net, found = re.search(r"Net score=(-?[\d.]+); search score=(-?[\d.]+)", text).groups()
            assert float(net) == pytest.approx(float(found), abs=1e-7), (variant, query, hit.doc_id)


def test_v7_alpha_reaches_the_zone_weights_and_the_explainer(index, monkeypatch):
    """alpha = 0 keeps the global zone weights; a larger alpha moves them; search and explain agree."""
    import re
    from dataclasses import replace
    from prism.explain import explain
    from prism.parser import parse
    pq = parse("remdesivir trial")
    weights = {}
    for alpha in (0.0, 1.0):
        monkeypatch.setitem(VARIANTS, f"A{int(alpha)}", replace(VARIANTS["V7"], alpha=alpha))
        weights[alpha] = search_module._components(pq, index, VARIANTS[f"A{int(alpha)}"])[4]
        assert sum(weights[alpha].values()) == pytest.approx(1.0)
        for hit in search_module.search("remdesivir trial", k=3, variant=f"A{int(alpha)}"):
            text = explain(hit, pq, index, variant=f"A{int(alpha)}")
            net, found = re.search(r"Net score=(-?[\d.]+); search score=(-?[\d.]+)", text).groups()
            assert float(net) == pytest.approx(float(found), abs=1e-7)
    assert weights[0.0] == pytest.approx(VARIANTS["V7"].zone_weights)
    assert weights[1.0] != pytest.approx(weights[0.0])
