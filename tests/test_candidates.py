"""B3: candidate sets for example queries on the five-document corpus, checked by hand.

The corpus (analyzed tokens; the all zone is title then abstract):
  d0  contact trace app        | mobile contact trace app reduce spread covid19 | 2020 lancet pmc
  d1  trace contact network    | contact network trace mobile phone data        | 2021 nature medline
  d2  remdesivir trial         | remdesivir random control trial hospit patient | 2020 nejm   pmc
  d3  hydroxychloroquin trial  | trial hydroxychloroquin remdesivir compar ...  | 2021 lancet medline
  d4  vaccin efficaci          | vaccin efficaci trace trace mobile contact     | none bmj    arxiv
"""
import pytest

import prism.parser
from prism.candidates import candidates, describe
from prism.parser import parse
from tests.fake_analyzer import fake_analyze
from tests.fake_index import FIVE_DOCS, FakeIndex


@pytest.fixture(autouse=True)
def _use_fake_analyzer(monkeypatch):
    monkeypatch.setattr(prism.parser, "analyze", fake_analyze)


@pytest.fixture(scope="module")
def index():
    return FakeIndex(FIVE_DOCS)


def run(query, index, **kwargs):
    result = candidates(parse(query), index, **kwargs)
    return None if result is None else result.tolist()


@pytest.mark.parametrize("query, expected", [
    # The spec's table.
    ("vaccine efficacy", None),                               # bare words restrict nothing
    ("vaccine efficacy year>=2020", [0, 1, 2, 3]),            # d4 has no year
    ('"contact tracing" AND mobile', [0]),                    # phrase only in d0
    ("remdesivir NEAR/5 trial", [2, 3]),
    ("title:remdesivir NOT hydroxychloroquine", [2]),
    # NOT with only bare words beside it: everything except.
    ("remdesivir NOT hydroxychloroquine", [0, 1, 2, 4]),
    ("NOT trial", [0, 1, 4]),
    ("NOT NOT trial", [2, 3]),
    # AND / OR and brackets.
    ("trial OR mobile", [0, 1, 2, 3, 4]),
    ("trial AND mobile", []),
    ("(trial OR mobile) AND year>=2021", [1, 3]),
    ("(hydroxychloroquine vaccine) AND trial", [3]),          # bracketed bare words: any of them
    ("mobile OR NOT trial", [0, 1, 4]),
    # Filters and zones.
    ('journal:"lancet"', [0, 3]),
    ("year>=2020 year<2021", [0, 2]),
    ('year=2020 AND NOT journal:"nejm"', [0]),
    ("source:pmc NOT title:trial", [0]),
    ('title:"contact tracing"', [0]),
    ('title:"tracing contact"', [1]),                         # order matters
    # NEAR in the all zone: d1 has "trace contact" at 0, 1; d4 has a gap of 2.
    ("contact NEAR/1 trace", [0, 1]),
    # Stop words and unseen words.
    ("trial vaccine AND the", [4]),                           # vaccine still restricts
    ("zebra", None),
    ("zebra AND trial", []),
    ("long-covid", []),                                       # becomes a phrase
    ("the", None),
])
def test_candidate_sets(index, query, expected):
    assert run(query, index) == expected


def test_champions_only_apply_without_a_restriction():
    # champion_r=1: trace has tf 2 in d0, d1 and d4 (tie -> lowest id d0); mobile tf 1 in each.
    index = FakeIndex(FIVE_DOCS, champion_r=1)
    assert run("vaccine efficacy", index, use_champions=True) == [4]
    assert run("trace mobile", index, use_champions=True) == [0]
    assert run("trace mobile", index) is None
    assert run("trace year>=2021", index, use_champions=True) == [1, 3]
    assert run("the", index, use_champions=True) is None


def test_champions_fall_back_to_full_postings(index):
    # index has no champion lists built (champion() returns None).
    assert run("trace mobile", index, use_champions=True) == [0, 1, 4]


def test_results_are_sorted_int32_arrays(index):
    for query in ["NOT trial", "trial OR mobile", 'journal:"lancet"', "remdesivir NEAR/5 trial"]:
        result = candidates(parse(query), index)
        assert result.dtype.name == "int32"
        assert (result[1:] > result[:-1]).all()


# --- B4: describe() ---

DESCRIBE_QUERIES = [
    "vaccine efficacy year>=2020", '"contact tracing" AND mobile', "remdesivir NEAR/5 trial",
    "title:remdesivir NOT hydroxychloroquine", "NOT trial", "(trial OR mobile) AND year>=2021",
    "(hydroxychloroquine vaccine) AND trial", "mobile OR NOT trial", "zebra AND trial",
    '(trace AND mobile) OR NOT (trial AND year>=2020)', "trial vaccine AND the",
]


@pytest.mark.parametrize("query", DESCRIBE_QUERIES)
def test_describe_reports_the_same_count_as_candidates(index, query):
    n = len(candidates(parse(query), index))
    last = describe(parse(query), index).splitlines()[-1]
    assert last == f"Result: {n:,} doc{'' if n == 1 else 's'} in the candidate set"


def test_describe_exact_output(index):
    assert describe(parse('"contact tracing" AND mobile year>=2020'), index) == "\n".join([
        'Query: "contact tracing" AND mobile year>=2020',
        "Ranking terms: contact, trace, mobile",
        'Phrases: "contact trace"',
        "AND, shortest list first -> 1 doc",
        '  start  phrase "contact trace"  1 doc',
        "  AND    mobile                  df 3    -> 1 doc",
        "  AND    year>=2020              4 docs  -> 1 doc",
        "Result: 1 doc in the candidate set",
    ])


def test_describe_stops_early_and_says_so(index):
    text = describe(parse("trial AND zebra AND mobile"), index)
    assert "start  zebra   df 0" in text                  # rarest operand goes first
    assert text.count("skipped (already empty)") == 2


def test_describe_unrestricted_and_champions(index):
    assert "No restrictions" in describe(parse("vaccine efficacy"), index)
    text = describe(parse("trace mobile"), FakeIndex(FIVE_DOCS, champion_r=1), use_champions=True)
    assert "Champion lists, OR of each term -> 1 doc" in text
    assert "trace   1 doc (champion list)" in text
