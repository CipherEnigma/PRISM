"""B2: parser output (ranking terms, phrases, AST) and syntax errors."""
import pytest

import prism.parser
from prism.parser import (
    And, Filter, Near, Not, Or, Phrase, QuerySyntaxError, Seq, Term, parse,
)
from tests.fake_analyzer import fake_analyze


@pytest.fixture(autouse=True)
def _use_fake_analyzer(monkeypatch):
    monkeypatch.setattr(prism.parser, "analyze", fake_analyze)


# The spec's table, row by row.
@pytest.mark.parametrize("query, terms, ast", [
    ("vaccine efficacy",
     ["vaccin", "efficaci"],
     Seq((Term("vaccin"), Term("efficaci")))),
    ("vaccine efficacy year>=2020",
     ["vaccin", "efficaci"],
     Seq((Term("vaccin"), Term("efficaci"), Filter("year", ">=", 2020)))),
    ('"contact tracing" AND mobile',
     ["contact", "trace", "mobile"],
     Seq((And((Phrase(("contact", "trace")), Term("mobile"))),))),
    ("remdesivir NEAR/5 trial",
     ["remdesivir", "trial"],
     Seq((Near("remdesivir", "trial", 5),))),
    ("title:remdesivir NOT hydroxychloroquine",
     ["remdesivir"],
     Seq((Term("remdesivir", "title"), Not(Term("hydroxychloroquin"))))),
])
def test_spec_table(query, terms, ast):
    pq = parse(query)
    assert pq.raw == query
    assert pq.terms == terms
    assert pq.ast == ast


@pytest.mark.parametrize("query, ast", [
    # Filters: ":" means "=", quoted values are taken as typed, year becomes an int.
    ('journal:"lancet"', Seq((Filter("journal", "=", "lancet"),))),
    ("source:pmc", Seq((Filter("source", "=", "pmc"),))),
    ("year<2021", Seq((Filter("year", "<", 2021),))),
    ("publish_month>=2020-03", Seq((Filter("publish_month", ">=", "2020-03"),))),
    # Zones, with or without a space, on a word or a phrase.
    ('title:"contact tracing"', Seq((Phrase(("contact", "trace"), "title"),))),
    ("abstract: remdesivir", Seq((Term("remdesivir", "abstract"),))),
    # Precedence: NOT binds tightest, then AND, then OR, then juxtaposition.
    ("trial OR mobile AND phone", Seq((Or((Term("trial"), And((Term("mobile"), Term("phone"))))),))),
    ("NOT trial AND mobile", Seq((And((Not(Term("trial")), Term("mobile"))),))),
    ("(trial OR mobile) AND phone", Seq((And((Seq((Or((Term("trial"), Term("mobile"))),)),
                                               Term("phone"))),))),
    # Stop words disappear; an AND that loses an operand still restricts.
    ("the vaccine", Seq((Term("vaccin"),))),
    ("trial vaccine AND the", Seq((Term("trial"), And((Term("vaccin"),))))),
    # Lowercase "and" is just a (stop) word.
    ("vaccine and trial", Seq((Term("vaccin"), Term("trial")))),
    # A word that analyzes to several tokens becomes a phrase; covid-19 stays one token.
    ("long-covid", Seq((Phrase(("long", "covid")),))),
    ("COVID-19 SARS-CoV-2", Seq((Term("covid19"), Term("sarscov2")))),
    ("", Seq(())),
])
def test_ast(query, ast):
    assert parse(query).ast == ast


def test_terms_dedupe_in_order_and_skip_not():
    pq = parse('trial "remdesivir trial" NOT mobile NOT "contact tracing"')
    assert pq.terms == ["trial", "remdesivir"]
    assert pq.phrases == [["remdesivir", "trial"]]


def test_phrases_and_near_terms_count_for_ranking():
    pq = parse('"contact tracing app" OR remdesivir NEAR/3 trial')
    assert pq.terms == ["contact", "trace", "app", "remdesivir", "trial"]
    assert pq.phrases == [["contact", "trace", "app"]]


@pytest.mark.parametrize("query, message", [
    ('"contact tracing', "Unbalanced quote"),
    ('trial "a" "b', "Unbalanced quote"),
    ("(vaccine AND trial", "Unbalanced parenthesis"),
    ("vaccine)", "Unbalanced parenthesis"),
    ("remdesivir NEAR/5", "missing the word after"),
    ("NEAR/5 trial", "missing the word before"),
    ("remdesivir NEAR/x trial", "whole-number distance"),
    ("the NEAR/3 trial", "stop word"),
    ("long-covid NEAR/3 trial", "more than one word"),
    ("foo:bar", "Unknown field 'foo'"),
    ("vaccine AND", "AND at position 8 needs something after it"),
    ("AND vaccine", "needs something before it"),
    ("vaccine OR )", "needs something after it"),
    ("NOT", "needs something after it"),
    ("year>=twenty", "whole number"),
    ("year>=", "needs a value"),
    ("title:", "needs a word or a phrase"),
])
def test_syntax_errors(query, message):
    with pytest.raises(QuerySyntaxError, match=message):
        parse(query)
