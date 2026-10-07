"""Analyzer rules from the spec, with answers worked out by hand."""
import pytest

from prism.analyzer import analyze


def test_spec_example():
    assert analyze("COVID-19 patients' ACE2 receptors") == ["covid19", "patient", "ace2", "receptor"]


@pytest.mark.parametrize("text", ["COVID-19", "covid 19", "covid19", "Covid-19"])
def test_covid_spellings_collapse(text):
    assert analyze(text) == ["covid19"]


@pytest.mark.parametrize("text", ["SARS-CoV-2", "sars cov 2", "SARS-CoV2", "sarscov2"])
def test_sars_cov_2_spellings_collapse(text):
    assert analyze(text) == ["sarscov2"]


def test_stop_words_and_single_characters_are_dropped():
    # "the", "of" and "and" are stop words, "a" and "x" are one character
    assert analyze("The a x of cats and dogs") == ["cat", "dog"]


def test_porter_stemming():
    assert analyze("running vaccines receptors") == ["run", "vaccin", "receptor"]


def test_positions_are_counted_after_stop_word_removal():
    # Documented limitation: "contact the tracing" is indistinguishable from "contact tracing".
    assert analyze("contact the tracing") == analyze("contact tracing") == ["contact", "trace"]


def test_tokens_are_alphanumeric_only():
    assert analyze("ace-2/ACE2, p53!") == ["ace", "ace2", "p53"]


@pytest.mark.parametrize("text", ["", "   ", "!!! ??? ...", "the of and", None])
def test_nothing_survives(text):
    assert analyze(text) == []


def test_analysis_is_deterministic_and_idempotent_on_plain_words():
    text = "Remdesivir trial in hospitalized patients"
    assert analyze(text) == analyze(text)
    assert analyze(text) == ["remdesivir", "trial", "hospit", "patient"]
