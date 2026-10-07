"""Index: hand-checked postings on a five-document corpus, invariants, save/load, champions, filters, authority.

Corpus and analysis (stop words dropped, Porter-stemmed; positions are indexes into the token list):
  d0  title "Cat sat"       -> [cat, sat]            abstract "The cat sat on the mat."  -> [cat, sat, mat]
  d1  title "Dog"           -> [dog]                 abstract "The dog chased the cat."  -> [dog, chase, cat]
  d2  title "Mat"           -> [mat]                 abstract "A mat is a mat."          -> [mat, mat]
  d3  title ""              -> []                    abstract ""                          -> []
  d4  title "Cat cat cat"   -> [cat, cat, cat]       abstract "Cats."                     -> [cat]
"""
import numpy as np
import pytest

from prism.index import Index
from prism.indexer import build_index
from prism.schema import Record

DOCS = [
    Record("d0", "Cat sat", "The cat sat on the mat.", year=2019, journal="Nature", source="PMC", publish_month="2019-04"),
    Record("d1", "Dog", "The dog chased the cat.", year=2020, journal="Lancet", source="PMC", publish_month="2020-03"),
    Record("d2", "Mat", "A mat is a mat.", year=2020, journal="lancet", source="Elsevier", publish_month="2020-03"),
    Record("d3", "", "", year=None),
    Record("d4", "Cat cat cat", "Cats.", year=2021, journal="BMJ", publish_month="2021-01"),
]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny")
    build_index(DOCS, out)
    return out


@pytest.fixture(scope="module")
def index(built):
    return Index.load(built)


def _p(index, term, zone):
    docs, tfs = index.postings(term, zone)
    return docs.tolist(), tfs.tolist()


def test_ids_round_trip(index):
    assert index.n_docs == 5 and index.zones == ["title", "abstract", "all"]
    assert [index.doc_id(i) for i in range(5)] == ["d0", "d1", "d2", "d3", "d4"]
    assert [index.internal_id(f"d{i}") for i in range(5)] == [0, 1, 2, 3, 4]
    assert index.internal_id("nope") == -1
    assert index.title(4) == "Cat cat cat"
    with pytest.raises(IndexError):
        index.doc_id(5)


def test_postings_by_hand(index):
    assert _p(index, "cat", "abstract") == ([0, 1, 4], [1, 1, 1])
    assert _p(index, "cat", "title") == ([0, 4], [1, 3])
    assert _p(index, "cat", "all") == ([0, 1, 4], [2, 1, 4])        # d0: title + abstract, d4: 3 + 1
    assert _p(index, "mat", "abstract") == ([0, 2], [1, 2])
    assert _p(index, "dog", "title") == ([1], [1])
    assert _p(index, "chase", "abstract") == ([1], [1])


def test_df_and_unseen_terms(index):
    assert index.df("cat", "abstract") == 3 and index.df("cat", "title") == 2 and index.df("cat", "all") == 3
    docs, tfs = index.postings("zebra", "all")
    assert len(docs) == 0 and len(tfs) == 0 and index.df("zebra", "all") == 0
    assert index.champion("cat", "all") is None            # short lists use the full postings
    assert index.positions("zebra", "all", 0).tolist() == []


def test_positions_by_hand(index):
    assert index.positions("cat", "title", 4).tolist() == [0, 1, 2]
    assert index.positions("cat", "abstract", 0).tolist() == [0]
    assert index.positions("sat", "abstract", 0).tolist() == [1]
    assert index.positions("mat", "abstract", 2).tolist() == [0, 1]
    assert index.positions("chase", "abstract", 1).tolist() == [1]
    assert index.positions("cat", "abstract", 2).tolist() == []          # d2 has no "cat"


def test_all_zone_keeps_title_and_abstract_apart(index):
    """Title and abstract are joined with a gap, so a phrase cannot straddle the boundary."""
    title_end = index.positions("sat", "all", 0)[0]            # d0 title: cat=0, sat=1
    abstract_start = index.positions("mat", "all", 0)[0]       # d0 abstract: cat, sat, mat after a gap
    assert title_end == 1 and abstract_start - title_end > 3


def test_lengths_and_averages(index):
    assert index.lengths("abstract").tolist() == [3, 3, 2, 0, 1]
    assert index.lengths("title").tolist() == [2, 1, 1, 0, 3]
    assert index.lengths("all").tolist() == [5, 4, 3, 0, 4]
    assert index.doc_len(0, "all") == 5
    assert index.avg_len("abstract") == pytest.approx(9 / 5)
    assert index.avg_len("title") == pytest.approx(7 / 5)
    assert index.avg_len("all") == pytest.approx(16 / 5)


def test_cosine_norms_by_hand(index):
    assert index.doc_norm(0, "abstract") == pytest.approx(3 ** 0.5)               # three terms, tf 1 each
    assert index.doc_norm(2, "abstract") == pytest.approx(1 + np.log10(2))        # one term, tf 2
    assert index.doc_norm(4, "title") == pytest.approx(1 + np.log10(3))           # one term, tf 3
    assert index.doc_norm(4, "abstract") == pytest.approx(1.0)
    assert index.doc_norm(3, "abstract") == 0.0                                    # empty document
    assert index.norms("abstract").shape == (5,)


def test_postings_invariants_for_every_term(index):
    """Ascending ids, positive tfs, sorted positions of the right count, df = list length."""
    for zone in index.zones:
        for term in list(index._postings[zone]):
            docs, tfs = index.postings(term, zone)
            assert len(docs) > 0 and np.all(np.diff(docs) > 0), (zone, term)
            assert np.all(tfs > 0) and index.df(term, zone) == len(docs)
            for doc, tf in zip(docs.tolist(), tfs.tolist()):
                pos = index.positions(term, zone, doc)
                assert len(pos) == tf and np.all(np.diff(pos) > 0), (zone, term, doc)


def test_save_then_load_gives_identical_answers(index, tmp_path):
    index.save(tmp_path / "copy")
    other = Index.load(tmp_path / "copy")
    assert other.n_docs == index.n_docs
    for zone in index.zones:
        np.testing.assert_array_equal(other.norms(zone), index.norms(zone))
        np.testing.assert_array_equal(other.lengths(zone), index.lengths(zone))
        for term in index._postings[zone]:
            for a, b in zip(other.postings(term, zone), index.postings(term, zone)):
                np.testing.assert_array_equal(a, b)
            for doc in index.postings(term, zone)[0].tolist():
                np.testing.assert_array_equal(other.positions(term, zone, doc), index.positions(term, zone, doc))
    assert [other.field_value(i, "year") for i in range(5)] == [index.field_value(i, "year") for i in range(5)]
    assert [other.title(i) for i in range(5)] == [index.title(i) for i in range(5)]


def test_year_filter_all_operators(index):
    def where(op, value):
        return index.docs_where("year", op, value).tolist()
    assert where(">=", 2020) == [1, 2, 4] and where(">", 2020) == [4] and where("<=", 2019) == [0]
    assert where("<", 2020) == [0] and where("=", 2020) == [1, 2]
    assert index.field_value(3, "year") in (None, 0)                # unknown year is stored as 0
    assert 3 not in where(">=", 1)


def test_journal_and_source_lookup_ignore_case(index):
    assert index.docs_where("journal", "=", "lancet").tolist() == [1, 2]
    assert index.docs_where("journal", "=", "LANCET").tolist() == [1, 2]
    assert index.docs_where("source", "=", "pmc").tolist() == [0, 1]
    assert index.docs_where("journal", "=", "nonexistent").tolist() == []
    assert index.field_value(0, "publish_month") == "2019-04"
    for result in (index.docs_where("year", ">=", 2019), index.docs_where("journal", "=", "lancet")):
        assert result.tolist() == sorted(result.tolist())


def test_authority_without_citations_is_zero(index):
    assert [index.authority(i, "raw") for i in range(5)] == [0.0] * 5
    with pytest.raises(ValueError):
        index.authority(0, "bogus")


def test_authority_with_citations(tmp_path):
    counts = {"a": 0, "b": 5, "c": 50}
    recs = [Record(k, f"t {k}", f"abstract {k}", year=2020, publish_month="2020-03", citations=v)
            for k, v in counts.items()]
    build_index(recs, tmp_path)
    ix = Index.load(tmp_path)
    raw = [ix.authority(i, "raw") for i in range(3)]
    cohort = [ix.authority(i, "cohort") for i in range(3)]
    assert raw[0] == 0.0 and raw[2] == pytest.approx(1.0) and raw[1] == pytest.approx(np.log1p(5) / np.log1p(50))
    # cohort of 3 (merged into its year): percentiles by minimum rank are 1/3, 2/3, 1
    assert cohort == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert raw == sorted(raw) and cohort == sorted(cohort)
    assert all(0.0 <= v <= 1.0 for v in raw + cohort)


def test_champion_lists_hold_the_best_documents(tmp_path):
    """Terms with more than 500 documents get a champion list of the top 500 by tf / length, ascending."""
    recs = [Record(f"p{i}", f"title {i}", "alpha " * (i % 3 + 1) + " ".join(f"w{i}x{k}" for k in range(i % 7)))
            for i in range(600)]
    build_index(recs, tmp_path)
    ix = Index.load(tmp_path)
    champ = ix.champion("alpha", "abstract")
    assert champ is not None and len(champ) == 500 and np.all(np.diff(champ) > 0)
    docs, tfs = ix.postings("alpha", "abstract")
    lengths = ix.lengths("abstract")[docs]
    order = sorted(range(len(docs)), key=lambda i: (-tfs[i] / lengths[i], docs[i]))[:500]
    assert champ.tolist() == sorted(docs[order].tolist())
    assert ix.champion("zebra", "abstract") is None            # unseen term
