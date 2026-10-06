"""Hour-0 smoke test: the frozen interfaces import and the stub search() runs for every variant."""
import pytest

from prism.config import VARIANTS
from prism.schema import Result
from prism.search import search


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_stub_search_returns_results(variant):
    results = search("covid symptoms", k=3, variant=variant)
    assert len(results) == 3
    assert all(isinstance(r, Result) for r in results)


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        search("x", variant="nope")
