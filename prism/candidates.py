"""AST + Index -> candidate doc ids (B). Interface frozen in hour 0."""
import numpy as np

from prism.index import Index
from prism.parser import ParsedQuery


def candidates(pq: ParsedQuery, index: Index, use_champions: bool = False) -> np.ndarray | None:
    """Sorted internal doc ids satisfying every operator, or None meaning unrestricted."""
    raise NotImplementedError
