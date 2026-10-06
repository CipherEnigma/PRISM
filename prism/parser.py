"""Query string -> AST (B). Interface frozen in hour 0; internals are B's to design."""
from dataclasses import dataclass


class QuerySyntaxError(ValueError):
    """Readable message for unbalanced quote/parenthesis, missing NEAR operand, unknown field."""


@dataclass
class ParsedQuery:
    raw: str
    terms: list[str]            # analyzed terms used for ranking (everything outside NOT)
    phrases: list[list[str]]    # analyzed phrase term lists
    ast: object                 # Boolean and filter structure, opaque to C


def parse(query: str) -> ParsedQuery:
    raise NotImplementedError
