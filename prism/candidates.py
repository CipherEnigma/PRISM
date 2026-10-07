"""AST + Index -> candidate doc ids (B). Interface frozen in hour 0.

The AST is evaluated bottom-up into sorted id arrays:

- Term -> the zone's postings; Phrase and Near -> boolean_ops; Filter -> index.docs_where.
- And -> intersect the positive children shortest first, then subtract the NOT children,
  so the whole collection is only built for a NOT with no positive sibling.
- Or -> union. A lone Not -> every doc except the child's.
- Seq -> the AND of its restrictions (everything but bare words). With no restrictions,
  a nested Seq such as `(vaccine efficacy)` matches any of its words, and the top-level
  Seq returns None: the query is unrestricted.

describe() runs the same evaluation with a trace switched on and prints what it did: each
operand's size, the order of the intersections and every intermediate result size.
"""
import numpy as np

from prism.boolean_ops import (
    ID_DTYPE, intersect_many, near, np_difference, phrase_match, union_many,
)
from prism.index import Index
from prism.parser import And, Filter, Near, Not, Or, ParsedQuery, Phrase, Seq, Term, is_bare


def candidates(pq: ParsedQuery, index: Index, use_champions: bool = False) -> np.ndarray | None:
    """Sorted internal doc ids satisfying every operator, or None meaning unrestricted."""
    restrictions = _restrictions(pq.ast)
    if restrictions:
        return _and(restrictions, index)
    if use_champions and pq.terms:
        return _champion_union(pq.terms, index)
    return None


def describe(pq: ParsedQuery, index: Index, use_champions: bool = False) -> str:
    """Readable account of how candidates() evaluates pq, for the report and the video."""
    lines = [f"Query: {pq.raw}", f"Ranking terms: {', '.join(pq.terms) or '(none)'}"]
    if pq.phrases:
        lines.append("Phrases: " + ", ".join(f'"{" ".join(p)}"' for p in pq.phrases))
    restrictions = _restrictions(pq.ast)
    if restrictions:
        trace: list[str] = []
        result = _and(restrictions, index, trace)
        lines += trace
        lines.append(f"Result: {_docs(len(result))} in the candidate set")
    elif use_champions and pq.terms:
        trace = []
        result = _champion_union(pq.terms, index, trace)
        lines += trace
        lines.append(f"Result: {_docs(len(result))} in the candidate set")
    else:
        lines.append("No restrictions: every doc containing at least one ranking term is a "
                     "candidate (candidates() returns None).")
    return "\n".join(lines)


def _restrictions(node) -> list:
    if node is None:
        return []
    if isinstance(node, Seq):
        return [c for c in node.children if not is_bare(c)]
    return [node]


def _champion_union(terms: list[str], index: Index, trace: list[str] | None = None) -> np.ndarray:
    """Union of each term's champion list in the all zone; full postings if none was built."""
    lists, rows = [], []
    for term in terms:
        champ = index.champion(term, "all")
        if champ is None:
            ids = index.postings(term, "all")[0]
            rows.append(("", term, f"df {len(ids):,} (no champion list, full postings)", ""))
        else:
            ids = champ
            rows.append(("", term, f"{_docs(len(ids))} (champion list)", ""))
        lists.append(ids)
    result = union_many(lists)
    if trace is not None:
        _block(trace, 0, f"Champion lists, OR of each term -> {_docs(len(result))}", rows)
    return result


def evaluate(node, index: Index, trace: list[str] | None = None, depth: int = 0) -> np.ndarray:
    """Sorted ids of the docs matching one AST node. trace, if given, collects describe() lines."""
    if isinstance(node, Term):
        return np.asarray(index.postings(node.term, node.zone)[0], dtype=ID_DTYPE)
    if isinstance(node, Phrase):
        return phrase_match(index, list(node.terms), node.zone)
    if isinstance(node, Near):
        return near(index, node.t1, node.t2, node.k, node.zone)
    if isinstance(node, Filter):
        return np.asarray(index.docs_where(node.field, node.op, node.value), dtype=ID_DTYPE)
    if isinstance(node, And):
        return _and(list(node.children), index, trace, depth)
    if isinstance(node, Or):
        return _union(list(node.children), "OR", index, trace, depth)
    if isinstance(node, Not):
        return _and([node], index, trace, depth)
    if isinstance(node, Seq):
        restrictions = _restrictions(node)
        if restrictions:
            return _and(restrictions, index, trace, depth)
        return _union(list(node.children), "ANY OF", index, trace, depth)
    raise TypeError(f"Unknown query node: {node!r}")


def _flatten(children: list) -> list:
    """(a AND b) AND c -> a AND b AND c, so shortest-first ordering sees every operand."""
    out = []
    for c in children:
        if isinstance(c, And):
            out += _flatten(list(c.children))
        elif isinstance(c, Seq) and _restrictions(c):
            out += _flatten(_restrictions(c))
        else:
            out.append(c)
    return out


def _and(children: list, index: Index, trace: list[str] | None = None, depth: int = 0) -> np.ndarray:
    children = _flatten(children)
    if len(children) == 1 and isinstance(children[0], (Or, Seq)):
        return evaluate(children[0], index, trace, depth)     # nothing to AND it with
    sub = [] if trace is not None else None     # lines from nested groups, printed after ours
    positives = [(c, evaluate(c, index, sub, depth + 1)) for c in children if not isinstance(c, Not)]
    negatives = [(c.child, evaluate(c.child, index, sub, depth + 1))
                 for c in children if isinstance(c, Not)]
    positives.sort(key=lambda p: len(p[1]))     # same stable order intersect_many uses

    rows = []
    if positives:
        steps: list[int] = []
        result = intersect_many([ids for _, ids in positives], steps=steps)
        first, first_ids = positives[0]
        rows.append(("start", _label(first), _size(first, first_ids), ""))
        for n, (node, ids) in enumerate(positives[1:]):
            after = f"-> {_docs(steps[n])}" if n < len(steps) else "skipped (already empty)"
            rows.append(("AND", _label(node), _size(node, ids), after))
    else:
        result = np.arange(index.n_docs, dtype=ID_DTYPE)
        rows.append(("start", "all docs", _docs(index.n_docs), ""))
    for node, ids in negatives:
        if len(result) == 0:
            rows.append(("NOT", _label(node), _size(node, ids), "skipped (already empty)"))
            continue
        result = np_difference(result, ids)
        rows.append(("NOT", _label(node), _size(node, ids), f"-> {_docs(len(result))}"))

    if trace is not None:
        title = "AND, shortest list first" if len(positives) > 1 else "AND"
        _block(trace, depth, f"{title} -> {_docs(len(result))}", rows)
        trace.extend(sub)
    return result


def _union(children: list, name: str, index: Index, trace: list[str] | None, depth: int) -> np.ndarray:
    sub = [] if trace is not None else None
    evaluated = [(c, evaluate(c, index, sub, depth + 1)) for c in children]
    result = union_many([ids for _, ids in evaluated])
    if trace is not None:
        rows = [("", _label(c), _size(c, ids), "") for c, ids in evaluated]
        _block(trace, depth, f"{name} -> {_docs(len(result))}", rows)
        trace.extend(sub)
    return result


def _label(node) -> str:
    """Short name for an operand; groups are expanded in their own block below."""
    zone = "" if getattr(node, "zone", "all") == "all" else f"{node.zone}:"
    if isinstance(node, Term):
        return f"{zone}{node.term}"
    if isinstance(node, Phrase):
        return f'{zone}phrase "{" ".join(node.terms)}"'
    if isinstance(node, Near):
        return f"{node.t1} NEAR/{node.k} {node.t2}"
    if isinstance(node, Filter):
        value = f'"{node.value}"' if isinstance(node.value, str) and " " in node.value else node.value
        return f"{node.field}{node.op}{value}"
    if isinstance(node, Seq):
        restrictions = _restrictions(node)
        if len(restrictions) == 1:
            return _label(restrictions[0])
        return "(AND group, below)" if restrictions else "(ANY OF group, below)"
    if isinstance(node, Or):
        return "(OR group, below)"
    if isinstance(node, And):
        return "(AND group, below)"
    if isinstance(node, Not):
        return "(NOT group, below)"
    return repr(node)


def _size(node, ids: np.ndarray) -> str:
    return f"df {len(ids):,}" if isinstance(node, Term) else _docs(len(ids))


def _docs(n: int) -> str:
    return f"{n:,} doc" if n == 1 else f"{n:,} docs"


def _block(trace: list[str], depth: int, title: str, rows: list[tuple[str, str, str, str]]) -> None:
    """Append a heading and aligned rows (operation, operand, size, running result)."""
    indent = "  " * depth
    trace.append(indent + title)
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    for row in rows:
        cells = [row[i].ljust(widths[i]) for i in range(3)] + [row[3]]
        trace.append(indent + "  " + "  ".join(c for c in cells if c).rstrip())
