"""Query string -> AST (B). Interface frozen in hour 0; internals are B's to design.

Query language (operators only in capitals):

    query    := seq
    seq      := or_expr+                     # juxtaposition, e.g. `vaccine efficacy year>=2020`
    or_expr  := and_expr ( "OR" and_expr )*
    and_expr := unary ( "AND" unary )*
    unary    := "NOT" unary | primary
    primary  := "(" seq ")" | near | zoned | filter | phrase | term
    near     := term "NEAR/" INT term
    zoned    := ("title" | "abstract") ":" ( term | phrase )
    filter   := FIELD OP VALUE               # year>=2020  journal:"lancet"  source:pmc
    phrase   := '"' words '"'

The rule that decides everything: a bare word sitting directly in a seq is a ranking term
and restricts nothing. Everything else (an operand of AND/OR/NOT/NEAR, a phrase, a zoned
term, a filter) is a restriction, and the restrictions in a seq are ANDed. A seq with no
restrictions at all matches any of its bare words (see candidates.py).
"""
import re
from dataclasses import dataclass

from prism.analyzer import analyze

ZONE_PREFIXES = {"title", "abstract"}
FIELDS = {"year", "journal", "source", "publish_month"}
INT_FIELDS = {"year"}


class QuerySyntaxError(ValueError):
    """Readable message for unbalanced quote/parenthesis, missing NEAR operand, unknown field."""


@dataclass
class ParsedQuery:
    raw: str
    terms: list[str]            # analyzed terms used for ranking (everything outside NOT)
    phrases: list[list[str]]    # analyzed phrase term lists
    ast: object                 # Boolean and filter structure, opaque to C


# --- AST nodes ---

@dataclass(frozen=True)
class Term:
    term: str
    zone: str = "all"           # "all" means no zone prefix was written


@dataclass(frozen=True)
class Phrase:
    terms: tuple[str, ...]
    zone: str = "all"


@dataclass(frozen=True)
class Near:
    t1: str
    t2: str
    k: int
    zone: str = "all"


@dataclass(frozen=True)
class Filter:
    field: str
    op: str                     # =, >=, <=, >, < (":" is stored as "=")
    value: object


@dataclass(frozen=True)
class And:
    children: tuple


@dataclass(frozen=True)
class Or:
    children: tuple


@dataclass(frozen=True)
class Not:
    child: object


@dataclass(frozen=True)
class Seq:
    """Items written side by side. Bare Terms rank only; the other items restrict."""
    children: tuple


def is_bare(node) -> bool:
    """A word with no operator, quotes or zone: ranks but does not restrict when in a Seq."""
    return isinstance(node, Term) and node.zone == "all"


# --- Lexer ---

@dataclass
class Token:
    kind: str       # WORD PHRASE LPAREN RPAREN AND OR NOT NEAR ZONE FIELD EOF
    text: str       # what the user typed, for error messages
    pos: int        # character offset in the query
    value: object = None    # NEAR: k; ZONE: (zone, word or None); FIELD: (field, op, value or None)


_FIELD_RE = re.compile(r"^([A-Za-z_]+)(>=|<=|:|=|>|<)(.*)$")
_NEAR_RE = re.compile(r"^NEAR/(\d+)$")
_WORD_END = set(' \t\r\n()"')


def tokenize(query: str) -> list[Token]:
    tokens = []
    i = 0
    while i < len(query):
        ch = query[i]
        if ch.isspace():
            i += 1
        elif ch in "()":
            tokens.append(Token("LPAREN" if ch == "(" else "RPAREN", ch, i))
            i += 1
        elif ch == '"':
            end = query.find('"', i + 1)
            if end == -1:
                raise QuerySyntaxError(
                    f"Unbalanced quote: the quote at position {i} is never closed.")
            tokens.append(Token("PHRASE", query[i:end + 1], i, query[i + 1:end]))
            i = end + 1
        else:
            start = i
            while i < len(query) and query[i] not in _WORD_END:
                i += 1
            tokens.append(_classify(query[start:i], start))
    tokens.append(Token("EOF", "end of query", len(query)))
    return tokens


def _classify(word: str, pos: int) -> Token:
    if word in ("AND", "OR", "NOT"):
        return Token(word, word, pos)
    if word.startswith("NEAR/"):
        m = _NEAR_RE.match(word)
        if not m:
            raise QuerySyntaxError(
                f"'{word}' at position {pos}: NEAR needs a whole-number distance, like NEAR/5.")
        return Token("NEAR", word, pos, int(m.group(1)))
    m = _FIELD_RE.match(word)
    if m:
        name, op, rest = m.group(1).lower(), m.group(2), m.group(3)
        if name in ZONE_PREFIXES and op == ":":
            return Token("ZONE", word, pos, (name, rest or None))
        if name in FIELDS:
            return Token("FIELD", word, pos, (name, "=" if op == ":" else op, rest or None))
        known = ", ".join(sorted(ZONE_PREFIXES | FIELDS))
        raise QuerySyntaxError(f"Unknown field '{m.group(1)}' in '{word}'. Known fields: {known}.")
    return Token("WORD", word, pos)


# --- Recursive-descent parser ---

class _Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.i + offset, len(self.tokens) - 1)]

    def next(self) -> Token:
        tok = self.peek()
        self.i += 1
        return tok

    def parse_query(self):
        node = self.seq()
        tok = self.peek()
        if tok.kind == "RPAREN":
            raise QuerySyntaxError(f"Unbalanced parenthesis: ')' at position {tok.pos} has no '('.")
        return node

    def seq(self):
        items = []
        while self.peek().kind not in ("EOF", "RPAREN"):
            if self.peek().kind in ("AND", "OR"):
                tok = self.peek()
                raise QuerySyntaxError(f"{tok.text} at position {tok.pos} needs something before it.")
            items.append(self.or_expr())
        return Seq(tuple(x for x in items if x is not None))

    def or_expr(self):
        children = [self.and_expr()]
        while self.peek().kind == "OR":
            op = self.next()
            children.append(self._operand(op, self.and_expr))
        return _combine(Or, children)

    def and_expr(self):
        children = [self.unary()]
        while self.peek().kind == "AND":
            op = self.next()
            children.append(self._operand(op, self.unary))
        return _combine(And, children)

    def unary(self):
        if self.peek().kind == "NOT":
            op = self.next()
            child = self._operand(op, self.unary)
            return None if child is None else Not(child)
        return self.primary()

    def _operand(self, op: Token, rule):
        if self.peek().kind in ("EOF", "RPAREN", "AND", "OR"):
            raise QuerySyntaxError(f"{op.text} at position {op.pos} needs something after it.")
        return rule()

    def primary(self):
        tok = self.next()
        if tok.kind == "LPAREN":
            inner = self.seq()
            if self.peek().kind != "RPAREN":
                raise QuerySyntaxError(
                    f"Unbalanced parenthesis: '(' at position {tok.pos} is never closed.")
            self.next()
            return inner
        if tok.kind == "WORD" and self.peek().kind == "NEAR":
            return self.near(tok)
        if tok.kind == "NEAR":
            raise QuerySyntaxError(
                f"{tok.text} at position {tok.pos} is missing the word before it.")
        if tok.kind == "ZONE":
            return self.zoned(tok)
        if tok.kind == "FIELD":
            return self.filter(tok)
        if tok.kind == "PHRASE":
            return _phrase(tok.value, "all")
        if tok.kind == "WORD":
            return _word(tok.text, "all")
        raise QuerySyntaxError(f"Unexpected '{tok.text}' at position {tok.pos}.")

    def near(self, left: Token):
        op = self.next()
        right = self.next()
        if right.kind != "WORD":
            raise QuerySyntaxError(
                f"{op.text} at position {op.pos} is missing the word after it.")
        t1, t2 = _single(left, op), _single(right, op)
        return Near(t1, t2, op.value)

    def zoned(self, tok: Token):
        zone, word = tok.value
        if word is not None:
            return _word(word, zone)
        nxt = self.next()
        if nxt.kind == "PHRASE":
            return _phrase(nxt.value, zone)
        if nxt.kind == "WORD":
            return _word(nxt.text, zone)
        raise QuerySyntaxError(f"'{tok.text}' at position {tok.pos} needs a word or a phrase after it.")

    def filter(self, tok: Token):
        name, op, value = tok.value
        if value is None:
            nxt = self.next()
            if nxt.kind not in ("WORD", "PHRASE"):
                raise QuerySyntaxError(f"'{tok.text}' at position {tok.pos} needs a value.")
            value = nxt.value if nxt.kind == "PHRASE" else nxt.text
        value = value.strip()
        if name in INT_FIELDS:
            if not value.isdigit():
                raise QuerySyntaxError(f"{name} needs a whole number, got '{value}'.")
            value = int(value)
        return Filter(name, op, value)


def _combine(cls, children):
    """Drop operands that analyzed to nothing (stop words).

    With no operator written there is nothing to wrap. With one written, keep the wrapper
    even if a single operand survives: in `trial vaccine AND the`, vaccine must stay a
    restriction rather than turn into a bare ranking word.
    """
    kept = [c for c in children if c is not None]
    if not kept:
        return None
    return kept[0] if len(children) == 1 else cls(tuple(kept))


def _word(text: str, zone: str):
    """A typed word: one token -> Term, several (e.g. long-covid) -> Phrase, none -> dropped."""
    tokens = analyze(text)
    if not tokens:
        return None
    return Term(tokens[0], zone) if len(tokens) == 1 else Phrase(tuple(tokens), zone)


def _phrase(text: str, zone: str):
    tokens = analyze(text)
    return Phrase(tuple(tokens), zone) if tokens else None


def _single(tok: Token, op: Token) -> str:
    tokens = analyze(tok.text)
    if len(tokens) != 1:
        problem = "is a stop word" if not tokens else "is more than one word"
        raise QuerySyntaxError(
            f"{op.text} at position {op.pos}: '{tok.text}' {problem}; "
            "NEAR needs a single searchable word on each side.")
    return tokens[0]


# --- Public API ---

def parse(query: str) -> ParsedQuery:
    ast = _Parser(tokenize(query)).parse_query()
    terms: list[str] = []
    phrases: list[list[str]] = []
    _collect(ast, terms, phrases)
    return ParsedQuery(raw=query, terms=list(dict.fromkeys(terms)), phrases=phrases, ast=ast)


def _collect(node, terms: list[str], phrases: list[list[str]]) -> None:
    """Ranking terms and phrases from everything outside a NOT, in query order."""
    if isinstance(node, Term):
        terms.append(node.term)
    elif isinstance(node, Phrase):
        terms.extend(node.terms)
        if len(node.terms) > 1 and list(node.terms) not in phrases:
            phrases.append(list(node.terms))
    elif isinstance(node, Near):
        terms.extend([node.t1, node.t2])
    elif isinstance(node, (And, Or, Seq)):
        for child in node.children:
            _collect(child, terms, phrases)
    # Not and Filter contribute no ranking terms.
