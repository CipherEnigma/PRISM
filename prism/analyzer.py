"""A: shared analyzer for documents and queries.

Rules in order: lowercase; domain spellings (config.SPELLING_RULES); tokenize [a-z0-9]+;
drop NLTK stop words and 1-char tokens; Porter-stem. Position = index in the returned list.
Changing this invalidates every built index: tell everyone.
"""
from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from prism.config import SPELLING_RULES

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()

try:
    _STOPWORDS = set(stopwords.words("english"))
except LookupError:  # pragma: no cover - first-run fallback
    nltk.download("stopwords", quiet=True)
    _STOPWORDS = set(stopwords.words("english"))


def analyze(text: str) -> list[str]:
    """Analyze text into normalized, stemmed tokens."""
    if text is None:
        return []

    cleaned = str(text).lower()
    for pattern, replacement in SPELLING_RULES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    tokens = _TOKEN_RE.findall(cleaned)
    analyzed: list[str] = []
    for token in tokens:
        if len(token) <= 1:
            continue
        if token in _STOPWORDS:
            continue
        analyzed.append(_STEMMER.stem(token))
    return analyzed
