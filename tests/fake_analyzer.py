"""Stand-in for A's analyze() so B's tests run before the real one exists.

Follows the spec's rules (lowercase, domain spellings, [a-z0-9]+, drop stop words and
one-character tokens, stem) but with a small stop list and a lookup table instead of NLTK's
Porter stemmer. The table covers the words B's tests use and matches tests/fake_index.py.
"""
import re

from prism.config import SPELLING_RULES

STOP_WORDS = {"the", "a", "an", "and", "or", "not", "of", "in", "on", "for", "to", "is", "with"}
STEMS = {
    "tracing": "trace", "traced": "trace", "trials": "trial", "vaccine": "vaccin",
    "vaccines": "vaccin", "efficacy": "efficaci", "hydroxychloroquine": "hydroxychloroquin",
    "patients": "patient", "hospitalized": "hospit", "networks": "network", "phones": "phone",
}


def fake_analyze(text: str) -> list[str]:
    text = text.lower()
    for pattern, replacement in SPELLING_RULES:
        text = re.sub(pattern, replacement, text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [STEMS.get(t, t) for t in tokens if t not in STOP_WORDS and len(t) > 1]
