"""A: shared analyzer for documents and queries.

Rules in order: lowercase; domain spellings (config.SPELLING_RULES); tokenize [a-z0-9]+;
drop NLTK stop words and 1-char tokens; Porter-stem. Position = index in the returned list.
Changing this invalidates every built index: tell everyone.
"""


def analyze(text: str) -> list[str]:
    raise NotImplementedError
