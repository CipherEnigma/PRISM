# AI use log

One line per AI-assisted change: date, file(s), what changed, tool. The report's AI-use declaration is built from this file.

| Date | File(s) | What changed | Tool |
| --- | --- | --- | --- |
| 2026-10-06 | whole repo skeleton | Created layout, interface stubs (schema, Index, parse/candidates, search), config, placeholders | Claude Code (Sonnet 5.5) |
| 2026-10-06 | spec.md | Merged the Plan and Architecture and Build Spec docs into one file | Claude Code (Sonnet 5.5) |
| 2026-10-06 | prism/boolean_ops.py, tests/test_boolean_ops.py | B1: hand-written intersect/union/difference/intersect_skip, numpy equivalents, intersect_many (shortest first), union_many; property tests vs numpy | Claude Code (Opus 5.5) |
| 2026-10-06 | prism/boolean_ops.py, tests/fake_index.py, tests/test_phrase_near.py | B1: phrase_match and near (NEAR/k, unordered); in-memory FakeIndex and five-document test corpus; hand-checked phrase/NEAR tests | Claude Code (Opus 5.5) |
| 2026-10-06 | prism/parser.py, tests/fake_analyzer.py, tests/test_parser.py | B2: hand-written tokenizer and recursive-descent parser (AND/OR/NOT, NEAR/k, phrases, title:/abstract: zones, field filters), AST node types, ranking-term collection, QuerySyntaxError messages; stand-in analyzer for tests; parser tests | Claude Code (Opus 5.5) |
| 2026-10-06 | prism/candidates.py, tests/test_candidates.py | B3: candidates() evaluating the AST bottom-up (AND shortest first then NOT via difference, OR, lone NOT, filters, phrase/NEAR), bare-word rule, champion-list option; 25 hand-checked candidate-set tests | Claude Code (Opus 5.5) |
| 2026-10-06 | prism/candidates.py, prism/boolean_ops.py, tests/test_candidates.py | B4: describe() plan printer sharing candidates()' evaluation via a trace (operand sizes, shortest-first order, intermediate sizes, early stop); nested ANDs flattened; intersect_many records step sizes; describe tests | Claude Code (Opus 5.5) |
| 2026-10-07 | prism/boolean_ops.py, tests/fake_index.py, tests/test_phrase_near.py | Vectorized phrase_match and near (int64 doc*STRIDE+position keys, np.isin / searchsorted) used when Index.positions_flat exists, loop versions kept as reference; tests run both paths and compare them on a random corpus | Claude Code (Opus 5.5) |
| 2026-10-07 | prism/index.py, scripts/build_index.py | (A's files, edited by B with A's OK) Added Index.positions_flat for vectorized phrase/NEAR; fixed --subset qrels reading (BEIR qrels/test.tsv, header row, corpus-id in column 2) | Claude Code (Opus 5.5) |
