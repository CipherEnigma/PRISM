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
| 2026-10-07 | eval/metrics.py, tests/test_metrics.py | D1: P@10, nDCG@10 (graded gains), Recall@100, evaluate/mean helpers, relevance_stats (Recall@100 ceiling); hand-computed toy tests | Claude Code (Sonnet 5.5) |
| 2026-10-07 | eval/data.py, eval/stats.py, eval/baselines.py, tests/test_eval_data.py, tests/test_eval_stats.py | D: topic/qrels loading, seeded split with overwrite guard, paired bootstrap / sign test / Holm / Spearman / repeated k-fold, placeholder tf-idf baseline; hand-checked tests | Claude Code (Sonnet 5.5) |
| 2026-10-07 | eval/metrics.py | D: Judged@k, condensed nDCG, evaluate_extended | Claude Code (Sonnet 5.5) |
| 2026-10-07 | eval/runner.py, scripts/run_eval.py, tests/test_eval_runner.py | D2: run_variant with warm-up and latency, TREC run file, per-topic CSV, summary JSON; CLI with --variant/--all/--report | Claude Code (Sonnet 5.5) |
| 2026-10-07 | eval/tune.py, eval/pipeline.py, scripts/run_tune.py, tests/test_eval_tune.py, tests/test_eval_pipeline.py | D3: grid search over cached score components with a tuning-half-only guard, CV check, gate/adaptive-zone/champion-size stages wired to the real engine | Claude Code (Sonnet 5.5) |
| 2026-10-07 | eval/plots.py, eval/report.py, tests/test_eval_report.py | D4: results/stats/latency tables, nDCG bars, per-topic differences, specificity scatter, sensitivity and champion plots, report builder | Claude Code (Sonnet 5.5) |
| 2026-10-07 | tests/test_end_to_end.py | Tiny-corpus end-to-end test through the real indexer, parser and search() plus the harness | Claude Code (Sonnet 5.5) |
| 2026-10-07 | eval/split.json, README.md, scripts/demo.sh | Seeded 25/25 split; README with exact data links and protocol; demo script that runs live queries | Claude Code (Sonnet 5.5) |
