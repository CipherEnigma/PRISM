# PRISM: Paper Retrieval using Indexed Structure and Merit

A structure-aware search engine for COVID-19 research papers (CSD358, Track T6). Zones, phrases, metadata filters and an age-normalized, query-gated authority score, evaluated against flat tf-idf and BM25 on the 50 TREC-COVID topics. The core engine is hand-written Python (numpy, pandas, nltk); no Elasticsearch, Lucene or vector database.

**Start with [spec.md](spec.md)**: the combined plan, architecture and per-member build spec. AI assistance is logged in [AI_USE.md](AI_USE.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

Run every command from the repository root. `scripts/run_eval.py` and `scripts/run_tune.py` add the root to the import path themselves; for the other scripts, if you see `No module named 'prism'`, run `pip install -e .` once or prefix the command with `PYTHONPATH=.`. `scikit-learn` is only needed for the placeholder baseline in the tests (they skip without it).

## Data

Two public downloads go in `data/` (git-ignored; every machine downloads its own copy).

| File | Where from | Size |
| --- | --- | --- |
| BEIR TREC-COVID (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip | 74 MB |
| CORD-19 `metadata.csv`, release 2020-07-16 | https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/metadata.csv | 269 MB |

`metadata.csv` is downloadable on its own; the 3.6 GB release tarball is not needed.

```bash
mkdir -p data && cd data
curl -LO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip && unzip trec-covid.zip
curl -L -o metadata.csv https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/metadata.csv
cd ..
```

The zip unpacks to `data/trec-covid/` with `corpus.jsonl` (171,332 papers: `_id`, `title`, `text` = abstract), `queries.jsonl` (50 topics) and `qrels/test.tsv` (66,336 judgments plus a header row). If your unzip adds an extra `trec-covid/` level, move the files up so the paths above hold.

Checked facts about the data (computed by `eval/metrics.py: relevance_stats` on the files above):

- Every `cord_uid` in the corpus has a row in `metadata.csv` (100%); 74% have a DOI and 99.99% a publish date.
- Qrels scores are 2 (14,217 rows), 1 (10,456), 0 (41,661) and -1 (2 rows). **Relevant means score >= 1**, fixed for the whole project; the two -1 rows count as non-relevant.
- There are 24,673 relevant documents, 493.5 per topic (minimum 111, median 481, maximum 1,266). Every topic has more than 100 relevant documents, so **Recall@100 cannot exceed about 0.27 on average**, whatever the system. Read Recall@100 with that ceiling in mind.
- Queries use the short `text` field of each topic (the BEIR convention), e.g. "what is the origin of COVID-19".

## Build the index

```bash
python scripts/data_stats.py                         # document counts, metadata coverage, year histogram
python scripts/build_index.py --out index/            # full build
python scripts/build_index.py --out index/ --subset 30000   # development: every judged paper plus a seeded sample
```

If an index was built from a wrong or partial `metadata.csv` (check: it must be 269,219,095 bytes, and the index must report year known for 100% of papers), the metadata part can be recomputed without re-indexing the text:

```bash
python scripts/patch_index_metadata.py --src path/to/index.pkl --out index/
python scripts/patch_index_metadata.py --src index/ --out index/ --citations data/citations.jsonl   # later, to add citation counts
```

## Search

```bash
python scripts/run_search.py "coronavirus origin" --explain
python scripts/run_search.py '"contact tracing" AND mobile' --plan
python scripts/run_search.py 'title:remdesivir year>=2020' --variant V6
bash scripts/demo.sh                                  # the fixed demo queries, live
```

Query syntax: bare words are ranked; quoted phrases, `AND`, `OR`, `NOT`, `NEAR/k`, `title:` / `abstract:` and filters such as `year>=2020` restrict. Operators are capitalized.

## Evaluate

```bash
python scripts/run_eval.py --variant V0               # P@10, nDCG@10, Recall@100 on all 50 topics
python scripts/run_eval.py --all --report             # every variant, then results/report.md and figures
python scripts/run_tune.py                            # tune on the tuning half only -> results/tuned.json
pytest                                                # unit and end-to-end tests (no network)
```

Outputs, all git-ignored: `runs/<V>.run` (TREC format), `results/per_topic_<V>.csv`, `results/summary_<V>.json`, `results/report.md`, `results/figs/*.png`.

Protocol:

- **Split.** `eval/split.json` is a seeded (42) 25/25 split of the topics into a tuning half and a reporting half. It is created once and never edited. Weights, beta and champion size are tuned on the tuning half only (`eval/tune.py` refuses any other topic); results are reported on the reporting half and on all 50 topics.
- **Metrics.** P@10, nDCG@10 with graded gains (gain = qrels score, discount 1/log2(rank+1), ideal ordering of the topic's judged documents) and Recall@100. Unjudged documents count as non-relevant, which can understate every system, so the report also gives Judged@10 and a condensed nDCG@10 that drops unjudged documents.
- **Significance.** About 25 reporting topics, so every variant is compared with V0 by a paired bootstrap 95% CI on the mean difference, an exact sign test with wins/losses/ties, and Holm correction across variants.
- **Latency.** Median and 95th percentile per query, after one untimed warm-up call.

## Results

Generated, not typed: run `python scripts/run_eval.py --all --report` and read `results/report.md`. This section is filled in from that file once the final runs are done.

## Status

Working and tested: loader, analyzer, indexer, query parser and Boolean/phrase/NEAR operations, cosine and BM25 scoring, authority and gate, explainer, and the evaluation harness (metrics, runner, split, tuning, statistics, plots), including an end-to-end test on a hand-written corpus.

Still open: the final full-index runs, the citation source for authority (recency is used as a fallback while citation counts are missing), tuned parameters in `prism/config.py`, the report and the demo video.

## Credits

- **TREC-COVID**: Voorhees et al., *TREC-COVID: Constructing a Pandemic Information Retrieval Test Collection*; topics and relevance judgments.
- **CORD-19**: Wang et al., the COVID-19 Open Research Dataset, Allen Institute for AI; `metadata.csv` release 2020-07-16.
- **BEIR**: Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*; the packaged copy of TREC-COVID used here.
