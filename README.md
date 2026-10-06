# PRISM: Paper Retrieval using Indexed Structure and Merit

A structure-aware search engine for COVID-19 research papers (CSD358, Track T6). Zones, phrases, metadata filters and an age-normalized, query-gated authority score, evaluated against flat tf-idf and BM25 on the 50 TREC-COVID topics. The core engine is hand-written Python (numpy, pandas, nltk).

Status: skeleton. Interfaces are frozen; modules are stubs.

**Start with [spec.md](spec.md)**: the combined plan, architecture and per-member build spec (who builds what, interfaces, checkpoints).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

## TODO (owners fill in)

- Data download (A): BEIR TREC-COVID + CORD-19 `metadata.csv`
- Build index (A): `python scripts/build_index.py --out index/`
- Run a query (C): `python scripts/run_search.py "query" --explain`
- Run evaluation (D): `python scripts/run_eval.py --variant V0`
- Authority source and coverage (C): hour-2 decision
- Relevance score values in qrels (D)
- Results table, credits (TREC-COVID, CORD-19, BEIR), pointer to `AI_USE.md`
