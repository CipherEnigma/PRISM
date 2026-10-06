# PRISM: Combined Spec (Plan, Architecture and Build Spec)

PRISM (Paper Retrieval using Indexed Structure and Merit) is a search engine for COVID-19 research papers that ranks by the structure of each paper (title and abstract zones, exact phrases, metadata filters) and by an authority score that adapts to paper age and to how specific the query is. It is evaluated against flat tf-idf and BM25 on the 50 TREC-COVID topics. Course CSD358, Track T6 (vertical search for science), 36-hour hackathon, four members.

This file merges the two team docs, "PRISM Plan and Architecture" (the why and the overall plan) and "PRISM Build Spec and Work Division" (the hands-on build detail). **If the two ever disagreed, the Build Spec parts win**: this file already applies those corrections (abstract-only corpus, champion lists instead of body tiers, the final `Index` API).

## How to use this file

- Find your role in [Quick reference](#quick-reference-who-does-what) and read your member section in Part 2.
- Everyone reads: Project summary, Shared contracts, Integration and Gates.
- Interfaces are frozen. You may add a method; you may not change a signature without all four agreeing.
- Edit only your own files. Ask the owner (or open a small PR they review) for anything else.

## Quick reference: who does what

| Member | Owns | Hands off to |
| --- | --- | --- |
| **A: Data and index** | `loader.py`, `analyzer.py`, `indexer.py`, `index.py`, `scripts/build_index.py`, `scripts/data_stats.py` | B and C read the index through one `Index` class |
| **B: Query engine** | `parser.py`, `boolean_ops.py`, `candidates.py` | C receives a candidate set plus the parsed ranking terms |
| **C: Ranking, authority, novelty** | `scoring.py`, `authority.py`, `gate.py`, `search.py`, `explain.py` | D calls `search(query, k, variant)` |
| **D: Evaluation and delivery** | `eval/*`, `scripts/run_eval.py`, `README.md`, `AI_USE.md`, report, video edit | Everyone: the harness is the source of truth for every change |

| Hour | Who delivers | What must work |
| --- | --- | --- |
| 0.5 | All | Repo created, interface stubs and fake `search()` committed, everyone can run the harness against the stub |
| 2 | A, C | Data stats posted; the hour-2 citation decision recorded |
| 3 | A | `all`-zone index built and pushed, with `norms` and `lengths` |
| 4 (Gate 1) | D, C | V0 flat tf-idf and BM25 numbers on all 50 topics, written down |
| 12 | A, B | All zones with positions and the parametric index; parser handling phrases and Boolean operators |
| 20 (Gate 2) | C, D | V2 end to end through `search()` with the explainer and V2 numbers |
| 28 (Gate 3) | All | Feature freeze; V3 to V7 numbers collected; no new code except bug fixes |
| 36 | D | README, report, video, submission form |

---

# Part 1: Plan and Architecture

## Project summary

We build PRISM, a structure-aware search engine for COVID-19 research papers, and show with measured results that using zones, phrases, metadata and authority beats flat tf-idf text search.

**Why the name.** A prism splits light into its parts; PRISM splits each paper into its parts (title, abstract, metadata) and scores them separately. The acronym names the two halves of the net score: Indexed Structure (zones, positions, fields) and Merit (age-normalized, query-gated authority).

This is a search engine, not RAG: a query returns a ranked list of papers with a per-result score explanation, and no LLM generates text. The IR engine (index, query parser, scorer) is written by us. Libraries are used only for parsing, stemming, data loading and the evaluation sanity check.

**Dataset.** TREC-COVID over CORD-19: 50 topics with human relevance judgments (qrels). We use the BEIR copy (about 171K papers, title and abstract, one zip download). Upgrade to the original CORD-19 release for body text and bibliographies only if time allows.

**Query types supported.**

- Free text, ranked by zone-weighted cosine similarity.
- Exact phrases, such as "contact tracing", answered from positional postings.
- Boolean and proximity: AND, OR, NOT, NEAR/k.
- Zone and metadata restrictions, such as `title:vaccine` or `year>=2020`.

**Success criteria.**

1. A working end-to-end system that runs live on real queries and reproduces from the README.
2. A table of P@10, nDCG@10 and Recall@100 for flat tf-idf, zoned tf-idf, zoned plus authority, and tiered variants, plus BM25 as a reference.
3. One honest limitation shown live, for example a query where authority promotes a highly cited but weaker match.
4. Every member explains the component they own in the demo video.

## Novelty and rubric alignment

The novelty comes from two to three measurable ideas that fix specific weaknesses of the obvious baseline (flat tf-idf plus raw citation counts), each tested as an ablation so the claim is backed by numbers, not asserted.

**N1: Age-normalized authority.** Raw citation counts mostly measure how long a paper has existed. In CORD-19 nearly every paper is from 2020, so raw counts reward whoever published earliest. Instead, set g(d) to the paper's percentile of citations within its publication-month cohort. Baseline: raw log citation count. Needs citation counts and publication dates, which the hour-2 check must confirm.

**N2: Query-adaptive authority gate.** Static quality should matter more when the query says little. Compute query specificity from the query terms' idf, and let the authority weight β(q) fall as specificity rises. A broad query such as "covid symptoms" leans on well-cited papers, while a specific one such as "remdesivir ACE2 binding" is decided by text match. Baseline: a single fixed β. Motivation from related work: Beel and Gipp show citation counts drive Google Scholar's ranking, and the crowd-ranking paper found citation impact can correlate negatively with how close a paper is to the author's work, so authority helps for some queries and hurts for others. The gate tests whether query specificity predicts which.

**N3 (stretch): Per-query zone weights.** Derive w_z(q) from how much of the query's idf mass appears in each zone, using zone-specific idf. If the query's rare terms appear in titles, title matches get more weight for that query. Baseline: zone weights tuned once globally.

**N4: Inspectable ranking.** Every result prints its per-zone scores, β(q), g(d) and net score, so the video can show why one paper outranks another.

**Honest framing.** We have not done an exhaustive literature search, so the report claims novelty only relative to the baseline and the systems we checked (Covidex, the Google Scholar analysis, Semantic Scholar's public description). If an idea does not beat V2, report it as a negative result with a short explanation.

**If time runs short, cut in this order:** N3 first, then phrase boosts (V4), then NEAR/k. Keep N1 and N2.

**How each rubric criterion is covered.**

| Criterion (marks) | What we show | Evidence |
| --- | --- | --- |
| Use of IR principles (30) | Zone, positional and parametric indexes; Boolean and NEAR/k with postings intersection ordered by document frequency; lnc.ltc cosine; heap top-K; champion lists; static quality g(d) and net score; BM25 as an outside-syllabus extra | Each principle maps to one module; the video shows actual postings, weights and score traces |
| Novelty and creativity (10) | N1 and N2, plus N3 if time allows | Ablations V5 to V7 against V2 and V0 |
| Working system (20) | A CLI that runs live on real queries; one command to build the index and one to search; no hard-coded outputs | README that reproduces the demo; fixed demo query list |
| Evaluation metrics (15) | 50 topics, tuning and reporting split, P@10, nDCG@10, Recall@100, BM25 reference, per-topic wins and losses, latency | One results table plus two graphs |
| Relevance to the track (5) | Uses zones, metadata, citations and authority rather than flat text | Report section 1 |
| Report quality (10) | The required 7-part structure, work division, AI-use declaration | Sections of this doc map one-to-one |
| Video explanation (10) | Problem, live queries, one limitation, pipeline with intermediate output, results, each member on their component | Script drafted in the last six hours |

## Architecture

Two paths over one shared on-disk index:

```text
OFFLINE (run once)                          ONLINE (every query)
records (loader) -> analyzer -> indexer     query string -> parser -> candidates -> scorer -> top-K -> explainer
        |                                                   |            |            |
        +--------> index/ on disk <-------------------------+------------+------------+
                         ^
                         +---- evaluation harness (same index, every variant V0..V7, R)
```

The offline path reads papers, splits them into zones, analyzes the text and writes postings. The online path and the evaluation harness both read the same index, so every variant is measured on identical data.

**Data structures stored in the index.**

| Structure | Contents | Used by |
| --- | --- | --- |
| Zone postings | term to (doc id, term frequency), one index per zone: `title`, `abstract`, `all` | Candidates, scorer |
| Positional postings | term to doc to word positions, for `title` and `abstract` (`all` optional) | Phrase and NEAR/k matching |
| Parametric index | field value to sorted doc ids, for year, journal, source, publish month | Filters such as `year>=2020` |
| Champion lists | top-r documents per term (by tf / doc length) for long posting lists | Candidate generation, tier fallback |
| Document statistics | document frequency per term per zone, cosine norm and length per document per zone | Cosine normalization, idf, BM25 |
| Authority arrays | g(d) in [0, 1] per paper: raw and cohort | Net score |

**Scoring.** Each zone is scored with lnc.ltc cosine similarity (log tf and cosine normalization on the document side, log tf with idf and cosine normalization on the query side). Zone scores are combined with weights and added to the authority score.

```latex
\mathrm{net}(d) = \sum_{z \in \{\mathrm{title},\,\mathrm{abstract}\}} w_z \, \cos(q, d_z) \; + \; \beta \, g(d)
```

The weights w_z and the authority weight β are tuned on half of the topics only. Setting β to 0 gives V1, and a single `all` zone gives V0.

**Query language.** Free text is ranked. Quoted phrases and operators restrict the candidate set. Examples: `"contact tracing" AND mobile`, `vaccine NEAR/5 efficacy`, `title:remdesivir year>=2020`.

## Evaluation plan

Every feature is justified by a measured change against flat tf-idf on the same 50 TREC-COVID topics.

**System variants.**

| ID | System | Question it answers |
| --- | --- | --- |
| V0 | Flat tf-idf (lnc.ltc) over `all` (title + abstract) | Baseline |
| V1 | Zone-weighted tf-idf | Does zone structure help? |
| V2 | V1 + raw authority g(d) in a net score | Does citation authority help or hurt? |
| V3 | V2 with champion lists (full postings on fallback) | How much speed is saved, and at what quality cost? |
| V4 | V2 with phrase boosts from the positional index | Do exact phrases help ranking? |
| V5 | V2 with age-normalized authority (N1) | Does cohort normalization beat raw counts? |
| V6 | V5 with query-adaptive β(q) (N2) | Does the gate beat a fixed β? |
| V7 (stretch) | V6 with per-query zone weights (N3) | Do adaptive zones beat global ones? |
| R | BM25 (own implementation), reference only | Where do we stand against the standard strong baseline? |

Report which of V5 to V7 beat V2 on the reporting half, per topic and on average. A variant that does not help is a negative result with a short explanation.

**Metrics.** P@10, nDCG@10 and Recall@100 per topic and averaged, plus median and 95th-percentile query latency.

**Protocol.**

1. Split the 50 topics once, seeded, into a tuning half and a reporting half. Tune zone weights, β and the champion size only on the tuning half. Report on the other half, and also all 50.
2. Treat unjudged documents as non-relevant (the standard convention) and say so, since this can understate every system.
3. Report per-topic wins and losses against V0, not only averages. With about 25 reporting topics, small average differences are not reliable.
4. Sanity-check our tf-idf and BM25 against a library implementation on a few queries. Large disagreement means a bug, not a finding.

**Qualitative checks for the video.** One query where zones help, one where a phrase query beats bag-of-words, and one limitation (authority ranks a highly cited but weaker paper above a better match). Use explainer output as evidence.

## Related work

The closest prior work is the set of systems built for TREC-COVID itself. The main lesson is that plain BM25 over well-chosen index units is hard to beat, so our gains must be measured carefully against it. "Opened" means we read the page while preparing the doc; "listing only" means only the title and link were checked, so read it before citing.

| Paper | What it did | What we take from it | Checked |
| --- | --- | --- | --- |
| [TREC-COVID: Constructing a Pandemic Information Retrieval Test Collection](https://arxiv.org/pdf/2005.04474) | Built the topics and judgments we evaluate on | Evaluation protocol, how qrels were pooled (unjudged documents count as non-relevant) | Listing only |
| [Searching for Scientific Evidence in a Pandemic: An Overview of TREC-COVID](https://arxiv.org/pdf/2104.09632) | Summary of all rounds and participant approaches | Which ideas worked across systems; use for the Related Work paragraph | Listing only |
| [Covidex](https://arxiv.org/pdf/2007.07846) (Zhang et al., 2020) | Anserini BM25 over three indexes (title+abstract, full text, paragraph), then neural reranking | The index-unit choice is our zone design problem: paragraph indexing was suboptimal in round 1 and rank fusion across indexes helped in round 2 | Opened |
| [SLEDGE](https://arxiv.org/pdf/2005.02365) and [SLEDGE-Z](https://arxiv.org/pdf/2010.05987) | COVID literature search baselines | Reference point for how strong a learned reranker is; we are not competing with neural systems | Listing only |
| [CO-Search](https://arxiv.org/pdf/2006.09595) | Semantic search, QA and summarization for COVID | An example of the "beyond IR" side, for the novelty comparison | Listing only |
| [Simple BM25 extension to multiple weighted fields](https://ir.webis.de/anthology/2004.cikm_conference-2004.6) (Robertson et al., 2004) | BM25F: combine zones inside the scoring function | Formal justification for weighting zones differently | Listing only |
| [Google Scholar's Ranking Algorithm: The Impact of Citation Counts](https://isg.beel.org/pubs/Google%20Scholar's%20Ranking%20Algorithm%20-%20The%20Impact%20of%20Citation%20Counts%20--%20preprint.pdf) (Beel and Gipp) | Queried Google Scholar 1,561 times and compared citation counts with rank position over 1.36 million articles | Production academic search favors highly cited papers, which suits "popular standard literature" rather than newest work: the tradeoff g(d) will show | Opened |
| [Learning to Rank Scientific Documents from the Crowd](https://arxiv.org/pdf/1611.01400) | Ranked cited papers using per-section cosine, citation impact and age | Supports per-zone cosine, citation impact, recency. Citation impact correlated negatively with author-judged closeness: a warning that authority can hurt | Opened |
| [Semantic Scholar ranking FAQ](https://webflow.semanticscholar.org/faq/ranking-function) | States relevance uses "different aspects of the query and the paper details"; offers sorting by relevance, citations, influential papers, date | A real academic engine blends text relevance with citation signals; no detail on signals, so do not cite it for specifics | Opened |
| PageRank (Page, Brin, Motwani, Winograd, 1999) | Link-analysis authority score | Optional PageRank g(d). Cited from memory; confirm the reference before use | Not checked |
| Introduction to Information Retrieval (Manning, Raghavan, Schütze) | Textbook source for zones, tiered indexes, champion lists, net score | Primary citation for the syllabus concepts. Cited from memory | Not checked |

**Where we are different.** Those systems add neural rerankers on top of BM25. Ours stays within classical IR and makes the structure visible: per-zone postings, phrase and proximity operators, parametric filters, champion lists and an explainable net score, each ablated against a flat baseline. That is a smaller claim than beating neural systems, and the report should state it plainly.

## Risks and fallbacks

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Citation counts or publication dates missing or too sparse for N1 and N2 | Hour-2 check fails | Use an external citation-count source, or recency as the static score; if authority is too weak, report that and keep zone and phrase work as the core |
| BEIR copy has only title and abstract, so there is no body zone | Loader output at hour 2 | Zones are title versus abstract (already assumed); move to original CORD-19 full text only if hours 12 to 20 have slack |
| Index build too slow or too big in Python for about 171K papers | First full build takes more than a few minutes, or memory errors | Develop on a 30K to 50K subset; store postings as compact arrays; positions for title and abstract only |
| Zone weighting shows no gain over flat tf-idf | V1 and V0 differ little at Gate 2 | Report with per-topic analysis; the ablation is still a valid result. Lean on phrase, authority and novelty variants |
| Authority or the gate hurts nDCG@10 | V2, V5 or V6 below V0 on the tuning half | Report as a negative result with per-topic breakdown; consistent with related work's warning |
| Parser and NEAR/k consume too much time | Parser unfinished at hour 14 | Cut NEAR/k first; keep phrases, AND, OR, NOT and the filters |
| Interface mismatch between members | Integration bugs at Gate 1 or 2 | Freeze `Index` and `search()` signatures in hour 0; develop against stubs |
| Overfitting weights on about 25 topics | Big gap between tuning and reporting numbers | Tune on one half only; report both halves and per-topic wins and losses |
| Live demo fails during recording | Any crash in rehearsal | Prebuilt index, fixed demo query list, record in short segments per member |

## Submission checklist

- [ ] Repository with a README covering setup, how to run, where the data comes from, what works and what is still planned
- [ ] Demo video, 5 to 8 minutes, unlisted link, no slides, each member explains their component
- [ ] Report PDF, at most 8 pages excluding references and appendix, in the required 7-part structure
- [ ] Pipeline diagram in the report (adapt the architecture above)
- [ ] Work division note and AI-use declaration in the report
- [ ] Datasets credited (TREC-COVID, CORD-19, and BEIR)
- [ ] Evaluation table with P@10, nDCG@10, Recall@100 against the baseline, plus graphs
- [ ] One limitation shown live in the video
- [ ] Form submitted before the deadline

**Data ethics.** The plan uses downloaded public datasets and does no crawling. If an external API is used for citation counts, respect its rate limits and terms, and collect no personal data.

---

# Part 2: Build Spec and Work Division

## Ground rules and setup

We build one Python package, `prism`, as four modules owned by four people and joined by three fixed interfaces; inside its own module each owner designs freely.

**Decisions, so nobody re-argues them mid-build:**

1. **Corpus.** BEIR copy of TREC-COVID, title and abstract only. Full text is a late upgrade, not part of the core.
2. **Stack.** Python 3.10+, numpy, pandas, nltk (stopwords and Porter stemmer), matplotlib. The `beir` package is used only to load the data. The core engine uses no Elasticsearch, Lucene, FAISS or vector database. `rank_bm25` and `pytrec_eval` are allowed only as cross-checks in tests, never as the engine.
3. **Zones.** `title`, `abstract`, and `all` (title and abstract concatenated). The `all` zone powers flat V0 and BM25. A `body` zone is added only if full text is joined later.
4. **Tiering.** Implemented as champion lists (the top-r documents per term), because without body text there is no body tier. This replaces the "tier 1 = title and abstract, tier 2 = body" wording while we stay abstract-only.
5. **Authority g(d).** Needs a citation count per paper. The source is decided at the hour-2 check: an external API by PubMed ID or DOI, or a fallback described in Member C.
6. **Evaluation.** 50 topics, one fixed seeded split into tuning and reporting halves, and "relevant" means qrels score at least 1.

**Repository layout.**

```text
prism/
  README.md  requirements.txt  AI_USE.md  .gitignore   # data/ and index/ are ignored
  prism/
    schema.py        # Record, Result            (shared, frozen in hour 0)
    config.py        # paths, zone weights, variant configs
    loader.py        # A: BEIR + metadata -> Record stream
    analyzer.py      # A: text -> tokens (shared by docs and queries)
    indexer.py       # A: build_index(records) -> files in index/
    index.py         # A: Index class (read API, frozen in hour 0)
    parser.py        # B: query string -> AST
    boolean_ops.py   # B: intersect, union, difference, phrase, near
    candidates.py    # B: AST + Index -> candidate doc ids (+ champion fallback)
    scoring.py       # C: lnc.ltc cosine, zone combination, BM25
    authority.py     # C: g(d) raw and age-normalized (N1)
    gate.py          # C: query specificity, beta(q) (N2), zone weights (N3)
    search.py        # C: search(query, k, variant) orchestrator
    explain.py       # C: per-result score breakdown
  eval/
    metrics.py  runner.py  tune.py  plots.py             # D
  scripts/
    build_index.py  run_search.py  run_eval.py
  tests/
```

**Hour-0 setup, all four together (about 30 minutes).**

- [ ] Create the GitHub repo, add everyone, and commit the layout above with empty files.
- [ ] Commit `schema.py`, the `Index` class signatures, and a `search()` stub that returns fake results, exactly as written in Shared contracts. This lets D wire the evaluation harness on day one.
- [ ] One branch per person: `a-index`, `b-query`, `c-rank`, `d-eval`. Merge to `main` at each gate.
- [ ] Create `AI_USE.md` and add a line every time an AI tool writes or changes code. The report needs this declaration.
- [ ] Everyone runs the environment setup below and confirms `import nltk` and `import beir` work.

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas nltk matplotlib beir pytest rank_bm25
pip install pytrec_eval            # optional cross-check; skip if it fails to build
python -c "import nltk; nltk.download('stopwords')"
```

**Conventions.** Type hints on public functions. Seeds fixed wherever randomness appears. Tests never touch the network. Index files are built by a script and never committed.

## Shared contracts

Three interfaces are frozen in hour 0 and committed as stubs: the `Index` read API (A to B and C), `parse` and `candidates` (B to C), and `search` with `Result` (C to D). Change one only with all four people agreeing. Adding a method is allowed.

**1. Record and Result** (`prism/schema.py`).

```python
from dataclasses import dataclass

@dataclass
class Record:
    doc_id: str                  # external id, the string used in the qrels
    title: str
    abstract: str
    body: str | None = None      # only if full text is joined later
    year: int | None = None
    journal: str | None = None
    source: str | None = None
    pubmed_id: str | None = None
    doi: str | None = None
    publish_month: str | None = None   # "YYYY-MM", needed for age-normalized authority
    citations: int | None = None       # filled in by C's authority step

@dataclass
class Result:
    doc_id: str                  # external id
    score: float                 # net score
    zone_scores: dict[str, float]    # {"title": ..., "abstract": ...}
    authority: float             # g(d) actually used
    beta: float                  # authority weight used for this query
    matched_terms: list[str]
    title: str
```

**2. Analyzer** (`prism/analyzer.py`), one function used for documents and queries alike: `analyze(text: str) -> list[str]`. Rules, in order:

1. Lowercase.
2. Normalize a short list of domain spellings before tokenizing, for example `covid-19`, `covid 19` and `covid19` all become `covid19`, and `sars-cov-2` becomes `sarscov2`. Keep the list in `config.py` so it is a documented design choice.
3. Tokenize with the pattern `[a-z0-9]+`.
4. Drop NLTK English stop words and one-character tokens.
5. Porter-stem what remains.

A token's position is its index in the returned list (after stop-word removal). A phrase is consecutive positions. Known limitation to mention in the report: "contact the tracing" would match the phrase "contact tracing".

**3. Index read API** (`prism/index.py`), implemented by A, used by B and C. Internal document ids are integers 0 to N-1; external ids are strings.

```python
class Index:
    n_docs: int
    zones: list[str]                       # ["title", "abstract", "all"]
    def doc_id(self, i: int) -> str: ...                     # internal -> external
    def internal_id(self, doc_id: str) -> int: ...
    def postings(self, term: str, zone: str): ...            # (doc_ids ascending, tfs) as numpy arrays; empty if unseen
    def positions(self, term: str, zone: str, doc: int): ... # sorted word positions as a numpy array
    def df(self, term: str, zone: str) -> int: ...
    def doc_norm(self, doc: int, zone: str) -> float: ...    # cosine norm of the lnc document vector
    def doc_len(self, doc: int, zone: str) -> int: ...       # tokens, for BM25
    def avg_len(self, zone: str) -> float: ...
    def field_value(self, doc: int, name: str): ...          # year, journal, source, publish_month
    def docs_where(self, name: str, op: str, value): ...     # sorted internal ids; op in =, >=, <=, >, <
    def champion(self, term: str, zone: str): ...            # top-r internal ids ascending, or None if not built
    def authority(self, doc: int, mode: str = "raw") -> float: ...  # g(d); modes "raw", "cohort"
    def title(self, doc: int) -> str: ...
    # added at C's request (adding is allowed):
    def norms(self, zone: str): ...                          # full numpy array of doc norms
    def lengths(self, zone: str): ...                        # full numpy array of doc lengths
```

**4. Parsed query and candidates** (`prism/parser.py`, `prism/candidates.py`), implemented by B, used by C.

```python
@dataclass
class ParsedQuery:
    raw: str
    terms: list[str]            # analyzed terms used for ranking (everything outside NOT)
    phrases: list[list[str]]    # analyzed phrase term lists
    ast: object                 # Boolean and filter structure, opaque to C

def parse(query: str) -> ParsedQuery: ...
def candidates(pq: ParsedQuery, index: Index, use_champions: bool = False):
    """Sorted internal doc ids satisfying every operator, or None meaning unrestricted."""
```

**5. Search entry point** (`prism/search.py`), implemented by C, called by D and by the demo.

```python
def search(query: str, k: int = 10, variant: str = "V2") -> list[Result]: ...
```

`variant` is a key into a dictionary in `config.py` whose values are a `VariantConfig`: zone weights, scorer (`cosine` or `bm25`), authority mode (`none`, `raw`, `cohort`), beta, `gate` (on or off), `adaptive_zones` (on or off), `champions` (on or off), and `phrase_boost`. That keeps every evaluated system reproducible from one name.

**Query semantics, so B and C agree.** A query is free-text terms that are ranked plus optional operators that only restrict. With no operators, every document containing at least one query term is a candidate. With operators, ranking happens only inside the candidate set returned by `candidates`, and the ranking vector uses all `terms` outside NOT.

## Member A: Data and index

A delivers the on-disk index and the `Index` class; everyone else is blocked on A for real data, so the first deliverable is a working `all`-zone index by about hour 3, before the fancier zones.

**Files owned:** `loader.py`, `analyzer.py`, `indexer.py`, `index.py`, `scripts/build_index.py`, tests for these.

**A1. Loader (hours 0 to 2).**

- Load the BEIR copy with the `beir` loader. The dataset card lists only `_id`, `title` and `text` per document (title is an empty string when missing), with IDs like `ug7v899j`. So the BEIR copy has no year, journal or citation data. The `text` field is the abstract.
- Get `metadata.csv` from a CORD-19 release (the July 16, 2020 one matches round 5). Check first whether that file can be downloaded on its own, since the full tarball is large. Columns we need: `cord_uid`, `publish_time`, `journal`, `source_x`, `doi`, `pubmed_id`.
- Join on `cord_uid` equal to the BEIR `_id`. Verify on ten IDs before trusting it. Drop duplicate `cord_uid` rows (keep the first). Derive `year` and `publish_month` ("YYYY-MM") from `publish_time`.
- Output: `iter_records() -> Iterator[Record]`. A document with no metadata row still becomes a Record with the optional fields set to None.
- Write `scripts/data_stats.py` that prints: document count, share with empty abstract, share with a year, share with a DOI or PubMed ID, and the year histogram. Post the numbers in the team chat. C needs them for the hour-2 authority decision.

**A2. Analyzer (hours 0 to 2).** Implement exactly the five rules in Shared contracts. Unit-test with strings whose answer you work out by hand, for example `"COVID-19 patients' ACE2 receptors"` should give `["covid19", "patient", "ace2", "receptor"]`. Once merged, nobody edits the analyzer without telling everyone, because changing it invalidates every built index.

**A3. Indexer (hours 1 to 12).** Build one inverted index per zone: `title`, `abstract`, and `all` (title plus abstract). Internal document ids are integers 0 to N-1 in record order, so postings are sorted automatically.

```python
# per zone, while streaming records in order
for doc, rec in enumerate(records):
    tokens = analyze(text_for_zone(rec, zone))
    for term, positions in group_positions(tokens).items():
        docs[term].append(doc)               # array('i'), ascending
        tfs[term].append(len(positions))     # array('h')
        pos_flat[term].extend(positions)     # array('i')
        pos_len[term].append(len(positions))
    doc_len[doc] = len(tokens)
    doc_norm[doc] = sqrt(sum((1 + log10(tf)) ** 2 for tf in term_counts))
```

Implementation notes that will save hours:

- Use `array.array` for the per-term lists, not Python lists of ints. With roughly 171K abstracts, a Python list of every position uses several gigabytes. `array('i')` uses 4 bytes per entry.
- After the pass, convert each term's arrays to numpy and store positions as one flat array plus offsets (the cumulative sum of `pos_len`). `positions(term, zone, doc)` finds the doc's slot with `np.searchsorted(docs, doc)` and slices.
- Persist with `pickle` (protocol 4) or `np.savez`. Either is fine for a hackathon. Record the build time and the size on disk.
- `df(term, zone)` is the length of the posting list. `avg_len(zone)` is the mean of `doc_len`.
- Positions are needed for `title` and `abstract` only. The `all` zone needs positions only if you want phrase matching on it; skip them there if memory is tight.
- Build the `all` zone first, push it, and tell the team. D's baseline needs only `all` postings plus `doc_norm`.

**A4. Parametric index (hours 6 to 12).**

- `year`: a numpy int16 array, 0 meaning unknown. `docs_where("year", ">=", 2020)` is a boolean mask followed by `np.nonzero`.
- `journal`, `source`: a dictionary from lowercase value to a sorted array of document ids. Support the `=` operator.
- `publish_month`: a string array, used by C for age cohorts.
- Return sorted numpy arrays of internal ids from every call so B can intersect them directly.

**A5. Champion lists (hours 12 to 20).** For each term in the `abstract` and `all` zones whose posting list is longer than r (start with r = 500; tune later), keep the top r documents by term frequency divided by document length. Store them sorted ascending and return `None` for rarer terms (use the full posting list). `champion(term, zone)` returns the array. Later C can rebuild champions ordered by tf plus authority.

**A6. Build script and subset mode.** `python scripts/build_index.py --out index/` builds everything. Add `--subset N`, which keeps all documents that appear in the qrels plus a random sample up to N (seeded). Use it for development: a 30K to 50K subset builds in a fraction of the time and still contains every judged paper.

**Done when:**

- [ ] A tiny five-document corpus has hand-checked postings and positions in a test, and `save` followed by `load` returns identical answers.
- [ ] The full build finishes. Note the build time, the vocabulary size per zone, the number of postings, and the size on disk. If a rebuild takes longer than you can tolerate (guideline: over 15 to 20 minutes), profile before continuing.
- [ ] `postings` always returns ascending ids. A test asserts this for 100 random terms.
- [ ] B and C can run against the real index. Target: `all` zone at hour 3, all zones and positions at hour 12.

## Member B: Query engine

B turns a query string into a candidate set using postings operations written by hand, so the video can show a real intersection running in increasing document-frequency order.

**Files owned:** `parser.py`, `boolean_ops.py`, `candidates.py`, tests for these.

**B1. Postings operations (hours 0 to 8).** Work on sorted numpy arrays of internal ids. B can start immediately against random arrays, with no dependency on A.

```python
def intersect(a, b):                       # classic two-pointer merge; show this in the video
    i = j = 0; out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]: out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]: i += 1
        else: j += 1
    return np.array(out, dtype=np.int32)

def intersect_many(lists):                 # query optimization: shortest list first, stop early
    lists = sorted(lists, key=len)
    result = lists[0]
    for nxt in lists[1:]:
        if len(result) == 0: break
        result = intersect(result, nxt)
    return result
```

Also write `union` and `difference` the same way, plus an optional `intersect_skip` with skip pointers. For the full-size run use the numpy equivalents (`np.intersect1d`, `np.union1d`, `np.setdiff1d`) and keep your hand-written versions as the reference. A test must assert both give identical output on 1,000 random pairs.

- **Phrase** `phrase_match(index, terms, zone)`: take the documents in `intersect_many` of the terms' postings, then for each document keep it if some position `p` of the first term has `p + j` among the positions of term `j` for every `j`. Use `np.isin` on the position arrays.
- **NEAR/k** `near(index, t1, t2, k, zone)`: within the documents containing both terms, keep a document if some pair of positions has `0 < |p1 - p2| <= k`. Unordered by default.

**B2. Query language and parser (hours 6 to 16).** Hand-write a tokenizer and a recursive-descent parser. Operators are only recognised in capitals (`AND`, `OR`, `NOT`, `NEAR/5`).

```text
or_expr  := and_expr ( "OR" and_expr )*
and_expr := unary ( "AND" unary )*
unary    := "NOT" unary | primary
primary  := "(" or_expr ")" | near | zoned | filter | phrase | term
near     := term "NEAR/" INT term
zoned    := ("title" | "abstract") ":" ( term | phrase )
filter   := FIELD OP VALUE          # year>=2020   journal:"lancet"   source:pmc
phrase   := '"' words '"'
```

The rule that decides everything: **a bare word is a ranking term and restricts nothing.** Anything with an operator, quotes, a zone prefix or a filter is a restriction. Restrictions at the top level are ANDed together. So:

| Query | Ranking terms | Restriction |
| --- | --- | --- |
| `vaccine efficacy` | vaccine, efficacy | none (all documents with at least one term are candidates) |
| `vaccine efficacy year>=2020` | vaccine, efficacy | year >= 2020 |
| `"contact tracing" AND mobile` | contact, trace, mobile | phrase AND term |
| `remdesivir NEAR/5 trial` | remdesivir, trial | the two within 5 positions |
| `title:remdesivir NOT hydroxychloroquine` | remdesivir | remdesivir in title, minus documents with the second term |

Rules for the details:

- Every word and phrase goes through `analyze()` before touching the index. A word that analyses to several tokens (for example `ace-2`) becomes a phrase. A word that analyses to nothing (a stop word) is dropped.
- A bare term with no zone uses the `all` zone. `title:` and `abstract:` use those zones.
- `ParsedQuery.terms` holds every analyzed term outside a NOT, de-duplicated in order, including terms inside phrases and NEAR. `phrases` holds the analyzed phrase term lists.
- Raise a `QuerySyntaxError` with a readable message for an unbalanced quote or parenthesis, a missing NEAR operand, or an unknown field. The CLI prints it instead of crashing.

**B3. Candidates (hours 12 to 22).** Evaluate the AST bottom-up into sorted id arrays.

- `Term` returns the zone's posting ids. `Phrase` and `Near` call B1. `Filter` calls `index.docs_where`.
- `And` intersects its positive children with `intersect_many` (shortest first) and then subtracts its `Not` children with `difference`. This avoids ever building the whole universe.
- A `Not` with no positive sibling means "everything except": use `np.arange(n_docs)` minus the child.
- No restriction anywhere in the query returns `None` (unrestricted).
- `use_champions=True` applies only when there is no restriction: return the union of each query term's champion list (`index.champion`), using the full posting list for any term where the champion is `None`. When there is a restriction, ignore champions, because the restriction already shrinks the set. C decides when to fall back to full postings (see Member C).

**B4. Plan printer (hours 16 to 24).** `describe(pq, index) -> str` returns a readable account of what the engine did: each operand with its document frequency, the order in which they were intersected (shortest first), and the size of each intermediate result. Example: `AND: mobile (df 1,204) -> phrase "contact trace" (df 410) -> 87 docs`. This is the evidence for the query-optimization point in the video.

**Done when:**

- [ ] Property test: the hand-written `intersect`, `union` and `difference` match numpy on 1,000 random sorted pairs.
- [ ] On the five-document test corpus, phrase and NEAR/k results are verified by hand, including a case where the terms are present but not adjacent.
- [ ] At least 15 example queries (the table above plus edge cases) have expected candidate sets in tests, and each syntax error produces a clear message.
- [ ] On the full index a phrase query and an AND query each return in a couple of seconds (a guideline, not a hard limit). Note the timings for the report.
- [ ] `describe()` output is readable enough to paste into the report.

## Member C: Ranking, authority and novelty

C owns everything that decides the order of results, including the two novelty ideas, so C carries the most weight in the rubric's IR-principles and novelty marks.

**Files owned:** `scoring.py`, `authority.py`, `gate.py`, `search.py`, `explain.py`, tests for these.

**C1. Zoned cosine scorer (hours 0 to 10).** Term-at-a-time lnc.ltc scoring, written against the `Index` API (use a stub index until A's real one lands).

```python
def cosine_scores(index, zone, terms, candidates=None):
    N = index.n_docs
    qw = {}
    for t, tf in Counter(terms).items():            # query side: ltc = log tf, idf, cosine
        df = index.df(t, zone)
        if df: qw[t] = (1 + log10(tf)) * log10(N / df)
    qnorm = sqrt(sum(w * w for w in qw.values())) or 1.0
    acc = np.zeros(N)
    for t, wq in qw.items():
        docs, tfs = index.postings(t, zone)
        acc[docs] += (wq / qnorm) * (1 + np.log10(tfs))   # doc side: lnc = log tf, no idf
    acc /= index.norms(zone)                              # divide by the document's cosine norm
    if candidates is not None:
        mask = np.zeros(N, bool); mask[candidates] = True; acc[~mask] = 0
    return acc
```

Notes:

- `norms(zone)` and `lengths(zone)` are already in the `Index` stub. Changing an existing signature is not allowed.
- Zone combination: `net = sum(w_z * cos_z) + beta * g(d)`. Start with weights title 2 and abstract 1, normalized to sum to 1, and let D tune them.
- Top-K uses a heap: `heapq.nlargest(k, ...)` over the documents with a nonzero score, with ties broken by smaller document id so runs are reproducible. Show this in the video.
- Cosine values are small, often well below 0.5, so the authority weight beta must be small too. D should grid-search values like 0, 0.02, 0.05, 0.1 and 0.2.

**C2. BM25 reference (hours 8 to 14).** Implement it yourself on the `all` zone, using `lengths` and `avg_len`. Defaults k1 = 1.2, b = 0.75. As a bug check, compare the top-10 overlap with `rank_bm25` on a handful of queries (their idf variant differs slightly, so expect close, not identical).

```latex
\mathrm{BM25}(q,d) = \sum_{t \in q} \ln\!\Big(1 + \frac{N - df_t + 0.5}{df_t + 0.5}\Big) \cdot \frac{tf_{t,d}\,(k_1 + 1)}{tf_{t,d} + k_1\,(1 - b + b\,|d| / \mathrm{avgdl})}
```

**C3. Authority g(d) and age normalization, N1 (hours 2 to 20).**

Hour-2 decision: where do citation counts come from? Try in this order and stop at the first that works.

1. **OpenAlex API.** Look papers up in batches by DOI or PubMed ID. Its filter accepts up to 100 values joined with `|` per request, with `per_page=100`, and returns `cited_by_count` per work. Cache every response to `data/citations.jsonl` so you never repeat a call. Check OpenAlex's current rate-limit, key and contact-email guidance before the first run, and check the exact identifier format it expects for PubMed IDs.
2. **Semantic Scholar API.** An alternative if OpenAlex coverage is poor. We have not verified its current batch limits or key requirements, so read its docs first.
3. **Fallback if neither works in time:** use recency as the static quality score (a newer publish date gets a higher g(d)). It is a legitimate static quality signal, and the report simply says the authority signal was recency. Dropping authority entirely is the last resort.

Report coverage: the share of documents with a count. Documents without a count get g = 0 and a flag in the explainer.

```python
# raw authority
g_raw = np.log1p(c) / np.log1p(c.max())

# N1: percentile of the citation count within the paper's publication-month cohort
df["cohort"] = df.publish_month                      # "YYYY-MM"
df.loc[df.groupby("cohort").cohort.transform("size") < 30, "cohort"] = df.year   # merge tiny months into the year
g_cohort = df.groupby("cohort").citations.rank(pct=True, method="min")
g_cohort = g_cohort.fillna(df.citations.rank(pct=True, method="min"))  # unknown date: global percentile
```

Use `method="min"` so the large block of zero-citation papers all get the lowest percentile and not an average rank. Save both arrays as `index/authority_raw.npy` and `index/authority_cohort.npy`, aligned with internal ids, and ask A to expose them through `Index.authority(doc, mode)`.

**C4. Query-adaptive authority gate, N2 (hours 18 to 26).**

```python
def specificity(index, terms):                     # in [0, 1]
    idfs = [log10(index.n_docs / index.df(t, "all")) / log10(index.n_docs)
            for t in terms if index.df(t, "all") > 0]
    return sum(idfs) / len(idfs) if idfs else 0.0

def beta_of_q(s, beta_max, s_lo, s_hi):            # high beta for vague queries, low for specific ones
    return beta_max * min(1.0, max(0.0, (s_hi - s) / (s_hi - s_lo)))
```

Before tuning anything, print the specificity of all 50 topic queries and plot the histogram. If it barely varies across topics, the gate cannot work, and the honest result is to say so. Otherwise pick `s_lo` and `s_hi` from that spread and let D tune `beta_max`, `s_lo` and `s_hi` on the tuning half only.

**C5. Per-query zone weights, N3, stretch (hours 24 to 28).** For each zone, measure how much of the query's idf mass appears in that zone: `c_z = sum(idf_z(t) / log10(N) for t in terms if df_z(t) > 0)`. The adaptive weight is `w_z(q) = (1 - alpha) * w_z_global + alpha * c_z / sum(c)`. D tunes `alpha` over 0, 0.25, 0.5, 0.75, 1. Skip this if Gate 3 is near.

**C6. `search()` orchestration (hours 6 to 20).**

```python
def search(query, k=10, variant="V2"):
    cfg = VARIANTS[variant]
    pq = parse(query)
    cand = candidates(pq, index, use_champions=cfg.champions)
    cos = {z: cosine_scores(index, z, pq.terms, cand) for z in cfg.zone_weights}
    w = adaptive_weights(...) if cfg.adaptive_zones else cfg.zone_weights
    beta = beta_of_q(specificity(index, pq.terms), ...) if cfg.gate else cfg.beta
    net = sum(w[z] * cos[z] for z in cos) + beta * authority_array(cfg.authority_mode)
    if cfg.champions and (net > 0).sum() < k:                 # tier fallback: champions found too little
        return search_with(cfg, use_champions=False)           # rescore on the full postings
    # optional phrase boost (V4): re-score only the top 200 for adjacent query-term pairs
    return top_k_with_explanations(net, k)
```

BM25 variants (`cfg.scorer == "bm25"`) bypass the zone combination and score the `all` zone. Documents that score zero on every zone must not be returned, even if authority is positive.

**C7. Explainer (hours 16 to 24).** `explain(result, pq, index) -> str` prints, for one result and each zone, every query term with its tf, df, idf, query weight, document weight and product (these sum to the zone cosine), then the weighted zone sum, `g(d)`, `beta(q)` and the net score. This is the most important thing to show in the video.

**Also expose** `score_components(query)`, returning the per-zone cosine arrays and the authority array for one query, so D can cache them for tuning.

**Done when:**

- [ ] On a three-document corpus, the cosine scores match a hand calculation written in a code comment, and BM25 top-10 overlaps `rank_bm25` on test queries.
- [ ] `g` values lie in [0, 1], increase with citation count within a cohort, and the cohort percentile test passes on synthetic data.
- [ ] The hour-2 decision is recorded in the repo README with the coverage figure.
- [ ] The specificity histogram exists, and the gate is monotone: a more specific query never gets a larger beta.
- [ ] Every variant in `config.py` runs through `search()` end to end, and the explainer reproduces a result's score to within rounding.

## Member D: Evaluation and delivery

D makes every other person's work measurable and turns it into the submission, so D's harness must be running by hour 4 and D owns the README, report and video assembly.

**Files owned:** `eval/metrics.py`, `eval/runner.py`, `eval/tune.py`, `eval/plots.py`, `scripts/run_eval.py`, `README.md`, `AI_USE.md`, the report, and the video edit.

**D1. Metrics and data (hours 0 to 4).** Load topics and qrels with the `beir` loader (`GenericDataLoader(...).load(split="test")`).

- "Relevant" means qrels score at least 1. Fix this now and never change it later. Check the score values that actually appear in the qrels (the BEIR loader may give graded scores) and write them in the README.
- P@10: relevant documents in the top 10, divided by 10.
- nDCG@10 with graded gains: gain equals the qrels score, discount `1 / log2(rank + 1)`, normalized by the ideal ordering of that topic's judged documents.
- Recall@100: relevant documents in the top 100 divided by the number of relevant documents for the topic. **Check the ceiling first:** print the number of relevant documents per topic. The dataset card reports a relevance density of 493.5 per query on average, and if the real figure is in the hundreds, Recall@100 cannot be close to 1, because it is capped at 100 divided by that count. Report it with that caveat.
- Test each metric on a toy qrels file whose answers you compute by hand. Optionally cross-check against `pytrec_eval` on a few topics.

**D2. Runner (hours 2 to 6).** `run_variant(variant, queries, k=100)` returns for each topic the ranked list of (doc id, score). Save outputs so nothing is recomputed:

- a TREC-format run file, `runs/V2.run`, with lines `qid Q0 docid rank score tag`
- per-topic metrics in `results/per_topic_V2.csv`
- latency per query, from which the table reports the median and the 95th percentile

Until A's index exists, validate the harness against a placeholder baseline (scikit-learn `TfidfVectorizer` on a small slice, or `rank_bm25`). Gate 1 needs real V0 numbers, but a validated harness by hour 2 means the numbers arrive the moment the `all` zone is pushed. As a sanity check on the whole pipeline, compare your BM25 number against the BEIR paper's published BM25 result for TREC-COVID. Yours should be in the same neighbourhood, not far below. If it is far below, something is wrong in the analyzer, the index, or the metric.

**D3. Split and tuning (hours 4 to 8 to define, hours 20 to 28 to run).**

```python
qids = sorted(queries); random.Random(42).shuffle(qids)
tune, report = qids[:25], qids[25:]          # write to eval/split.json, commit, never change
```

Tune only on `tune`, report on `report` (and also show all 50). Tune in sequence so the grid stays small:

1. V1: title weight in {1, 2, 3, 4} with abstract fixed at 1.
2. V2: beta in {0, 0.02, 0.05, 0.1, 0.2}.
3. V6 (gate): `beta_max`, `s_lo`, `s_hi` on a coarse grid chosen from C's specificity histogram.
4. V7 (if built): `alpha` in {0, 0.25, 0.5, 0.75, 1}.
5. V3: champion size r in {200, 500, 1000, 2000}, choosing by quality loss versus speed.

To make this fast, use C's `score_components(query)`: cache the per-zone cosine arrays and authority array for the 25 tuning topics once and re-weight them in memory, because the net score is linear in them. No grid point then needs to re-run the index.

**D4. Results and plots (hours 20 to 32).**

- Results table: P@10, nDCG@10, Recall@100 for V0 to V7 and BM25, on the reporting half and on all 50 topics.
- Bar chart of nDCG@10 by variant.
- Per-topic plot of V2 minus V0 (sorted differences), showing wins and losses, not only the mean.
- Latency table for the champion-list variant against the full postings.
- A scatter of query specificity against the per-topic gain of authority. This is the picture that tells the N2 story, whether it works or not.
- Because there are only about 25 reporting topics, give a paired bootstrap confidence interval (or a sign test) for each variant against V0 and avoid claiming small differences:

```python
diffs = np.array(per_topic_v - per_topic_v0)
boot = [rng.choice(diffs, len(diffs)).mean() for _ in range(10000)]
ci = np.percentile(boot, [2.5, 97.5])
```

**D5. Delivery (hours 24 to 36).**

- **README:** what PRISM is, setup commands, exactly how to download the data, how to build the index, how to run one query with `--explain`, how to run the evaluation, the results table, what works and what is still planned, dataset credits (TREC-COVID, CORD-19, BEIR), and a pointer to `AI_USE.md`.
- **`scripts/demo.sh`:** one command that runs the fixed demo queries so the video is reproducible.
- **Report** (at most 8 pages): follow the 7-part structure from the assignment. Use the architecture as the pipeline diagram, add the work-division note and the AI-use declaration built from `AI_USE.md`.
- **Video** (5 to 8 minutes, no slides): problem and track fit (up to 1 minute), live queries including one limitation, the pipeline with real intermediate output (postings, the `describe()` plan, the `explain()` breakdown), the results table and graphs, and each member explaining their component over their own module's output (about a minute each). Everyone records their segment separately and D stitches them.

**Done when:**

- [ ] Metrics pass hand-checked toy tests, and the BM25 sanity check against the published number is done.
- [ ] `python scripts/run_eval.py --variant V0` prints the three metrics and writes the run file and per-topic CSV. This is the Gate 1 deliverable.
- [ ] The split file is committed once and never edited.
- [ ] A single command reproduces the final results table.
- [ ] The README lets someone who has never seen the repo reproduce the demo on a clean machine (test it on a teammate's laptop).

## Integration, testing and demo

Integration is where hackathon projects fail, so the four modules meet at the fixed checkpoints in the Quick reference table and an end-to-end test runs the whole path on a tiny corpus from hour 6 onward.

**Gates.**

1. **Hour 2 check (C with A).** Do citation counts and publication dates exist for enough papers? If yes, build N1 as planned. If no, use the fallback in C3 and say so in the report.
2. **Gate 1, hour 4.** V0 runs on all 50 topics and its P@10, nDCG@10 and Recall@100 are written down. No structure work starts before this exists.
3. **Gate 2, hour 20.** V2 runs end to end through `search()` with the explainer. If it does not, stop adding features and fix it.
4. **Gate 3, hour 28.** Feature freeze. The last eight hours are for final numbers, the README, the report and the video.

**Working agreements.**

- Each person edits only their own files. If you need a change in someone else's module, ask them or make a small pull request they review.
- Run `pytest` before every merge to `main`. Re-run the evaluation harness after any change to scoring, the analyzer or the index, and paste the new V0 and V2 numbers in the team chat so regressions are caught at once.
- If the analyzer or index format changes, A tells everyone and rebuilds, because old indexes are no longer valid.
- When blocked on someone, work against the stub and keep going. Do not wait.
- Merge to `main` at every gate. Keep `AI_USE.md` current from the first hour. Record the exact build and run commands, since the README must reproduce the demo.

**Test checklist.**

- [ ] **Tiny-corpus end-to-end test** (`tests/test_end_to_end.py`): build an index over a handful of hand-written documents, run five queries covering a term query, a phrase, a Boolean query, a zone query and a filter, and assert the expected top document for each.
- [ ] **Postings invariants:** ascending document ids, positive term frequencies, positions sorted, `df` equals the posting-list length.
- [ ] **Operations versus numpy:** intersection, union and difference match the numpy equivalents on random input.
- [ ] **Scores:** the cosine for a hand-computed three-document example, and BM25 top-10 overlap with `rank_bm25`.
- [ ] **Explainer consistency:** the sum of per-term contributions equals the zone cosine, and the final net score equals `search()`'s score.
- [ ] **Determinism:** the same query returns identical results twice, including ties.
- [ ] **Subset smoke test:** the whole pipeline on the 30K subset finishes and produces nonzero metrics.

**Demo queries for the video.** Choose the exact queries from real runs, and do not script expected outputs, because a faked or hard-coded demo scores zero for the working system. Use this order and fill in the specific queries during tuning:

| Step | Shows | Pick a query that |
| --- | --- | --- |
| 1 | Ranked retrieval with the explainer | Is a plain two to three word topic query |
| 2 | Phrase plus Boolean, with the `describe()` plan | Combines a quoted phrase with AND |
| 3 | Proximity | Uses NEAR/k between two terms |
| 4 | Zone and metadata filter | Uses `title:` with a year filter |
| 5 | Authority helping | Is a broad query where gated beta is high |
| 6 | The gate lowering beta | Is a very specific query |
| 7 | A limitation, live | Is one where authority promotes a well-cited but weaker paper. Find it in the per-topic losses from D |

---

# Part 3: Kickoff prompt for coding assistants

Paste the master prompt into your coding assistant at the start of every session, then add the one role paragraph for whoever is working. The master prompt carries the rules and interfaces so the assistant cannot drift from the shared design. Make sure the hour-0 commit contains the interface stubs exactly as written in Shared contracts, since the prompt tells the assistant not to change them.

## Master prompt

```markdown
You are helping a four-person student team build PRISM, a classical information-retrieval search engine, in about 36 hours for a university IR hackathon (course CSD358, Track T6: vertical search for science). Work like a careful senior engineer: small steps, tests for core logic, and ask before changing any interface.

GOAL
Build a structure-aware search engine over COVID-19 research papers. Data: the BEIR copy of TREC-COVID (171,332 papers with fields _id, title, text where text is the abstract; 50 topics with relevance judgments). The engine ranks using document structure (title and abstract zones, exact phrases, metadata filters) plus an authority score, and is evaluated against flat tf-idf and BM25 on the 50 topics with P@10, nDCG@10 and Recall@100. No UI beyond a command-line interface.

HARD RULES
1. The core engine (analyzer, inverted index, query parser, Boolean operations, scoring, top-K) is written by us in Python using numpy, pandas and nltk. Do NOT use Elasticsearch, Lucene, Pyserini, FAISS, vector databases or embeddings in the core engine. rank_bm25 and pytrec_eval may appear only in tests as cross-checks.
2. Everything runs for real on real data. No hard-coded or faked outputs, ever.
3. Do not change the shared interfaces below without asking the human. Adding a method is fine; changing a signature is not.
4. Keep runs deterministic (fixed seeds). No network access in tests. Never commit data/ or index/.
5. Record every AI-assisted change in AI_USE.md (file, what changed, which tool).

REPO LAYOUT
prism/ (schema, config, loader, analyzer, indexer, index, parser, boolean_ops, candidates, scoring, authority, gate, search, explain), eval/ (metrics, runner, tune, plots), scripts/ (build_index, run_search, run_eval), tests/.

SHARED INTERFACES
- Record: doc_id (str, the id used in the qrels), title, abstract, body (None for now), year, journal, source, pubmed_id, doi, publish_month (YYYY-MM), citations (all optional except the first three).
- Result: doc_id, score, zone_scores (dict), authority, beta, matched_terms, title.
- analyze(text) -> list[str]: lowercase; normalize domain spellings (covid-19 / covid 19 -> covid19, sars-cov-2 -> sarscov2); tokenize with [a-z0-9]+; drop NLTK English stop words and one-character tokens; Porter-stem. A token's position is its index in the returned list.
- Index (internal doc ids are ints 0..N-1): n_docs, zones [title, abstract, all], doc_id(i), internal_id(doc_id), postings(term, zone) -> (ascending doc ids, tfs) numpy arrays, positions(term, zone, doc), df(term, zone), doc_norm(doc, zone), doc_len(doc, zone), avg_len(zone), field_value(doc, name), docs_where(name, op, value) -> sorted ids, champion(term, zone) -> top-r ids or None, authority(doc, mode) with modes raw and cohort, title(doc).
- parse(query) -> ParsedQuery(raw, terms, phrases, ast); candidates(pq, index, use_champions) -> sorted internal ids, or None meaning unrestricted.
- search(query, k=10, variant=V2) -> list[Result]. variant names a VariantConfig in config.py with: zone_weights, scorer (cosine or bm25), authority_mode (none, raw, cohort), beta, gate, adaptive_zones, champions, phrase_boost.

QUERY SEMANTICS
Bare words are ranking terms only. Quoted phrases, AND, OR, NOT, NEAR/k, title: or abstract: prefixes, and field filters such as year>=2020 are restrictions. Operators are capitalized. A query with no restriction ranks every document containing at least one query term.

VARIANTS
V0 flat tf-idf (lnc.ltc, zone all); V1 zone-weighted tf-idf; V2 V1 plus raw authority; V3 V2 with champion lists; V4 V2 with phrase boost; V5 V2 with age-normalized authority (N1); V6 V5 with a query-adaptive authority weight (N2); V7 V6 with per-query zone weights (N3, stretch); R = BM25 reference.

NOVELTY
N1: authority g(d) is the percentile of a paper's citation count within its publication-month cohort, not the raw count. N2: the authority weight beta falls as the query's mean normalized idf (its specificity) rises. N3: zone weights adapt to how much of the query's idf mass appears in each zone.

HOW TO WORK
- First read the existing code and tests, then say in two or three sentences what you plan to do.
- Write the test for core logic before or alongside the code, using tiny hand-checkable examples.
- Run the tests, then report: what you built, what passes, what is blocked, and any open question for the human. Do not guess at a missing decision; ask.
- Keep functions small and typed, with short docstrings. Prefer clear code over clever code, because every member must be able to explain it in a video.
```

## Role add-ons

Add one of these to the master prompt.

```markdown
ROLE A (data and index): You own loader.py, analyzer.py, indexer.py, index.py and scripts/build_index.py. Priority order: (1) the loader joining the BEIR corpus to CORD-19 metadata.csv on cord_uid and a data_stats script; (2) the analyzer with unit tests; (3) an index of the all zone with doc_norm and doc_len, saved to index/ and loadable through the Index class; (4) title and abstract zones with positions; (5) the parametric index; (6) champion lists; (7) a --subset N option that keeps all judged documents. Use array.array for per-term lists while building to save memory, and store positions as one flat array plus offsets per term.

ROLE B (query engine): You own parser.py, boolean_ops.py and candidates.py. Priority order: (1) hand-written intersect, union and difference on sorted numpy arrays, plus intersect_many that processes the shortest list first, with a property test against numpy; (2) phrase_match and near using positions; (3) a recursive-descent parser with the grammar and the bare-word-versus-restriction rule, raising QuerySyntaxError for bad input; (4) candidates() including the champion option; (5) describe() that prints the operands, their document frequencies and the intersection order. Develop against a stub Index until real data arrives.

ROLE C (ranking and novelty): You own scoring.py, authority.py, gate.py, search.py and explain.py. Priority order: (1) lnc.ltc cosine per zone with term-at-a-time accumulation and heap top-K; (2) BM25 on the all zone; (3) search() with VariantConfig and the tier fallback; (4) authority from a citation-count source decided at hour 2 (OpenAlex or Semantic Scholar by DOI or PubMed ID, cached to disk; fallback is recency), raw and cohort-percentile versions; (5) the specificity function and the gate for N2, after printing the specificity histogram of the 50 topics; (6) explain() that reproduces each score term by term; (7) N3 if time allows. Expose score_components(query) so the tuning grid can reuse cached arrays.

ROLE D (evaluation and delivery): You own eval/, scripts/run_eval.py, README.md and AI_USE.md. Priority order: (1) metrics P@10, nDCG@10 with graded gains and Recall@100, tested on a toy qrels file; relevant means score at least 1; (2) the runner that writes TREC-format run files, per-topic CSVs and latency; validate it with a placeholder baseline before A's index exists; (3) the seeded 25/25 topic split saved to eval/split.json and never changed; (4) tuning on the tuning half only, in sequence, reusing cached score components; (5) the results table, nDCG@10 bar chart, per-topic win/loss plot, specificity-versus-gain scatter, and bootstrap confidence intervals; (6) README, demo.sh, and the report and video skeletons.
```
