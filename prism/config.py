"""Paths, analyzer spellings and variant configs. Every evaluated system is one key of VARIANTS."""
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

ZONES = ["title", "abstract", "all"]

# Domain spellings normalized by the analyzer before tokenizing (documented design choice).
# Regex patterns applied in order to lowercase text.
SPELLING_RULES: list[tuple[str, str]] = [
    (r"sars[-\s]?cov[-\s]?2", "sarscov2"),
    (r"covid[-\s]?19", "covid19"),
]

CHAMPION_R = 500        # champion list size per term (D tunes over 200/500/1000/2000)
SPLIT_SEED = 42         # topic split seed (eval/split.json)


@dataclass(frozen=True)
class VariantConfig:
    zone_weights: dict[str, float] = field(default_factory=lambda: {"all": 1.0})
    scorer: str = "cosine"              # "cosine" | "bm25"
    authority_mode: str = "none"        # "none" | "raw" | "cohort"
    beta: float = 0.0                   # fixed authority weight (ignored when gate is on)
    gate: bool = False                  # N2: beta(q) from query specificity
    s_lo: float = 0.2                   # specificity where gated authority starts decreasing
    s_hi: float = 0.8                   # specificity where gated authority reaches zero
    adaptive_zones: bool = False        # N3: per-query zone weights
    champions: bool = False             # tiered candidate generation
    phrase_boost: float = 0.0           # explicit quoted phrases only; 0 = off


_ZONED = {"title": 2 / 3, "abstract": 1 / 3}   # starting weights 2:1, tuned by D
_V1 = VariantConfig(zone_weights=_ZONED)
_V2 = replace(_V1, authority_mode="raw", beta=0.05)
_V5 = replace(_V2, authority_mode="cohort")
_V6 = replace(_V5, gate=True)

# Starting values only; real values come from tuning on the tuning half.
VARIANTS: dict[str, VariantConfig] = {
    "V0": VariantConfig(),                       # flat tf-idf (lnc.ltc), zone all
    "V1": _V1,                                   # zone-weighted tf-idf
    "V2": _V2,                                   # + raw authority
    "V3": replace(_V2, champions=True),          # + champion lists
    "V4": replace(_V2, phrase_boost=0.1),        # + phrase boost
    "V5": _V5,                                   # age-normalized authority (N1)
    "V6": _V6,                                   # + query-adaptive beta (N2)
    "V7": replace(_V6, adaptive_zones=True),     # + per-query zone weights (N3, stretch)
    "R": VariantConfig(scorer="bm25"),           # BM25 reference
}
