#!/usr/bin/env bash
# D: runs the fixed demo queries so the video is reproducible. Every step is a live search against
# the real index; nothing is scripted or cached. Needs index/ built (see README).
#
#   bash scripts/demo.sh            all steps
#   bash scripts/demo.sh 2 5        only steps 2 and 5
#
# The queries below are starting points. Final choices come from real runs: pick steps 5 to 7 from
# results/specificity.json and the per-topic wins and losses in results/per_topic_*.csv, then edit here.
set -euo pipefail
cd "$(dirname "$0")/.."

Q1="coronavirus origin"                            # plain topic query, with the explainer
Q2='"contact tracing" AND mobile'                  # phrase + Boolean, with the describe() plan
Q3='remdesivir NEAR/5 trial'                       # proximity
Q4='title:remdesivir year>=2020'                   # zone + metadata filter
Q5="covid symptoms"                                # broad query: gated beta should be high
Q6="remdesivir ACE2 binding"                       # very specific query: gated beta should be low
Q7=""                                              # LIMITATION: authority promotes a well-cited but weaker paper
V_AUTH=V6                                          # variant used for the authority steps (5 to 7)

run() { echo; echo "=== Step $1: $2"; shift 2; echo "\$ python scripts/run_search.py $*"; python scripts/run_search.py "$@"; }

STEPS=("$@")
[ "${#STEPS[@]}" -gt 0 ] || STEPS=(1 2 3 4 5 6 7)
has() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

has 1 && run 1 "ranked retrieval with the explainer" "$Q1" --explain -k 3
has 2 && run 2 "phrase + Boolean with the plan" "$Q2" --plan -k 5
has 3 && run 3 "proximity" "$Q3" -k 5
has 4 && run 4 "zone + metadata filter" "$Q4" -k 5
has 5 && run 5 "authority helping (broad query)" "$Q5" --variant "$V_AUTH" --explain -k 3
has 6 && run 6 "the gate lowering beta (specific query)" "$Q6" --variant "$V_AUTH" --explain -k 3
if has 7; then
  if [ -n "$Q7" ]; then
    run 7 "limitation: authority promotes a weaker paper" "$Q7" --variant "$V_AUTH" --explain -k 5
  else
    echo; echo "=== Step 7: set Q7 to a query from the per-topic losses (results/per_topic_*.csv) first"
  fi
fi
