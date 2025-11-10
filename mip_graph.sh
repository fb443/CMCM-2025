#!/usr/bin/env bash
set -euo pipefail

GRAPH="${1:-CMCM-2025/H.gpickle}"
FORMAT="${2:-gpickle}"   # gpickle|graphml|edgelist

# Model knobs
K="${K:-3}"
F="${F:-60,90,120,180}"      # minutes, comma-separated
TL="${TL:-120}"
MIPFOCUS="${MIPFOCUS:-1}"
THREADS="${THREADS:-0}"
OUTPKL="${OUTPKL:-results/memetic_results_from_mip_ithaca.pkl}"
WRITE_EDGES="${WRITE_EDGES:-results/edges_ithaca.csv}"

# Attribute names in H.gpickle
TIMEATTR="${TIMEATTR:-travel_time_sec}"       # seconds
DEADATTR="${DEADATTR:-deadhead_time_sec}"     # seconds
CYCATTR="${CYCATTR:-cycle_time_hr}"           # hours
FALLBACK="${FALLBACK:-weight}"                 # used if TIMEATTR missing
DEFAULT_TIME="${DEFAULT_TIME:-60.0}"           # seconds

if [[ ! -f "$GRAPH" ]]; then
  echo "error: graph not found: $GRAPH" >&2; exit 2
fi

python ithaca_gurobi_mip_carp.py \
  --graph "$GRAPH" \
  --graph-format "$FORMAT" \
  --time-attr "$TIMEATTR" \
  --deadhead-attr "$DEADATTR" \
  --cycle-attr "$CYCATTR" \
  --fallback-weight "$FALLBACK" \
  --default-time "$DEFAULT_TIME" \
  --write-edges "$WRITE_EDGES" \
  --K "$K" \
  --F "$F" \
  --time-limit "$TL" \
  --mipfocus "$MIPFOCUS" \
  --threads "$THREADS" \
  --out-pkl "$OUTPKL"
