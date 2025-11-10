#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via env or flags)
EDGES="${EDGES:-edges_conn_e40.csv}"
K="${K:-10}"                     # number of routes (upper bound)
F="${F:-30}"          # allowed caps (minutes)
TL="${TL:-120}"                  # time limit (seconds)
OUTPKL="${OUTPKL:-results_synth/memetic_results_from_mip.pkl}"
MIPFOCUS="${MIPFOCUS:-1}"
THREADS="${THREADS:-0}"

python gurobi_mip_carp.py \
  --edges "$EDGES" \
  --K "$K" --F "$F" \
  --time-limit "$TL" \
  --mipfocus "$MIPFOCUS" \
  --threads "$THREADS" \
  --out-pkl "$OUTPKL"
