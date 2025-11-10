module load gurobi
python ithaca_gurobi_mip_carp.py \
  --graph CMCM-2025/H.gpickle --graph-format gpickle \
  --require-min-cycle-min 1000 --require-max-cycle-min 1440 --autoname \
  --K 2 --F "1440" \
  --route-time-cap 240 \
  --alpha-dead 1.0 --route-penalty 20 \
  --time-limit 3600 --mipfocus 1 --threads 0
