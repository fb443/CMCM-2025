module load gurobi
python ithaca_gurobi_mip_carp.py \
  --graph CMCM-2025/H_2_fixed.gpickle --graph-format gpickle \
  --require-min-cycle-min 50 --require-max-cycle-min 75 --autoname \
  --K 2 --F "75" \
  --route-time-cap 60 \
  --alpha-dead 1.0 --route-penalty 20 \
  --time-limit 3600 --mipfocus 1 --threads 0
