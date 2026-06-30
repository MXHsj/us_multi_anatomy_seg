#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

datasets=(
  aulid
  blusg
  busbra
  busi
  camus
  mmotu
  oku
  roblus
  tnsc2020
  ultrabones100k
  umud
  uns
)

for dataset in "${datasets[@]}"; do
  if [[ "${dataset}" == "ultrabones100k" ]]; then
    python analysis/exploratory_data_analysis.py --dataset "${dataset}" --ultrabones-frame-fraction 0.05
  else
    python analysis/exploratory_data_analysis.py --dataset "${dataset}"
  fi
done
