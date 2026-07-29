#!/usr/bin/env bash
set -euo pipefail

datasets="busbra,gist514,mmotu,tnsc2020,ultrabones100k,umud,roblus"
metrics="dice,assd"
per_dataset="true"
formats="markdown,csv,latex"
output_prefix="analysis/tables/results_table"

python analysis/tabulate_results.py \
  --datasets "${datasets}" \
  --metrics "${metrics}" \
  --per-dataset "${per_dataset}" \
  --formats "${formats}" \
  --output-prefix "${output_prefix}"
