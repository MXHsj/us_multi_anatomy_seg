#!/usr/bin/env bash
set -euo pipefail

output_dir="analysis/figures/qualitative"

model="ultrasam"
protocol="gt_bbox"
gpu_id="1"

# Datasets in the order they appear in datasets.json.
# datasets=$(python -c "import json; print(','.join(json.load(open('datasets/datasets.json'))['labels']))")
datasets=(mmotu,busbra,gist514,tnsc2020,ultrabones100k,umud,roblus)
echo $datasets

result_metric="dice"
lower_threshold="0"
upper_threshold="5"

# Number of cross-dataset figures to save.
samples_per_dataset="25"

# Leave empty to generate a new random seed each run; set to reproduce a prior run.
seed=""

pred_color="#ff4a43"
show_image_background="false"

echo "Generating qualitative figures..."
python analysis/qualitative_results_analysis.py \
  --output-dir "${output_dir}" --datasets "${datasets}" \
  --model "${model}" --protocol "${protocol}" \
  --result-metric "${result_metric}" \
  --lower-threshold "${lower_threshold}" --upper-threshold "${upper_threshold}" \
  --samples-per-dataset "${samples_per_dataset}" \
  --pred-color "${pred_color}" --show-image-background "${show_image_background}" \
  --gpu-id "${gpu_id}" \
  --seed "${seed}"
