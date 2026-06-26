#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

output_dir="analysis/figures/results_vs_eda"

# Metrics to plot and their axis labels (same order, comma-separated).
# y-axis = result + derived metrics, x-axis = eda metrics.
result_metrics="dice"
results_plot_labels="Dice"

derived_metrics="fn_per_gt,fp_per_gt"
derived_plot_labels="FN / GT,FP / GT"

eda_metrics="target_bbox_area_ratio"
eda_plot_labels="Target area / bbox area"

# Generate the results-vs-EDA figures (plain and log-x).
echo "Generating results-vs-EDA figures..."
python analysis/quantitative_results_analysis.py \
  --output-dir "${output_dir}" \
  --result-metrics "${result_metrics}" --result-labels "${results_plot_labels}" \
  --derived-metrics "${derived_metrics}" --derived-labels "${derived_plot_labels}" \
  --eda-metrics "${eda_metrics}" --eda-labels "${eda_plot_labels}"

echo "Generating results-vs-EDA figures (log-x)..."
python analysis/quantitative_results_analysis.py \
  --output-dir "${output_dir}" \
  --result-metrics "${result_metrics}" --result-labels "${results_plot_labels}" \
  --derived-metrics "${derived_metrics}" --derived-labels "${derived_plot_labels}" \
  --eda-metrics "${eda_metrics}" --eda-labels "${eda_plot_labels}" \
  --log-x
