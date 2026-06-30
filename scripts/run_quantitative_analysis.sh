#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

output_dir="analysis/figures/results_vs_eda"

# Metrics to plot and their axis labels (same order, comma-separated).
# y-axis = result + derived metrics, x-axis = eda metrics.
result_metrics="dice" #,relative_area_error"
results_plot_labels="Dice" #,Error"
thres="1.0"
ultrabones_min_solidity="0.2"
filter_tnsc2020_target_bbox_one="true"
tnsc2020_min_aspect_ratio="0.2"

derived_metrics="" #fn_per_gt,fp_per_gt"
derived_plot_labels="FN / GT,FP / GT"

eda_metrics="target_area_fraction,target_bbox_area_ratio,solidity,circularity,aspect_ratio_feret"
eda_plot_labels="Target/image,Target/bbox,Solidity,Circularity,Aspect ratio"

# Datasets in the order they appear in datasets.json (columns left-to-right).
datasets=$(python -c "import json; print(','.join(json.load(open('datasets/datasets.json'))['labels']))")
# datasets="busbra"
echo $datasets

# What to generate, and which correlation coefficients the heatmaps use.
make_scatter="true"
make_heatmap="false"
correlations="pearson" #spearman"

# Figure file format: png, svg, or pdf.
plot_format="png"

# Data granularity before plotting:
#   false -> plots each bounding box separately.
#   true  -> aggregates results metrics per image using a target-area-weighted mean.
per_image="false"

# How to prepare the data before plotting:
#   false -> combine all images into one figure (scatter and heatmap).
#   true  -> one figure per dataset (the same plot, repeated per dataset).
per_dataset="true"

# Use a log-scaled x-axis (true/false).
log_x="false"

# Generate the results-vs-EDA figures.
echo "Generating results-vs-EDA figures..."
python analysis/quantitative_results_analysis.py \
  --output-dir "${output_dir}" --datasets "${datasets}" \
  --result-metrics "${result_metrics}" --result-labels "${results_plot_labels}" \
  --thres "${thres}" \
  --ultrabones-min-solidity "${ultrabones_min_solidity}" \
  --filter-tnsc2020-target-bbox-one "${filter_tnsc2020_target_bbox_one}" \
  --tnsc2020-min-aspect-ratio "${tnsc2020_min_aspect_ratio}" \
  --derived-metrics "${derived_metrics}" --derived-labels "${derived_plot_labels}" \
  --eda-metrics "${eda_metrics}" --eda-labels "${eda_plot_labels}" \
  --per-image "${per_image}" --per-dataset "${per_dataset}" --plot-format "${plot_format}" \
  --scatter "${make_scatter}" --heatmap "${make_heatmap}" --correlations "${correlations}" \
  --log-x "${log_x}"
