#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

output_dir="analysis/figures/results_vs_eda"

# Metrics to plot and their axis labels (same order, comma-separated).
# y-axis = result + derived metrics, x-axis = eda metrics.
result_metrics="dice,relative_area_error"
results_plot_labels="Dice,Error"

derived_metrics="fn_per_gt,fp_per_gt"
derived_plot_labels="FN / GT,FP / GT"

eda_metrics="target_area_fraction,target_bbox_area_ratio,solidity,circularity,aspect_ratio_feret,convexity"
eda_plot_labels="Target/image,Target/bbox,Solidity,Circularity,Aspect ratio,Convexity"

# Datasets in the order they appear in datasets.json (columns left-to-right).
datasets=$(python -c "import json; print(','.join(json.load(open('datasets/datasets.json'))['labels']))")

# What to generate, and which correlation coefficients the heatmaps use.
make_scatter="false"
make_heatmap="true"
correlations="pearson,log_pearson,spearman"

# Figure file format: png, svg, or pdf.
plot_format="pdf"

# Pool every dataset's images into one cloud: scatter becomes a single overlay
# (dots colored per dataset) and the heatmap correlations are computed across all
# combined images instead of per dataset. Set to "true" for per-dataset figures.
per_dataset="false"

# Generate the results-vs-EDA figures (plain + heatmaps, then log-x scatters).
echo "Generating results-vs-EDA figures..."
python analysis/quantitative_results_analysis.py \
  --output-dir "${output_dir}" --datasets "${datasets}" \
  --result-metrics "${result_metrics}" --result-labels "${results_plot_labels}" \
  --derived-metrics "${derived_metrics}" --derived-labels "${derived_plot_labels}" \
  --eda-metrics "${eda_metrics}" --eda-labels "${eda_plot_labels}" \
  --per-dataset "${per_dataset}" --plot-format "${plot_format}" \
  --scatter "${make_scatter}" --heatmap "${make_heatmap}" --correlations "${correlations}"

echo "Generating results-vs-EDA figures (log-x)..."
python analysis/quantitative_results_analysis.py \
  --output-dir "${output_dir}" --datasets "${datasets}" \
  --result-metrics "${result_metrics}" --result-labels "${results_plot_labels}" \
  --derived-metrics "${derived_metrics}" --derived-labels "${derived_plot_labels}" \
  --eda-metrics "${eda_metrics}" --eda-labels "${eda_plot_labels}" \
  --per-dataset "${per_dataset}" --plot-format "${plot_format}" \
  --scatter "${make_scatter}" --heatmap "false" \
  --log-x
