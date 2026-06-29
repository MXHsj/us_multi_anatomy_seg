#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# Models to compare (first model is the baseline for deltas) and protocol.
models="medsam,samus,ultrasam"
protocol="gt_bbox"

# Metrics to compare and plot (comma-separated, or 'all').
metrics="dice,assd" #,relative_area_error,hd95_norm,assd_norm"

# Per-dataset plot styles to generate (one figure each): bar, box, violin.
plot_types="box"

# Figure file format: png, svg, or pdf.
plot_format="pdf"

# Show outlier markers on box plots: true or false.
show_outliers="false"

# Metric panel layout: horizontal (side by side) or vertical (stacked).
orientation="horizontal"

output_csv="analysis/figures/compare_models_same_dataset_${protocol}_${models//,/_vs_}.csv"

echo "Comparing models (${models}) on ${protocol}..."
for plot_type in ${plot_types}; do
  python analysis/compare_models_same_dataset.py \
    --protocol "${protocol}" --models "${models}" \
    --metrics "${metrics}" --plot-type "${plot_type}" \
    --plot-format "${plot_format}" --show-outliers "${show_outliers}" \
    --orientation "${orientation}" \
    --output-csv "${output_csv}"
done
