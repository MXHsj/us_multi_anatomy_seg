# Quantitative Result Analysis

Use this folder for scripts that aggregate, compare, and visualize benchmark
outputs from `results/`.

Current scripts:

- `model_across_datasets.py`: compares the same model across datasets using
  summary-level metric mean/std values, and generates paper-style performance
  plots with mean bars, STD error bars, and per-sample scatter.
- `compare_models_same_dataset.py`: compares two models on the same dataset
  using matched sample IDs from `per_sample_metrics.csv`, and generates a
  grouped model-comparison plot with mean bars, STD error bars, and matched
  per-sample scatter.
- `quantiative_results_analysis.py`: joins per-sample benchmark results with
  `analysis/eda/<dataset>/sample_stats.csv` and plots result metrics against
  EDA measurements. By default it saves one non-log x-axis figure per EDA
  measurement; add `--log-x` to save `_logx` variants.

Both scripts understand the current benchmark metrics: `dice`, `iou`,
`precision`, `recall`, `specificity`, `balanced_accuracy`, `hd95`, `assd`,
and `relative_area_error`. Use `--metrics all` for the full metric set, or pass
a comma-separated subset such as `--metrics dice,iou,hd95,assd`.
