# Quantitative Result Analysis

Use this folder for scripts that aggregate, compare, and visualize benchmark
outputs from `results/`.

Current scripts:

- `model_across_datasets.py`: compares the same model across datasets using
  summary-level Dice and IoU mean/std values, and generates a paper-style
  performance plot with mean bars, STD error bars, and per-sample scatter.
- `compare_models_same_dataset.py`: compares two models on the same dataset
  using matched sample IDs from `per_sample_metrics.csv`, and generates a
  grouped model-comparison plot with mean bars, STD error bars, and matched
  per-sample scatter.
