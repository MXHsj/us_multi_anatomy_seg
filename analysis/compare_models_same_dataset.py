from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.metrics import METRIC_NAMES


DEFAULT_METRICS = tuple(METRIC_NAMES)
MODEL_COLORS = {
    "medsam": "#4C72B0",
    "samus": "#55A868",
    "ultrasam": "#C44E52",
}
MODEL_LABELS = {
    "medsam": "MedSAM",
    "samus": "SAMUS",
    "ultrasam": "UltraSAM",
}
MODEL_ALIASES = {
    "ultrasm": "ultrasam",
    "ultra-sm": "ultrasam",
    "ultra_sam": "ultrasam",
}
DATASET_LABELS = {
    "aulid": "AULID",
    "blusg": "BLUSG",
    "busbra": "BUS-BRA",
    "busi": "BUSI",
    "camus": "CAMUS",
    "oku": "OKU",
    "roblus": "RobLUS",
    "tnsc2020": "TNSC2020",
    "ultrabones100k": "UltraBones",
    "uns": "UNS",
}


def parse_result_dir_name(name: str) -> tuple[str, str, str] | None:
    parts = name.split("_")
    if len(parts) < 4:
        return None

    model = parts[0]
    if parts[1:3] == ["gt", "bbox"]:
        protocol = "gt_bbox"
        dataset = "_".join(parts[3:])
    elif parts[1:3] == ["jitter", "bbox"]:
        protocol = "jitter_bbox"
        dataset = "_".join(parts[3:])
    else:
        return None

    return model, protocol, dataset


def parse_metrics_arg(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return DEFAULT_METRICS
    metrics = tuple(metric.strip() for metric in value.split(",") if metric.strip())
    unknown = sorted(set(metrics) - set(DEFAULT_METRICS))
    if unknown:
        raise SystemExit(
            f"Unsupported metric(s): {', '.join(unknown)}. "
            f"Expected one of: {', '.join(DEFAULT_METRICS)}"
        )
    if not metrics:
        raise SystemExit("At least one metric must be selected.")
    return metrics


def normalize_model_name(value: str) -> str:
    model = value.strip().lower()
    return MODEL_ALIASES.get(model, model)


def parse_models_arg(value: str) -> tuple[str, ...]:
    models = tuple(
        normalize_model_name(model)
        for model in value.split(",")
        if model.strip()
    )
    if not models:
        raise SystemExit("At least one model must be selected.")
    if len(set(models)) != len(models):
        raise SystemExit("Model names must be unique.")
    return models


def read_metrics(csv_path: Path, metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row.get("sample_id", "")
            if not sample_id:
                continue
            try:
                values = {metric: float(row[metric]) for metric in metrics}
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in values.values()):
                rows[sample_id] = values
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def discover_results(results_dir: Path, protocol: str) -> dict[tuple[str, str], Path]:
    discovered: dict[tuple[str, str], Path] = {}
    for metrics_path in sorted(results_dir.glob("*/per_sample_metrics.csv")):
        parsed = parse_result_dir_name(metrics_path.parent.name)
        if parsed is None:
            continue
        model, result_protocol, dataset = parsed
        if result_protocol != protocol:
            continue
        discovered[(model, dataset)] = metrics_path
    return discovered


def compare_models(
    results_dir: Path,
    protocol: str,
    model_a: str,
    model_b: str,
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    discovered = discover_results(results_dir, protocol)
    datasets = sorted(
        dataset
        for result_model, dataset in discovered
        if result_model == model_a and (model_b, dataset) in discovered
    )

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        metrics_a = read_metrics(discovered[(model_a, dataset)], metrics)
        metrics_b = read_metrics(discovered[(model_b, dataset)], metrics)
        shared_ids = sorted(set(metrics_a) & set(metrics_b))
        if not shared_ids:
            continue

        row: dict[str, Any] = {"dataset": dataset}
        values_by_metric: dict[str, dict[str, list[float]]] = {}
        for metric in metrics:
            values_a = [metrics_a[sample_id][metric] for sample_id in shared_ids]
            values_b = [metrics_b[sample_id][metric] for sample_id in shared_ids]
            deltas = [b - a for a, b in zip(values_a, values_b)]
            values_by_metric[metric] = {
                model_a: values_a,
                model_b: values_b,
            }

            mean_a, std_a = mean_std(values_a)
            mean_b, std_b = mean_std(values_b)
            delta_mean, delta_std = mean_std(deltas)

            row[f"{model_a}_{metric}_mean"] = mean_a
            row[f"{model_a}_{metric}_std"] = std_a
            row[f"{model_b}_{metric}_mean"] = mean_b
            row[f"{model_b}_{metric}_std"] = std_b
            row[f"{model_b}_minus_{model_a}_{metric}_mean"] = delta_mean
            row[f"{model_b}_minus_{model_a}_{metric}_std"] = delta_std

        row["_values_by_metric"] = values_by_metric
        rows.append(row)

    return rows


def compare_model_set(
    results_dir: Path,
    protocol: str,
    models: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    discovered = discover_results(results_dir, protocol)
    candidate_datasets = sorted(
        {
            dataset
            for result_model, dataset in discovered
            if result_model in models
        }
    )
    datasets = [
        dataset
        for dataset in candidate_datasets
        if all((model, dataset) in discovered for model in models)
    ]

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        model_metrics = {
            model: read_metrics(discovered[(model, dataset)], metrics)
            for model in models
        }
        shared_ids = sorted(set.intersection(*(set(values) for values in model_metrics.values())))
        if not shared_ids:
            continue

        row: dict[str, Any] = {"dataset": dataset, "matched_samples": len(shared_ids)}
        values_by_metric: dict[str, dict[str, list[float]]] = {}
        for metric in metrics:
            values_by_metric[metric] = {}
            for model in models:
                values = [
                    model_metrics[model][sample_id][metric]
                    for sample_id in shared_ids
                ]
                values_by_metric[metric][model] = values
                mean, std = mean_std(values)
                row[f"{model}_{metric}_mean"] = mean
                row[f"{model}_{metric}_std"] = std

            baseline = models[0]
            baseline_values = values_by_metric[metric][baseline]
            for model in models[1:]:
                deltas = [
                    value - baseline_value
                    for baseline_value, value in zip(
                        baseline_values,
                        values_by_metric[metric][model],
                    )
                ]
                delta_mean, delta_std = mean_std(deltas)
                row[f"{model}_minus_{baseline}_{metric}_mean"] = delta_mean
                row[f"{model}_minus_{baseline}_{metric}_std"] = delta_std

        row["_values_by_metric"] = values_by_metric
        rows.append(row)

    return rows


def format_float(value: float) -> str:
    return f"{value:.4f}"


def column_names(model_a: str, model_b: str, metrics: tuple[str, ...]) -> list[str]:
    columns = ["dataset"]
    for metric in metrics:
        columns.extend(
            [
                f"{model_a}_{metric}_mean",
                f"{model_a}_{metric}_std",
                f"{model_b}_{metric}_mean",
                f"{model_b}_{metric}_std",
                f"{model_b}_minus_{model_a}_{metric}_mean",
                f"{model_b}_minus_{model_a}_{metric}_std",
            ]
        )
    return columns


def model_set_column_names(models: tuple[str, ...], metrics: tuple[str, ...]) -> list[str]:
    columns = ["dataset", "matched_samples"]
    baseline = models[0]
    for metric in metrics:
        for model in models:
            columns.extend([f"{model}_{metric}_mean", f"{model}_{metric}_std"])
        for model in models[1:]:
            columns.extend(
                [
                    f"{model}_minus_{baseline}_{metric}_mean",
                    f"{model}_minus_{baseline}_{metric}_std",
                ]
            )
    return columns


def print_markdown(rows: list[dict[str, Any]], columns: list[str]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [
            str(row[column])
            if column in {"dataset", "matched_samples"}
            else format_float(row[column])
            for column in columns
        ]
        print("| " + " | ".join(values) + " |")


def write_csv(rows: list[dict[str, Any]], columns: list[str], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def default_plot_path(protocol: str, model_a: str, model_b: str) -> Path:
    return (
        Path("analysis")
        / "figures"
        / f"compare_models_same_dataset_{protocol}_{model_a}_vs_{model_b}.png"
    )


def default_model_set_plot_path(protocol: str, models: tuple[str, ...]) -> Path:
    model_slug = "_vs_".join(models)
    return (
        Path("analysis")
        / "figures"
        / f"compare_models_same_dataset_{protocol}_{model_slug}.png"
    )


def display_dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.upper())


def display_model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def metric_label(metric: str) -> str:
    labels = {
        "dice": "Dice",
        "iou": "IoU",
        "precision": "Precision",
        "recall": "Recall",
        "specificity": "Specificity",
        "balanced_accuracy": "Balanced accuracy",
        "hd95": "HD95 (px)",
        "assd": "ASSD (px)",
        "relative_area_error": "Relative area error",
    }
    return labels.get(metric, metric)


def plot_rows(
    rows: list[dict[str, Any]],
    plot_path: Path,
    model_a: str,
    model_b: str,
    metrics: tuple[str, ...],
    title: str = "",
) -> None:
    plot_multi_model_rows(
        rows=rows,
        plot_path=plot_path,
        models=(model_a, model_b),
        metrics=metrics,
        title=title,
    )


def plot_multi_model_rows(
    rows: list[dict[str, Any]],
    plot_path: Path,
    models: tuple[str, ...],
    metrics: tuple[str, ...],
    title: str = "",
) -> None:
    mpl_config_dir = Path("analysis") / ".mplconfig"
    xdg_cache_dir = Path("analysis") / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir.resolve()))

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
        }
    )

    labels = [display_dataset_label(row["dataset"]) for row in rows]
    x_positions = np.arange(len(rows))
    group_width = 0.74
    bar_width = min(0.24, group_width / len(models))
    start_offset = -bar_width * (len(models) - 1) / 2
    offsets = {
        model: start_offset + idx * bar_width
        for idx, model in enumerate(models)
    }
    rng = np.random.default_rng(20240515)

    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig_width = max(9.5, len(rows) * 0.85, ncols * 4.2)
    fig_height = max(3.25, nrows * 2.75)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for axis, metric in zip(axes_flat, metrics):
        for model in models:
            means = [row[f"{model}_{metric}_mean"] for row in rows]
            stds = [row[f"{model}_{metric}_std"] for row in rows]
            color = MODEL_COLORS.get(model, "#666666")
            model_positions = x_positions + offsets[model]

            axis.bar(
                model_positions,
                means,
                width=bar_width,
                color=color,
                alpha=0.32,
                edgecolor=color,
                linewidth=1.1,
                zorder=2,
                label=display_model_label(model),
            )
            axis.errorbar(
                model_positions,
                means,
                yerr=stds,
                fmt="none",
                ecolor="#222222",
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                zorder=4,
            )

            for idx, row in enumerate(rows):
                values = row["_values_by_metric"][metric][model]
                if not values:
                    continue
                jitter = rng.uniform(-bar_width * 0.27, bar_width * 0.27, size=len(values))
                axis.scatter(
                    np.full(len(values), model_positions[idx]) + jitter,
                    values,
                    s=3,
                    color=color,
                    alpha=0.14,
                    linewidths=0,
                    zorder=3,
                )

        if metric == "relative_area_error":
            axis.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
        if metric in {"dice", "iou", "precision", "recall", "specificity", "balanced_accuracy"}:
            axis.set_ylim(0.0, 1.02)
        axis.set_title(metric_label(metric))
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_ylabel(metric_label(metric))
        axis.grid(axis="x", visible=False)
        axis.set_axisbelow(True)

    for axis in axes_flat[len(metrics) :]:
        axis.remove()

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=MODEL_COLORS.get(model, "#666666"),
            markeredgecolor=MODEL_COLORS.get(model, "#666666"),
            alpha=0.55,
            markersize=7,
            label=display_model_label(model),
        )
        for model in models
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(models),
    )

    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare models on the same datasets using matched sample IDs "
            "from per_sample_metrics.csv."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--protocol", default="gt_bbox", choices=["gt_bbox", "jitter_bbox"])
    parser.add_argument("--model-a", default="medsam")
    parser.add_argument("--model-b", default="samus")
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model list for multi-model comparison, "
            "for example 'medsam,samus,ultrasam'. One or more models are "
            "supported. If omitted, --model-a and --model-b are used for the "
            "original pairwise comparison."
        ),
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "Comma-separated metrics to compare/plot, or 'all'. "
            f"Available: {', '.join(DEFAULT_METRICS)}."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Figure output path. Defaults to analysis/figures/...",
    )
    parser.add_argument(
        "--plot-title",
        default="",
        help="Optional figure title. By default no title is drawn for paper-style output.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Print the table without generating a figure.",
    )
    args = parser.parse_args()
    metrics = parse_metrics_arg(args.metrics)

    if args.models is None:
        models = (
            normalize_model_name(args.model_a),
            normalize_model_name(args.model_b),
        )
        rows = compare_models(
            results_dir=args.results_dir,
            protocol=args.protocol,
            model_a=models[0],
            model_b=models[1],
            metrics=metrics,
        )
        columns = column_names(models[0], models[1], metrics)
    else:
        models = parse_models_arg(args.models)
        rows = compare_model_set(
            results_dir=args.results_dir,
            protocol=args.protocol,
            models=models,
            metrics=metrics,
        )
        columns = model_set_column_names(models, metrics)

    if not rows:
        raise SystemExit("No datasets with matched sample IDs found.")

    print_markdown(rows, columns)
    if args.output_csv is not None:
        write_csv(rows, columns, args.output_csv)
        print(f"\nSaved CSV to: {args.output_csv}")
    if not args.no_plot:
        if args.models is None:
            plot_path = args.plot_path or default_plot_path(
                args.protocol,
                models[0],
                models[1],
            )
        else:
            plot_path = args.plot_path or default_model_set_plot_path(args.protocol, models)
        plot_multi_model_rows(
            rows=rows,
            plot_path=plot_path,
            models=models,
            metrics=metrics,
            title=args.plot_title,
        )
        print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
