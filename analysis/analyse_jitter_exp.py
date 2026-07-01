from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.compare_models_same_dataset import (
    display_dataset_label,
    metric_label,
    order_datasets,
    parse_metrics_arg,
    read_metrics,
)


DEFAULT_DATASETS = (
    "aulid",
    "blusg",
    "busbra",
    "busi",
    "camus",
    "oku",
    "mmotu",
    "gist514",
    "roblus",
    "tnsc2020",
    "ultrabones100k",
    "umud",
    "uns",
)

BOUNDED_METRICS = {"dice", "iou", "precision", "recall", "specificity", "balanced_accuracy"}


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_scales(value: str) -> list[tuple[str, float]]:
    scales: list[tuple[str, float]] = []
    for item in split_csv(value):
        try:
            scale = float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid scale value: {item}") from exc
        if scale <= 0:
            raise argparse.ArgumentTypeError("Scale values must be > 0.")
        scales.append((item, scale))
    if not scales:
        raise argparse.ArgumentTypeError("At least one scale is required.")
    return scales


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def result_metrics_path(results_dir: Path, dataset: str, scale_text: str | None) -> Path:
    if scale_text is None:
        return results_dir / f"ultrasam_gt_bbox_{dataset}" / "per_sample_metrics.csv"
    return results_dir / f"ultrasam_scale_{scale_text}_bbox_{dataset}" / "per_sample_metrics.csv"


def collect_dataset_rows(
    *,
    results_dir: Path,
    dataset: str,
    scales: list[tuple[str, float]],
    metrics: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    baseline_path = result_metrics_path(results_dir, dataset, scale_text=None)
    if not baseline_path.exists():
        return [], [f"missing baseline: {baseline_path}"]

    baseline = read_metrics(baseline_path, metrics)
    if not baseline:
        return [], [f"empty baseline metrics: {baseline_path}"]

    scale_metrics: dict[float, dict[str, dict[str, float]]] = {}
    for scale_text, scale_value in scales:
        metrics_path = result_metrics_path(results_dir, dataset, scale_text=scale_text)
        if not metrics_path.exists():
            warnings.append(f"missing scale {scale_text}: {metrics_path}")
            continue
        values = read_metrics(metrics_path, metrics)
        if not values:
            warnings.append(f"empty scale {scale_text}: {metrics_path}")
            continue
        scale_metrics[scale_value] = values

    if not scale_metrics:
        return [], warnings

    matched_ids = set(baseline)
    for values in scale_metrics.values():
        matched_ids &= set(values)
    matched_ids = set(sorted(matched_ids))
    if not matched_ids:
        warnings.append(f"no shared sample_id values for {dataset}")
        return [], warnings

    rows: list[dict[str, Any]] = []

    baseline_row: dict[str, Any] = {
        "dataset": dataset,
        "scale": 1.0,
        "n": len(matched_ids),
    }
    for metric in metrics:
        raw_values = [baseline[sample_id][metric] for sample_id in matched_ids]
        raw_mean, raw_std = mean_std(raw_values)
        baseline_row[f"{metric}_mean"] = raw_mean
        baseline_row[f"{metric}_std"] = raw_std
        baseline_row[f"{metric}_delta_mean"] = 0.0
        baseline_row[f"{metric}_delta_std"] = 0.0
        baseline_row.setdefault("_values_by_metric", {})[metric] = {
            "raw": raw_values,
            "delta": [0.0 for _ in raw_values],
        }
    rows.append(baseline_row)

    for scale_value in sorted(scale_metrics):
        row: dict[str, Any] = {
            "dataset": dataset,
            "scale": scale_value,
            "n": len(matched_ids),
        }
        for metric in metrics:
            raw_values = [scale_metrics[scale_value][sample_id][metric] for sample_id in matched_ids]
            deltas = [
                scale_metrics[scale_value][sample_id][metric] - baseline[sample_id][metric]
                for sample_id in matched_ids
            ]
            raw_mean, raw_std = mean_std(raw_values)
            delta_mean, delta_std = mean_std(deltas)
            row[f"{metric}_mean"] = raw_mean
            row[f"{metric}_std"] = raw_std
            row[f"{metric}_delta_mean"] = delta_mean
            row[f"{metric}_delta_std"] = delta_std
            row.setdefault("_values_by_metric", {})[metric] = {
                "raw": raw_values,
                "delta": deltas,
            }
        rows.append(row)

    return rows, warnings


def write_summary_csv(path: Path, rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> None:
    fieldnames = ["dataset", "scale", "n"]
    for metric in metrics:
        fieldnames.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_delta_mean",
                f"{metric}_delta_std",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def configure_matplotlib() -> None:
    mpl_config_dir = ROOT_DIR / "analysis" / ".mplconfig"
    xdg_cache_dir = ROOT_DIR / "analysis" / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir.resolve()))

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 500,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
        }
    )


def plot_metric_lines(
    *,
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    column_kind: str,
    ylabel_suffix: str,
    output_path: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    datasets = order_datasets({row["dataset"] for row in rows})
    ncols = len(metrics)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(3, 3 * ncols), 2.3),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, metric in zip(axes_flat, metrics):
        for dataset in datasets:
            dataset_rows = sorted(
                [row for row in rows if row["dataset"] == dataset],
                key=lambda row: float(row["scale"]),
            )
            if not dataset_rows:
                continue
            x_values = [float(row["scale"]) for row in dataset_rows]
            mean_column = f"{metric}_mean" if column_kind == "raw" else f"{metric}_delta_mean"
            y_values = [float(row[mean_column]) * 100.0 for row in dataset_rows]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                label=display_dataset_label(dataset),
            )

        axis.axvline(1.0, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
        if column_kind == "delta":
            axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
        axis.set_xlabel("BBox scale factor")
        ylabel = (
            f"{metric_label(metric)} (%)"
            if column_kind == "raw"
            else rf"$\Delta$ {metric_label(metric)} (%)"
        )
        axis.set_ylabel(ylabel)
        if column_kind == "raw" and metric in BOUNDED_METRICS:
            axis.set_ylim(0.0, 100.0)
        axis.grid(alpha=0.35)

    for axis in axes_flat[len(metrics) :]:
        axis.remove()

    baseline_label = "Baseline"
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_items = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label != baseline_label
    ]
    legend_items.append(
        (
            Line2D([0], [0], color="#777777", linestyle="--", linewidth=0.8),
            baseline_label,
        )
    )
    handles = [handle for handle, _label in legend_items]
    labels = [label for _handle, label in legend_items]
    if handles:
        fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.55, -0.15), ncol=min(4, len(labels)))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze UltraSAM bbox scaling jitter experiments.")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "analysis" / "figures" / "prompt_robustness",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--scales", type=parse_scales, default=parse_scales("0.75,1.25,1.5,2.0"))
    parser.add_argument("--metrics", type=parse_metrics_arg, default=parse_metrics_arg("dice,iou"))
    parser.add_argument("--plot-format", default="png", choices=("png", "svg", "pdf"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = order_datasets(split_csv(args.datasets))
    all_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for dataset in datasets:
        rows, dataset_warnings = collect_dataset_rows(
            results_dir=args.results_dir,
            dataset=dataset,
            scales=args.scales,
            metrics=args.metrics,
        )
        all_rows.extend(rows)
        warnings.extend(f"{dataset}: {warning}" for warning in dataset_warnings)

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if not all_rows:
        raise SystemExit("No matched UltraSAM scaling jitter results found.")

    summary_path = args.output_dir / "ultrasam_scale_robustness_summary.csv"
    raw_plot_path = args.output_dir / f"ultrasam_scale_robustness_raw.{args.plot_format}"
    delta_plot_path = args.output_dir / f"ultrasam_scale_robustness_delta.{args.plot_format}"

    write_summary_csv(summary_path, all_rows, args.metrics)
    plot_metric_lines(
        rows=all_rows,
        metrics=args.metrics,
        column_kind="raw",
        ylabel_suffix="",
        output_path=raw_plot_path,
    )
    plot_metric_lines(
        rows=all_rows,
        metrics=args.metrics,
        column_kind="delta",
        ylabel_suffix="delta vs baseline",
        output_path=delta_plot_path,
    )

    print(f"Wrote summary to {summary_path}")
    print(f"Wrote raw plot to {raw_plot_path}")
    print(f"Wrote delta plot to {delta_plot_path}")


if __name__ == "__main__":
    main()
