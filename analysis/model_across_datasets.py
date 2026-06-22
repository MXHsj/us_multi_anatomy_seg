from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    "medicalsam3": "#8172B3",
    "medsam3": "#884cb0",
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
    elif parts[1:3] == ["text", "prompt"]:
        # Text-prompted protocol (e.g. medicalsam3_text_prompt_tnsc2020). Reported in
        # its own table -- never merged with the box (gt_bbox/jitter_bbox) results.
        protocol = "text"
        dataset = "_".join(parts[3:])
    else:
        return None

    if not dataset:
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


def metric_summary_columns(metrics: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for metric in metrics:
        columns.extend([f"{metric}_mean", f"{metric}_std"])
    return columns


def load_rows(
    results_dir: Path,
    protocol: str,
    model: str | None,
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_columns = metric_summary_columns(metrics)
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        parsed = parse_result_dir_name(summary_path.parent.name)
        if parsed is None:
            continue

        result_model, result_protocol, dataset = parsed
        if result_protocol != protocol:
            continue
        if model is not None and result_model != model:
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(summary.get(column) is None for column in metric_columns):
            continue

        rows.append(
            {
                "model": result_model,
                "dataset": dataset,
                "result_dir": summary_path.parent,
                **{column: float(summary[column]) for column in metric_columns},
            }
        )

    return sorted(rows, key=lambda row: (row["model"], row["dataset"]))


def format_float(value: float) -> str:
    return f"{value:.4f}"


def print_markdown(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> None:
    headers = ["model", "dataset", *metric_summary_columns(metrics)]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [
            row["model"],
            row["dataset"],
            *(format_float(row[column]) for column in metric_summary_columns(metrics)),
        ]
        print("| " + " | ".join(values) + " |")


def write_csv(rows: list[dict[str, Any]], output_csv: Path, metrics: tuple[str, ...]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["model", "dataset", *metric_summary_columns(metrics)]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows({column: row[column] for column in headers} for row in rows)


def read_per_sample_points(result_dir: Path, metrics: tuple[str, ...]) -> dict[str, list[float]]:
    points = {metric: [] for metric in metrics}
    metrics_path = result_dir / "per_sample_metrics.csv"
    if not metrics_path.exists():
        return points

    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for metric in metrics:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    points[metric].append(value)
    return points


def default_plot_path(protocol: str, model: str | None) -> Path:
    if model is None:
        raise ValueError("A model filter is required for the default plot path.")
    return Path("analysis") / "figures" / f"model_across_datasets_{protocol}_{model}.png"


def _display_label(row: dict[str, Any], include_model: bool) -> str:
    dataset = DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"]).upper())
    if include_model:
        return f"{row['model']}\n{dataset}"
    return dataset


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

    include_model = len({row["model"] for row in rows}) > 1
    labels = [_display_label(row, include_model=include_model) for row in rows]
    x_positions = np.arange(len(rows))
    width = 0.62
    rng = np.random.default_rng(20240515)

    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig_width = max(6.8, len(rows) * 0.42, ncols * 3.2)
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
        means = [row[f"{metric}_mean"] for row in rows]
        stds = [row[f"{metric}_std"] for row in rows]
        colors = [MODEL_COLORS.get(row["model"], "#666666") for row in rows]

        axis.bar(
            x_positions,
            means,
            width=width,
            color=colors,
            alpha=0.32,
            edgecolor=colors,
            linewidth=1.1,
            zorder=2,
        )
        axis.errorbar(
            x_positions,
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
            values = read_per_sample_points(row["result_dir"], metrics)[metric]
            if not values:
                continue
            jitter = rng.uniform(-width * 0.30, width * 0.30, size=len(values))
            axis.scatter(
                np.full(len(values), x_positions[idx]) + jitter,
                values,
                s=6,
                color=MODEL_COLORS.get(row["model"], "#666666"),
                alpha=0.18,
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

    handles = []
    for model in sorted({row["model"] for row in rows}):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=MODEL_COLORS.get(model, "#666666"),
                markeredgecolor=MODEL_COLORS.get(model, "#666666"),
                alpha=0.55,
                markersize=7,
                label=model,
            )
        )
    if len(handles) > 1:
        axes_flat[min(len(metrics), len(axes_flat)) - 1].legend(
            handles=handles,
            frameon=False,
            loc="best",
        )

    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize same-model performance across datasets."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--protocol", default="gt_bbox", choices=["gt_bbox", "jitter_bbox", "text"]
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional model filter, for example 'medsam', 'samus', or 'ultrasam'.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "Comma-separated metrics to summarize/plot, or 'all'. "
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
        help=(
            "Figure output path. Defaults to analysis/figures/... for --model; "
            "when --model is omitted, one default figure is saved per model."
        ),
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

    rows = load_rows(
        results_dir=args.results_dir,
        protocol=args.protocol,
        model=args.model or None,
        metrics=metrics,
    )

    if not rows:
        raise SystemExit("No matching result summaries found.")

    print_markdown(rows, metrics)
    if args.output_csv is not None:
        write_csv(rows, args.output_csv, metrics)
        print(f"\nSaved CSV to: {args.output_csv}")
    if args.no_plot:
        return

    if args.model or args.plot_path is not None:
        plot_path = args.plot_path or default_plot_path(args.protocol, args.model)
        plot_rows(rows, plot_path, metrics=metrics, title=args.plot_title)
        print(f"Saved plot to: {plot_path}")
        return

    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        plot_path = default_plot_path(args.protocol, model)
        plot_rows(model_rows, plot_path, metrics=metrics, title=args.plot_title)
        print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
