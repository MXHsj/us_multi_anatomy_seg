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
    "medicalsam3": "#8172B3",
    "medsam3": "#884cb0",
}
MODEL_LABELS = {
    "medsam": "MedSAM",
    "samus": "SAMUS",
    "ultrasam": "UltraSAM",
    "medicalsam3": "Medical SAM3",
    "medsam3": "MedSAM3",
}
# Fallback colors for the second series when both series share the same model
# (e.g. comparing Medical SAM3 label vs object), so the two bars stay distinct.
SECOND_SERIES_COLOR = "#DD8452"
PROTOCOL_LABELS = {
    "gt_bbox": "GT box",
    "jitter_bbox": "jitter box",
    "text": "text",
    "label": "label",
    "object": "object",
    "textbbox": "text+box",
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
        protocol = "text"
        dataset = "_".join(parts[3:])
    else:
        return None

    if not dataset:
        return None

    return model, protocol, dataset


def parse_prompt_family_dir(name: str) -> tuple[str, str] | None:
    """Decode a nested prompt-family dir name into (model, protocol).

    Medical SAM3's text-prompted runs are grouped one level deeper than the flat
    box convention: results/<model>_<style>_prompt/<dataset>/ (e.g.
    medicalsam3_label_prompt -> ('medicalsam3', 'label'), medicalsam3_object_prompt
    -> ('medicalsam3', 'object')). Returns None for names not ending in '_prompt'.
    """
    parts = name.split("_")
    if len(parts) >= 3 and parts[-1] == "prompt":
        model = "_".join(parts[:-2])
        protocol = parts[-2]
        if model and protocol:
            return model, protocol
    return None


def iter_run_dirs(results_dir: Path):
    """Yield (model, protocol, dataset, run_dir) for every discoverable run.

    Supports both layouts:
      * flat box/text convention: results/<model>_<protocol>_<dataset>/
      * nested prompt families:   results/<model>_<style>_prompt/<dataset>/
    """
    if not results_dir.is_dir():
        return
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        family = parse_prompt_family_dir(child.name)
        if family is not None:
            model, protocol = family
            for dataset_dir in sorted(child.iterdir()):
                if dataset_dir.is_dir():
                    yield model, protocol, dataset_dir.name, dataset_dir
            continue
        parsed = parse_result_dir_name(child.name)
        if parsed is not None:
            model, protocol, dataset = parsed
            yield model, protocol, dataset, child


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


def discover_results(results_dir: Path) -> dict[tuple[str, str, str], Path]:
    """Map (model, protocol, dataset) -> per_sample_metrics.csv for every run."""
    discovered: dict[tuple[str, str, str], Path] = {}
    for model, protocol, dataset, run_dir in iter_run_dirs(results_dir):
        metrics_path = run_dir / "per_sample_metrics.csv"
        if metrics_path.exists():
            discovered[(model, protocol, dataset)] = metrics_path
    return discovered


def compare_models(
    results_dir: Path,
    key_a: tuple[str, str],
    key_b: tuple[str, str],
    tag_a: str,
    tag_b: str,
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Compare two series, each identified by a (model, protocol) key.

    tag_a/tag_b are the column/legend identifiers; they differ from the model name
    when both series share a model (e.g. Medical SAM3 label vs object).
    """
    discovered = discover_results(results_dir)
    model_a, protocol_a = key_a
    model_b, protocol_b = key_b
    datasets = sorted(
        dataset
        for (result_model, result_protocol, dataset) in discovered
        if (result_model, result_protocol) == key_a
        and (model_b, protocol_b, dataset) in discovered
    )

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        metrics_a = read_metrics(discovered[(model_a, protocol_a, dataset)], metrics)
        metrics_b = read_metrics(discovered[(model_b, protocol_b, dataset)], metrics)
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
                tag_a: values_a,
                tag_b: values_b,
            }

            mean_a, std_a = mean_std(values_a)
            mean_b, std_b = mean_std(values_b)
            delta_mean, delta_std = mean_std(deltas)

            row[f"{tag_a}_{metric}_mean"] = mean_a
            row[f"{tag_a}_{metric}_std"] = std_a
            row[f"{tag_b}_{metric}_mean"] = mean_b
            row[f"{tag_b}_{metric}_std"] = std_b
            row[f"{tag_b}_minus_{tag_a}_{metric}_mean"] = delta_mean
            row[f"{tag_b}_minus_{tag_a}_{metric}_std"] = delta_std

        row["_values_by_metric"] = values_by_metric
        rows.append(row)

    return rows


def format_float(value: float) -> str:
    return f"{value:.4f}"


def column_names(tag_a: str, tag_b: str, metrics: tuple[str, ...]) -> list[str]:
    columns = ["dataset"]
    for metric in metrics:
        columns.extend(
            [
                f"{tag_a}_{metric}_mean",
                f"{tag_a}_{metric}_std",
                f"{tag_b}_{metric}_mean",
                f"{tag_b}_{metric}_std",
                f"{tag_b}_minus_{tag_a}_{metric}_mean",
                f"{tag_b}_minus_{tag_a}_{metric}_std",
            ]
        )
    return columns


def print_markdown(rows: list[dict[str, Any]], columns: list[str]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [
            str(row[column]) if column == "dataset" else format_float(row[column])
            for column in columns
        ]
        print("| " + " | ".join(values) + " |")


def write_csv(rows: list[dict[str, Any]], columns: list[str], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def default_plot_path(tag_a: str, tag_b: str) -> Path:
    return (
        Path("analysis")
        / "figures"
        / f"compare_models_same_dataset_{tag_a}_vs_{tag_b}.png"
    )


def display_dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.upper())


def display_series_label(model: str, protocol: str, show_protocol: bool) -> str:
    base = MODEL_LABELS.get(model, model)
    if show_protocol:
        return f"{base} ({PROTOCOL_LABELS.get(protocol, protocol)})"
    return base


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
    series: list[dict[str, str]],
    metrics: tuple[str, ...],
    title: str = "",
) -> None:
    # series: two dicts, each {"tag": ..., "color": ..., "label": ...}.
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
    bar_width = 0.32
    offsets = {
        series[0]["tag"]: -bar_width / 2,
        series[1]["tag"]: bar_width / 2,
    }
    rng = np.random.default_rng(20240515)

    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig_width = max(6.8, len(rows) * 0.48, ncols * 3.2)
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
        for spec in series:
            tag = spec["tag"]
            color = spec["color"]
            means = [row[f"{tag}_{metric}_mean"] for row in rows]
            stds = [row[f"{tag}_{metric}_std"] for row in rows]
            model_positions = x_positions + offsets[tag]

            axis.bar(
                model_positions,
                means,
                width=bar_width,
                color=color,
                alpha=0.32,
                edgecolor=color,
                linewidth=1.1,
                zorder=2,
                label=spec["label"],
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
                values = row["_values_by_metric"][metric][tag]
                if not values:
                    continue
                jitter = rng.uniform(-bar_width * 0.27, bar_width * 0.27, size=len(values))
                axis.scatter(
                    np.full(len(values), model_positions[idx]) + jitter,
                    values,
                    s=6,
                    color=color,
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

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=spec["color"],
            markeredgecolor=spec["color"],
            alpha=0.55,
            markersize=7,
            label=spec["label"],
        )
        for spec in series
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04 if not title else 1.10),
        ncol=2,
    )

    if title:
        fig.suptitle(title, y=1.02, fontsize=10)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two models on the same datasets using matched sample IDs "
            "from per_sample_metrics.csv."
        )
    )
    _PROTOCOLS = ["gt_bbox", "jitter_bbox", "text", "label", "object"]
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--protocol",
        default="gt_bbox",
        choices=_PROTOCOLS,
        help="Protocol used for both series unless overridden by --protocol-a/-b.",
    )
    parser.add_argument(
        "--protocol-a",
        default=None,
        choices=_PROTOCOLS,
        help="Protocol for series A (defaults to --protocol).",
    )
    parser.add_argument(
        "--protocol-b",
        default=None,
        choices=_PROTOCOLS,
        help=(
            "Protocol for series B (defaults to --protocol). Set this with "
            "--model-a/--model-b equal to compare prompt styles of one model, "
            "e.g. --model-a medicalsam3 --model-b medicalsam3 "
            "--protocol-a label --protocol-b object."
        ),
    )
    parser.add_argument("--model-a", default="medsam")
    parser.add_argument("--model-b", default="samus")
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

    protocol_a = args.protocol_a or args.protocol
    protocol_b = args.protocol_b or args.protocol
    key_a = (args.model_a, protocol_a)
    key_b = (args.model_b, protocol_b)

    # Disambiguate series identifiers/labels when the two share a model (e.g. one
    # model's label vs object prompt) or differ only by protocol; otherwise keep
    # plain model names so existing box-vs-box output is unchanged.
    show_protocol = (args.model_a == args.model_b) or (protocol_a != protocol_b)
    tag_a = f"{args.model_a}_{protocol_a}" if args.model_a == args.model_b else args.model_a
    tag_b = f"{args.model_b}_{protocol_b}" if args.model_a == args.model_b else args.model_b

    rows = compare_models(
        results_dir=args.results_dir,
        key_a=key_a,
        key_b=key_b,
        tag_a=tag_a,
        tag_b=tag_b,
        metrics=metrics,
    )

    if not rows:
        raise SystemExit("No datasets with matched sample IDs found.")

    columns = column_names(tag_a, tag_b, metrics)
    print_markdown(rows, columns)
    if args.output_csv is not None:
        write_csv(rows, columns, args.output_csv)
        print(f"\nSaved CSV to: {args.output_csv}")
    if not args.no_plot:
        color_a = MODEL_COLORS.get(args.model_a, "#4C72B0")
        color_b = MODEL_COLORS.get(args.model_b, "#55A868")
        if args.model_a == args.model_b:
            color_b = SECOND_SERIES_COLOR
        series = [
            {
                "tag": tag_a,
                "color": color_a,
                "label": display_series_label(args.model_a, protocol_a, show_protocol),
            },
            {
                "tag": tag_b,
                "color": color_b,
                "label": display_series_label(args.model_b, protocol_b, show_protocol),
            },
        ]
        plot_path = args.plot_path or default_plot_path(tag_a, tag_b)
        plot_rows(
            rows=rows,
            plot_path=plot_path,
            series=series,
            metrics=metrics,
            title=args.plot_title,
        )
        print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
