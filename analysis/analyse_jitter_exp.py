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
DATASET_COLOR_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
)
EXPERIMENT_CONFIG = {
    "scale": {
        "baseline_value": 1.0,
        "x_label": "BBox scale factor",
        "title": "Scale",
        "output_prefix": "ultrasam_scale_robustness",
    },
    "translation": {
        "baseline_value": 0.0,
        "x_label": "BBox translation fraction",
        "title": "Translation",
        "output_prefix": "ultrasam_translation_robustness",
    },
    "point": {
        "baseline_value": 0.0,
        "x_label": "Point prompt jitter fraction",
        "title": "Point prompt jitter",
        "output_prefix": "ultrasam_point_prompt_robustness",
    },
}
COMBINED_EXPERIMENTS = ("scale", "translation")


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_values(value: str) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for item in split_csv(value):
        try:
            numeric_value = float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid jitter value: {item}") from exc
        if numeric_value < 0:
            raise argparse.ArgumentTypeError("Jitter values must be >= 0.")
        values.append((item, numeric_value))
    if not values:
        raise argparse.ArgumentTypeError("At least one jitter value is required.")
    return values


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def result_metrics_path(
    *,
    jitter_results_dir: Path,
    dataset: str,
    experiment: str,
    value_text: str,
) -> Path:
    if experiment == "scale":
        return jitter_results_dir / "scale" / f"ultrasam_scale_{value_text}_bbox_{dataset}" / "per_sample_metrics.csv"
    if experiment == "translation":
        return jitter_results_dir / "trans" / f"ultrasam_trans_{value_text}_bbox_{dataset}" / "per_sample_metrics.csv"
    return jitter_results_dir / "point" / f"ultrasam_point_{value_text}_point_{dataset}" / "per_sample_metrics.csv"


def collect_dataset_rows(
    *,
    jitter_results_dir: Path,
    dataset: str,
    experiment: str,
    values: list[tuple[str, float]],
    baseline_value: float,
    metrics: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []

    experiment_metrics: dict[float, dict[str, dict[str, float]]] = {}
    baseline: dict[str, dict[str, float]] | None = None
    for value_text, numeric_value in values:
        metrics_path = result_metrics_path(
            jitter_results_dir=jitter_results_dir,
            dataset=dataset,
            experiment=experiment,
            value_text=value_text,
        )
        if not metrics_path.exists():
            warnings.append(f"missing {experiment} {value_text}: {metrics_path}")
            continue
        metric_values = read_metrics(metrics_path, metrics)
        if not metric_values:
            warnings.append(f"empty {experiment} {value_text}: {metrics_path}")
            continue
        experiment_metrics[numeric_value] = metric_values
        if math.isclose(numeric_value, baseline_value):
            baseline = metric_values

    if not experiment_metrics:
        return [], warnings
    if baseline is None:
        baseline_text = f"{baseline_value:g}"
        baseline_path = result_metrics_path(
            jitter_results_dir=jitter_results_dir,
            dataset=dataset,
            experiment=experiment,
            value_text=baseline_text,
        )
        warnings.append(f"missing baseline {experiment} {baseline_text}: {baseline_path}")
        return [], warnings

    matched_ids = set(baseline)
    for values in experiment_metrics.values():
        matched_ids &= set(values)
    matched_ids = set(sorted(matched_ids))
    if not matched_ids:
        warnings.append(f"no shared sample_id values for {dataset}")
        return [], warnings

    rows: list[dict[str, Any]] = []

    baseline_row: dict[str, Any] = {
        "experiment": experiment,
        "dataset": dataset,
        "jitter_value": baseline_value,
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

    for numeric_value in sorted(experiment_metrics):
        if math.isclose(numeric_value, baseline_value):
            continue
        row: dict[str, Any] = {
            "experiment": experiment,
            "dataset": dataset,
            "jitter_value": numeric_value,
            "n": len(matched_ids),
        }
        for metric in metrics:
            raw_values = [experiment_metrics[numeric_value][sample_id][metric] for sample_id in matched_ids]
            deltas = [
                experiment_metrics[numeric_value][sample_id][metric] - baseline[sample_id][metric]
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
    fieldnames = ["experiment", "dataset", "jitter_value", "n"]
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
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
        }
    )


def dataset_color_map(datasets: list[str]) -> dict[str, str]:
    ordered_known_datasets = order_datasets(DEFAULT_DATASETS)
    color_by_dataset = {
        dataset: DATASET_COLOR_PALETTE[index % len(DATASET_COLOR_PALETTE)]
        for index, dataset in enumerate(ordered_known_datasets)
    }
    next_color_index = len(color_by_dataset)
    for dataset in datasets:
        if dataset not in color_by_dataset:
            color_by_dataset[dataset] = DATASET_COLOR_PALETTE[next_color_index % len(DATASET_COLOR_PALETTE)]
            next_color_index += 1
    return color_by_dataset


def jitter_dataset_label(dataset: str) -> str:
    if dataset == "tnsc2020":
        return "TNSC"
    if dataset == "ultrabones100k":
        return "UltraBones100k"
    return display_dataset_label(dataset)


def plot_metric_lines(
    *,
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    column_kind: str,
    x_label: str,
    baseline_value: float,
    output_path: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    datasets = order_datasets({row["dataset"] for row in rows})
    colors = dataset_color_map(datasets)
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
                key=lambda row: float(row["jitter_value"]),
            )
            if not dataset_rows:
                continue
            x_values = [float(row["jitter_value"]) for row in dataset_rows]
            mean_column = f"{metric}_mean" if column_kind == "raw" else f"{metric}_delta_mean"
            y_values = [float(row[mean_column]) * 100.0 for row in dataset_rows]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.2,
                markersize=3.5,
                color=colors[dataset],
                label=jitter_dataset_label(dataset),
            )

        axis.axvline(baseline_value, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
        if column_kind == "delta":
            axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
        axis.set_xlabel(x_label)
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


def add_shared_legend(fig: Any, axes_flat: Any) -> None:
    from matplotlib.lines import Line2D

    seen: set[str] = set()
    legend_items = []
    baseline_label = "Baseline"
    for axis in axes_flat:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label == baseline_label or label in seen:
                continue
            seen.add(label)
            legend_items.append((handle, label))

    legend_items.append(
        (
            Line2D([0], [0], color="#777777", linestyle="--", linewidth=0.8),
            baseline_label,
        )
    )
    if not legend_items:
        return

    ncols = min(8, len(legend_items))
    handles = [handle for handle, _label in legend_items]
    labels = [label for _handle, label in legend_items]
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=ncols,
    )


def add_line_plots_legend(fig: Any, axes_flat: Any, *, y_anchor: float = -0.05) -> None:
    from matplotlib.lines import Line2D

    seen: set[str] = set()
    legend_items = []
    baseline_label = "Baseline"
    for axis in axes_flat:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label == baseline_label or label in seen:
                continue
            seen.add(label)
            legend_items.append((handle, label))

    legend_items.append(
        (
            Line2D([0], [0], color="#777777", linestyle="--", linewidth=0.8),
            baseline_label,
        )
    )
    if not legend_items:
        return

    line_left = min(axis.get_position().x0 for axis in axes_flat)
    line_right = max(axis.get_position().x1 for axis in axes_flat)
    handles = [handle for handle, _label in legend_items]
    labels = [label for _handle, label in legend_items]
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=((line_left + line_right) / 2.0, y_anchor),
        ncol=min(5, len(legend_items)),
    )


def str2bool(value: str) -> bool:
    return value.strip().lower() == "true"


def optional_float(value: str) -> float | None:
    if not value.strip():
        return None
    return float(value)


def prepare_eda_heatmap_data(
    *,
    results_dir: Path,
    eda_dir: Path,
    model: str,
    protocol: str,
    datasets: tuple[str, ...],
    result_metric: str,
    eda_metrics_arg: str,
    eda_labels_arg: str,
    result_label: str,
    correlation: str,
    per_image: bool,
    dice_threshold: float | None,
    ultrabones_min_solidity: float | None,
    filter_tnsc2020_target_bbox_one: bool,
    tnsc2020_min_aspect_ratio: float | None,
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, str], str]:
    from analysis import quantitative_results_analysis as qra

    available_eda_metrics = qra.discover_eda_metrics(eda_dir, list(datasets))
    if not available_eda_metrics:
        raise SystemExit(f"No EDA sample_stats.json box metrics found under {eda_dir}.")
    eda_metrics = qra.parse_csv_arg(eda_metrics_arg, available_eda_metrics, "EDA metric")
    plot_labels = {
        result_metric: result_label,
        **qra.labels_for(eda_metrics, eda_labels_arg),
    }

    data_by_dataset: dict[str, Any] = {}
    for dataset in datasets:
        try:
            data_by_dataset[dataset] = qra.load_dataset_points(
                results_dir,
                eda_dir,
                model,
                dataset,
                protocol,
                eda_metrics,
                tuple(
                    metric
                    for condition, metric in (
                        (dataset == "ultrabones100k" and ultrabones_min_solidity is not None, "solidity"),
                        (dataset == "tnsc2020" and filter_tnsc2020_target_bbox_one, "target_bbox_area_ratio"),
                        (dataset == "tnsc2020" and tnsc2020_min_aspect_ratio is not None, "aspect_ratio_feret"),
                    )
                    if condition
                ),
                (),
            )
        except FileNotFoundError as error:
            print(f"Skipping EDA heatmap dataset {dataset}: {error}", file=sys.stderr)

    if not data_by_dataset:
        raise SystemExit("No heatmap datasets had both per_box_metrics.json and EDA sample_stats.json files.")

    data_by_dataset = qra.filter_ultrabones_solidity(data_by_dataset, ultrabones_min_solidity)
    data_by_dataset = qra.filter_tnsc2020_full_target_bbox(data_by_dataset, filter_tnsc2020_target_bbox_one)
    data_by_dataset = qra.filter_tnsc2020_aspect_ratio(data_by_dataset, tnsc2020_min_aspect_ratio)
    if per_image:
        data_by_dataset = {
            dataset: qra.aggregate_per_image(df, (*eda_metrics, result_metric))
            for dataset, df in data_by_dataset.items()
        }
    data_by_dataset = qra.filter_by_dice_threshold(
        data_by_dataset,
        dice_threshold,
        enabled=result_metric == "dice",
    )
    if not data_by_dataset:
        raise SystemExit("No heatmap datasets had rows remaining after filtering.")

    return data_by_dataset, eda_metrics, plot_labels, qra.effective_correlation(correlation, False)


def draw_dataset_heatmap_axis(
    *,
    axis: Any,
    plt: Any,
    data_by_dataset: dict[str, Any],
    y_metric: str,
    eda_metrics: tuple[str, ...],
    method: str,
    plot_labels: dict[str, str],
) -> Any:
    from analysis import quantitative_results_analysis as qra

    dataset_labels = [jitter_dataset_label(dataset) for dataset in data_by_dataset]
    matrix = [
        [
            qra.corr_value(qra.finite_points(df, x_metric, y_metric, False), x_metric, y_metric, method)
            for df in data_by_dataset.values()
        ]
        for x_metric in eda_metrics
    ]
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_title("Target shape characteristics")
    axis.set_yticks(range(len(eda_metrics)), [qra.plot_label(metric, plot_labels) for metric in eda_metrics])
    axis.set_xticks(
        range(len(dataset_labels)),
        dataset_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axis.tick_params(axis="x", bottom=False)
    axis.grid(False)
    for row in range(len(eda_metrics)):
        for col in range(len(dataset_labels)):
            value = matrix[row][col]
            if not math.isnan(value):
                axis.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=qra.HEATMAP_CELL_FONTSIZE,
                    color=qra.heat_text_color(plt, value),
                )
    return image


def add_bottom_panel_labels(
    fig: Any,
    axes: tuple[Any, ...],
    labels: tuple[str, ...],
    *,
    y_anchor: float,
    fontsize: float = 14,
) -> None:
    for axis, label in zip(axes, labels):
        position = axis.get_position()
        fig.text(
            (position.x0 + position.x1) / 2.0,
            y_anchor,
            label,
            ha="center",
            va="top",
            fontsize=fontsize,
        )


def correlation_symbol_label(method: str) -> str:
    return "ρ" if method == "spearman" else "r"


def plot_combined_metric_pair(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    column_kind: str,
    experiments: tuple[str, ...],
    output_path: Path,
    heatmap_data: dict[str, Any],
    heatmap_eda_metrics: tuple[str, ...],
    heatmap_plot_labels: dict[str, str],
    heatmap_correlation: str,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from analysis import quantitative_results_analysis as qra

    datasets = order_datasets({row["dataset"] for row in rows})
    colors = dataset_color_map(datasets)
    fig, axes = plt.subplots(
        1,
        len(experiments) + 2,
        figsize=(8.5, 2.3),
        width_ratios=[1.0, 1.0, 0.1, 1],
        gridspec_kw={"wspace": 0.25},
        squeeze=False,
    )
    axes_flat = axes.ravel()
    line_axes = axes_flat[: len(experiments)]
    spacer_axis = axes_flat[len(experiments)]
    heatmap_axis = axes_flat[len(experiments) + 1]
    spacer_axis.remove()
    line_y_values: list[float] = []

    for axis, experiment in zip(line_axes, experiments):
        config = EXPERIMENT_CONFIG[experiment]
        experiment_rows = [row for row in rows if row["experiment"] == experiment]

        for dataset in datasets:
            dataset_rows = sorted(
                [row for row in experiment_rows if row["dataset"] == dataset],
                key=lambda row: float(row["jitter_value"]),
            )
            if not dataset_rows:
                continue
            x_values = [float(row["jitter_value"]) for row in dataset_rows]
            mean_column = f"{metric}_mean" if column_kind == "raw" else f"{metric}_delta_mean"
            y_values = [float(row[mean_column]) * 100.0 for row in dataset_rows]
            line_y_values.extend(value for value in y_values if math.isfinite(value))
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1,
                markersize=3,
                color=colors[dataset],
                label=jitter_dataset_label(dataset),
            )

        axis.axvline(
            float(config["baseline_value"]),
            color="#777777",
            linestyle="--",
            linewidth=0.8,
            zorder=0,
        )
        if column_kind == "delta":
            axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
        axis.set_title(str(config["title"]))
        axis.set_xlabel(str(config["x_label"]))
        axis.grid(alpha=0.35)

    if column_kind == "raw" and metric in BOUNDED_METRICS:
        y_limits = (0.0, 100.0)
    elif line_y_values:
        y_min = min(line_y_values)
        y_max = max(line_y_values)
        if column_kind == "delta":
            y_min = min(y_min, 0.0)
            y_max = max(y_max, 0.0)
        padding = max((y_max - y_min) * 0.05, 0.5)
        y_limits = (y_min - padding, y_max + padding)
    else:
        y_limits = None

    if y_limits is not None:
        for axis in line_axes:
            axis.set_ylim(*y_limits)

    ylabel = (
        f"{metric_label(metric)} (%)"
        if column_kind == "raw"
        else rf"$\Delta$ {metric_label(metric)} (%)"
    )
    line_axes[0].set_ylabel(ylabel)

    image = draw_dataset_heatmap_axis(
        axis=heatmap_axis,
        plt=plt,
        data_by_dataset=heatmap_data,
        y_metric="dice",
        eda_metrics=heatmap_eda_metrics,
        method=heatmap_correlation,
        plot_labels=heatmap_plot_labels,
    )
    fig.colorbar(
        image,
        ax=heatmap_axis,
        label=correlation_symbol_label(heatmap_correlation),
        shrink=0.82,
        pad=0.02,
    )
    fig.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.35, wspace=0.16)
    add_line_plots_legend(fig, line_axes, y_anchor=-0.07)
    add_bottom_panel_labels(fig, (*line_axes, heatmap_axis), ("a)", "b)", "c)"), y_anchor=-0.1, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    correlation_methods = ("pearson", "log_pearson", "spearman")
    parser = argparse.ArgumentParser(description="Analyze UltraSAM prompt jitter experiments.")
    parser.add_argument(
        "--jitter-results-dir",
        type=Path,
        default=ROOT_DIR / "experiments" / "prompt_robustness",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "analysis" / "figures" / "prompt_robustness",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--experiment", choices=("scale", "translation", "point", "both"), default="both")
    parser.add_argument("--scales", type=parse_values, default=parse_values("0.85,0.9,0.95,1,1.05,1.10,1.15"))
    parser.add_argument("--translations", type=parse_values, default=parse_values("0,0.025,0.05,0.075,0.10,0.125,0.15"))
    parser.add_argument("--point-jitters", type=parse_values, default=parse_values("0,0.025,0.05,0.075,0.10,0.125,0.15"))
    parser.add_argument("--metrics", type=parse_metrics_arg, default=parse_metrics_arg("dice,iou"))
    parser.add_argument("--plot-format", default="png", choices=("png", "svg", "pdf"))
    parser.add_argument("--heatmap-results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--heatmap-eda-dir", type=Path, default=ROOT_DIR / "analysis" / "eda")
    parser.add_argument("--heatmap-model", default="ultrasam")
    parser.add_argument("--heatmap-protocol", default="gt_bbox", choices=("gt_bbox", "jitter_bbox"))
    parser.add_argument(
        "--heatmap-datasets",
        default="mmotu,busbra,gist514,tnsc2020,ultrabones100k,umud",
        help="Comma-separated datasets for the Dice-vs-EDA heatmap in the combined figure.",
    )
    parser.add_argument(
        "--heatmap-eda-metrics",
        default="target_area_fraction,target_bbox_area_ratio,solidity,circularity,aspect_ratio_feret",
    )
    parser.add_argument(
        "--heatmap-eda-labels",
        default="Target/image,Target/bbox,Solidity,Circularity,Aspect ratio",
    )
    parser.add_argument("--heatmap-result-label", default="Dice")
    parser.add_argument("--heatmap-correlation", default="spearman", choices=correlation_methods)
    parser.add_argument("--heatmap-per-image", type=str2bool, default=False)
    parser.add_argument("--heatmap-dice-threshold", type=optional_float, default=1.0)
    parser.add_argument("--heatmap-ultrabones-min-solidity", type=optional_float, default=0.2)
    parser.add_argument("--heatmap-filter-tnsc2020-target-bbox-one", type=str2bool, default=True)
    parser.add_argument("--heatmap-tnsc2020-min-aspect-ratio", type=optional_float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = order_datasets(split_csv(args.datasets))
    experiments = COMBINED_EXPERIMENTS if args.experiment == "both" else (args.experiment,)
    all_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for experiment in experiments:
        if experiment == "scale":
            values = args.scales
        elif experiment == "translation":
            values = args.translations
        else:
            values = args.point_jitters
        baseline_value = float(EXPERIMENT_CONFIG[experiment]["baseline_value"])
        for dataset in datasets:
            rows, dataset_warnings = collect_dataset_rows(
                jitter_results_dir=args.jitter_results_dir,
                dataset=dataset,
                experiment=experiment,
                values=values,
                baseline_value=baseline_value,
                metrics=args.metrics,
            )
            all_rows.extend(rows)
            warnings.extend(f"{dataset} {experiment}: {warning}" for warning in dataset_warnings)

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if not all_rows:
        raise SystemExit(f"No matched UltraSAM {args.experiment} jitter results found.")

    if args.experiment == "both":
        output_prefix = "ultrasam_jitter_robustness"
    else:
        output_prefix = str(EXPERIMENT_CONFIG[args.experiment]["output_prefix"])
    summary_path = args.output_dir / f"{output_prefix}_summary.csv"

    write_summary_csv(summary_path, all_rows, args.metrics)
    written_plots: list[Path] = []
    if args.experiment == "both":
        heatmap_data, heatmap_eda_metrics, heatmap_plot_labels, heatmap_correlation = prepare_eda_heatmap_data(
            results_dir=args.heatmap_results_dir,
            eda_dir=args.heatmap_eda_dir,
            model=args.heatmap_model,
            protocol=args.heatmap_protocol,
            datasets=split_csv(args.heatmap_datasets),
            result_metric="dice",
            eda_metrics_arg=args.heatmap_eda_metrics,
            eda_labels_arg=args.heatmap_eda_labels,
            result_label=args.heatmap_result_label,
            correlation=args.heatmap_correlation,
            per_image=args.heatmap_per_image,
            dice_threshold=args.heatmap_dice_threshold,
            ultrabones_min_solidity=args.heatmap_ultrabones_min_solidity,
            filter_tnsc2020_target_bbox_one=args.heatmap_filter_tnsc2020_target_bbox_one,
            tnsc2020_min_aspect_ratio=args.heatmap_tnsc2020_min_aspect_ratio,
        )
        for metric in args.metrics:
            raw_plot_path = args.output_dir / f"{output_prefix}_{metric}_raw.{args.plot_format}"
            delta_plot_path = args.output_dir / f"{output_prefix}_{metric}_delta.{args.plot_format}"
            plot_combined_metric_pair(
                rows=all_rows,
                metric=metric,
                column_kind="raw",
                experiments=experiments,
                output_path=raw_plot_path,
                heatmap_data=heatmap_data,
                heatmap_eda_metrics=heatmap_eda_metrics,
                heatmap_plot_labels=heatmap_plot_labels,
                heatmap_correlation=heatmap_correlation,
            )
            plot_combined_metric_pair(
                rows=all_rows,
                metric=metric,
                column_kind="delta",
                experiments=experiments,
                output_path=delta_plot_path,
                heatmap_data=heatmap_data,
                heatmap_eda_metrics=heatmap_eda_metrics,
                heatmap_plot_labels=heatmap_plot_labels,
                heatmap_correlation=heatmap_correlation,
            )
            written_plots.extend([raw_plot_path, delta_plot_path])
    else:
        config = EXPERIMENT_CONFIG[args.experiment]
        raw_plot_path = args.output_dir / f"{output_prefix}_raw.{args.plot_format}"
        delta_plot_path = args.output_dir / f"{output_prefix}_delta.{args.plot_format}"
        plot_metric_lines(
            rows=all_rows,
            metrics=args.metrics,
            column_kind="raw",
            x_label=str(config["x_label"]),
            baseline_value=float(config["baseline_value"]),
            output_path=raw_plot_path,
        )
        plot_metric_lines(
            rows=all_rows,
            metrics=args.metrics,
            column_kind="delta",
            x_label=str(config["x_label"]),
            baseline_value=float(config["baseline_value"]),
            output_path=delta_plot_path,
        )
        written_plots.extend([raw_plot_path, delta_plot_path])

    print(f"Wrote summary to {summary_path}")
    for plot_path in written_plots:
        print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
