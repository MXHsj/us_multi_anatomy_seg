from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.metrics import METRIC_NAMES


DERIVED_RESULT_METRICS = ("fn_per_gt", "fp_per_gt")
RESULT_METRICS = (*METRIC_NAMES, "infer_ms", *DERIVED_RESULT_METRICS)
EDA_METRICS = (
    "image_height",
    "image_width",
    "mask_height",
    "mask_width",
    "target_area_pixels",
    "target_area_fraction",
    "bbox_width",
    "bbox_height",
    "bbox_area_pixels",
    "bbox_area_fraction",
    "target_bbox_area_ratio",
    "component_count",
    "largest_component_area_pixels",
    "shape_component_count",
    "shape_component_area_pixels",
    "component_weighted_bbox_width",
    "component_weighted_bbox_height",
    "component_weighted_bbox_area_pixels",
    "component_weighted_bbox_area_fraction",
    "component_weighted_target_bbox_area_ratio",
    "aspect_ratio_feret",
    "circularity",
    "convexity",
    "solidity",
)
X_METRICS = (*EDA_METRICS, *DERIVED_RESULT_METRICS)

DATASET_LABELS = json.loads((ROOT_DIR / "datasets" / "datasets.json").read_text())["labels"]


def parse_csv_arg(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return allowed
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(parsed) - set(allowed))
    if unknown:
        raise SystemExit(
            f"Unsupported {label}: {', '.join(unknown)}. "
            f"Expected one of: {', '.join(allowed)}"
        )
    if not parsed:
        raise SystemExit(f"At least one {label} must be selected.")
    return parsed


def parse_datasets_arg(value: str) -> tuple[str, ...] | None:
    if not value.strip():
        return None
    return tuple(dataset.strip() for dataset in value.split(",") if dataset.strip())


def result_dir(results_dir: Path, model: str, dataset: str, protocol: str) -> Path:
    return results_dir / f"{model}_{protocol}_{dataset}"


def discover_datasets(results_dir: Path, model: str, protocol: str) -> list[str]:
    prefix = f"{model}_{protocol}_"
    return sorted(path.name[len(prefix) :] for path in results_dir.glob(f"{prefix}*") if path.is_dir())


def _bbox_area_fraction(row: pd.Series) -> float:
    boxes = json.loads(row["bbox"])
    boxes_array = pd.DataFrame(boxes, columns=["x0", "y0", "x1", "y1"]).astype(float)
    box_area = float(((boxes_array["x1"] - boxes_array["x0"]) * (boxes_array["y1"] - boxes_array["y0"])).sum())
    image_area = float(row["height"]) * float(row["width"])
    return box_area / image_area if image_area else math.nan


def load_dataset_points(
    results_dir: Path,
    eda_dir: Path,
    model: str,
    dataset: str,
    protocol: str,
) -> pd.DataFrame:
    metrics_path = result_dir(results_dir, model, dataset, protocol) / "per_sample_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    stats_path = eda_dir / dataset / "sample_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    results = pd.read_csv(metrics_path)
    stats = pd.read_csv(stats_path, usecols=lambda column: column == "sample_id" or column in EDA_METRICS)
    merged = results.merge(stats, on="sample_id", how="left")
    merged["bbox_image_area_fraction"] = merged.apply(_bbox_area_fraction, axis=1)
    merged["fn_per_gt"] = 1.0 - merged["recall"]
    merged["fp_per_gt"] = merged["relative_area_error"] + merged["fn_per_gt"]
    return merged


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
        "fn_per_gt": "FN / GT foreground",
        "fp_per_gt": "FP / GT foreground",
        "infer_ms": "Inference time (ms)",
        "image_height": "Image height",
        "image_width": "Image width",
        "mask_height": "Mask height",
        "mask_width": "Mask width",
        "target_area_pixels": "Target area (px)",
        "target_area_fraction": "Target area / image area",
        "bbox_width": "BBox width",
        "bbox_height": "BBox height",
        "bbox_area_pixels": "BBox area (px)",
        "bbox_area_fraction": "BBox area / image area",
        "target_bbox_area_ratio": "Target area / bbox area",
        "component_count": "Component count",
        "largest_component_area_pixels": "Largest component area (px)",
        "shape_component_count": "Shape component count",
        "shape_component_area_pixels": "Shape component area (px)",
        "component_weighted_bbox_width": "Component-weighted bbox width",
        "component_weighted_bbox_height": "Component-weighted bbox height",
        "component_weighted_bbox_area_pixels": "Component-weighted bbox area (px)",
        "component_weighted_bbox_area_fraction": "Component-weighted bbox / image area",
        "component_weighted_target_bbox_area_ratio": "Component-weighted target / bbox area",
        "aspect_ratio_feret": "Feret aspect ratio",
        "circularity": "Circularity",
        "convexity": "Convexity",
        "solidity": "Solidity",
    }
    return labels.get(metric, metric)


def setup_matplotlib() -> Any:
    mpl_config_dir = Path("analysis") / ".mplconfig"
    xdg_cache_dir = Path("analysis") / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir.resolve()))

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
        }
    )
    return plt


def finite_points(df: pd.DataFrame, x_metric: str, y_metric: str, log_x: bool) -> pd.DataFrame:
    points = df[[x_metric, y_metric]].apply(pd.to_numeric, errors="coerce")
    points = points.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if log_x:
        points = points[points[x_metric] > 0]
    return points


def corr_label(points: pd.DataFrame, x_metric: str, y_metric: str, log_x: bool) -> str:
    if len(points) < 2 or points[x_metric].nunique() < 2 or points[y_metric].nunique() < 2:
        return "r=NA"
    corr_points = points[[x_metric, y_metric]].copy()
    if log_x:
        corr_points[x_metric] = corr_points[x_metric].map(math.log10)
    return f"r={corr_points.corr().iloc[0, 1]:.2f}"


def apply_x_axis_limits(axis: Any, log_x: bool, x_min: float, x_max: float) -> None:
    if log_x:
        axis.set_xscale("log")
        axis.set_xlim(right=x_max)
    else:
        axis.set_xlim(x_min, x_max)


def apply_y_axis_limits(axis: Any, y_metric: str) -> None:
    if y_metric == "fp_per_gt":
        axis.set_ylim(0.0, 6.0)


def plot_per_dataset(
    data_by_dataset: dict[str, pd.DataFrame],
    output_path: Path,
    x_metric: str,
    y_metric: str,
    model: str,
    protocol: str,
    log_x: bool,
    ncols: int,
    x_min: float,
    x_max: float,
) -> None:
    plt = setup_matplotlib()

    datasets = list(data_by_dataset)
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.2 * nrows),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("tab20")

    for axis, dataset in zip(axes.ravel(), datasets):
        points = finite_points(data_by_dataset[dataset], x_metric, y_metric, log_x)
        color = cmap(datasets.index(dataset) % 20)
        axis.scatter(points[x_metric], points[y_metric], s=14, alpha=0.5, edgecolor="none", color=color)
        apply_x_axis_limits(axis, log_x, x_min, x_max)
        label = DATASET_LABELS.get(dataset, dataset.upper())
        axis.set_title(f"{label} (n={len(points)}, {corr_label(points, x_metric, y_metric, log_x)})")
        axis.set_xlabel(metric_label(x_metric))
        apply_y_axis_limits(axis, y_metric)
        axis.grid(True, alpha=0.3)

    for axis in axes.ravel()[len(datasets) :]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(metric_label(y_metric))

    fig.suptitle(f"{model} ({protocol}) - {metric_label(y_metric)} vs {metric_label(x_metric)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    data_by_dataset: dict[str, pd.DataFrame],
    output_path: Path,
    x_metric: str,
    y_metric: str,
    model: str,
    protocol: str,
    log_x: bool,
    x_min: float,
    x_max: float,
) -> None:
    plt = setup_matplotlib()

    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    for index, (dataset, data) in enumerate(data_by_dataset.items()):
        points = finite_points(data, x_metric, y_metric, log_x)
        label = DATASET_LABELS.get(dataset, dataset.upper())
        axis.scatter(
            points[x_metric],
            points[y_metric],
            s=14,
            alpha=0.5,
            edgecolor="none",
            color=cmap(index % 20),
            label=f"{label} (n={len(points)})",
        )

    apply_x_axis_limits(axis, log_x, x_min, x_max)
    axis.set_xlabel(metric_label(x_metric))
    axis.set_ylabel(metric_label(y_metric))
    apply_y_axis_limits(axis, y_metric)
    axis.set_title(f"{model} ({protocol}) - {metric_label(y_metric)} vs {metric_label(x_metric)}")
    axis.legend(fontsize=7, markerscale=1.5, loc="best")
    axis.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def output_name(
    model: str,
    protocol: str,
    y_metric: str,
    x_metric: str,
    plot_kind: str,
    log_x: bool,
) -> str:
    suffix = "_logx" if log_x else ""
    return f"results_vs_eda_{plot_kind}_{protocol}_{model}_{y_metric}_vs_{x_metric}{suffix}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot benchmark result metrics against EDA measurements."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--eda-dir", type=Path, default=Path("analysis") / "eda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "figures" / "results_vs_eda",
    )
    parser.add_argument("--model", default="ultrasam")
    parser.add_argument("--protocol", default="gt_bbox", choices=["gt_bbox", "jitter_bbox"])
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset list. Defaults to all available for model/protocol.",
    )
    parser.add_argument(
        "--result-metrics",
        default="dice",
        help=f"Comma-separated result metrics or 'all'. Available: {', '.join(RESULT_METRICS)}.",
    )
    parser.add_argument(
        "--eda-metrics",
        default="all",
        help=(
            "Comma-separated x-axis metrics or 'all'. "
            f"Available EDA measurements plus derived result metrics: {', '.join(X_METRICS)}."
        ),
    )
    parser.add_argument(
        "--plot-kind",
        choices=["per_dataset", "overlay", "both"],
        default="per_dataset",
    )
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log-scaled x-axis and append _logx to output filenames.",
    )
    args = parser.parse_args()

    result_metrics = parse_csv_arg(args.result_metrics, RESULT_METRICS, "result metric")
    eda_metrics = parse_csv_arg(args.eda_metrics, X_METRICS, "x-axis metric")
    requested_datasets = parse_datasets_arg(args.datasets)
    datasets = list(requested_datasets) if requested_datasets else discover_datasets(args.results_dir, args.model, args.protocol)
    if not datasets:
        raise SystemExit(f"No datasets found for {args.model}_{args.protocol}_* in {args.results_dir}.")

    data_by_dataset: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        try:
            data_by_dataset[dataset] = load_dataset_points(
                args.results_dir,
                args.eda_dir,
                args.model,
                dataset,
                args.protocol,
            )
        except FileNotFoundError as error:
            print(f"Skipping {dataset}: {error}", file=sys.stderr)

    if not data_by_dataset:
        raise SystemExit("No datasets had both result metrics and EDA sample_stats.csv files.")

    plot_kinds = ("per_dataset", "overlay") if args.plot_kind == "both" else (args.plot_kind,)
    saved_paths: list[Path] = []
    for y_metric in result_metrics:
        for x_metric in eda_metrics:
            for plot_kind in plot_kinds:
                path = args.output_dir / output_name(
                    args.model,
                    args.protocol,
                    y_metric,
                    x_metric,
                    plot_kind,
                    args.log_x,
                )
                if plot_kind == "per_dataset":
                    plot_per_dataset(
                        data_by_dataset,
                        path,
                        x_metric,
                        y_metric,
                        args.model,
                        args.protocol,
                        args.log_x,
                        args.ncols,
                        args.x_min,
                        args.x_max,
                    )
                else:
                    plot_overlay(
                        data_by_dataset,
                        path,
                        x_metric,
                        y_metric,
                        args.model,
                        args.protocol,
                        args.log_x,
                        args.x_min,
                        args.x_max,
                    )
                saved_paths.append(path)

    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
