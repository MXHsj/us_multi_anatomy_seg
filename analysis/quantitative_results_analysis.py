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


# Result metrics are the y-axis columns read straight from per_sample_metrics.csv.
# x-axis metrics are read from the EDA sample_stats.csv columns at run time.
RESULT_METRICS = (*METRIC_NAMES, "infer_ms")

CORRELATION_METHODS = ("pearson", "log_pearson", "spearman")

DATASET_LABELS = json.loads((ROOT_DIR / "datasets" / "datasets.json").read_text())["labels"]


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_csv_arg(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return allowed
    names = split_csv(value)
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise SystemExit(
            f"Unsupported {label}: {', '.join(unknown)}. "
            f"Expected one of: {', '.join(allowed)}"
        )
    if not names:
        raise SystemExit(f"At least one {label} must be selected.")
    return names


def labels_for(metrics: tuple[str, ...], labels_value: str) -> dict[str, str]:
    """Pair each metric with the label in the same position (extras ignored)."""
    return dict(zip(metrics, split_csv(labels_value)))


def parse_datasets_arg(value: str) -> tuple[str, ...] | None:
    if not value.strip():
        return None
    return split_csv(value)


def str2bool(value: str) -> bool:
    return value.strip().lower() == "true"


def compute_derived_metrics(df: pd.DataFrame, derived_metrics: tuple[str, ...]) -> pd.DataFrame:
    """Compute each requested derived metric from the result columns."""
    df = df.copy()
    for metric in derived_metrics:
        if metric == "fn_per_gt":
            df[metric] = 1.0 - df["recall"]
        elif metric == "fp_per_gt":
            df[metric] = df["relative_area_error"] + (1.0 - df["recall"])
        else:
            raise NotImplementedError(f"Derived metric not implemented: {metric}")
    return df


def result_dir(results_dir: Path, model: str, dataset: str, protocol: str) -> Path:
    return results_dir / f"{model}_{protocol}_{dataset}"


def discover_datasets(results_dir: Path, model: str, protocol: str) -> list[str]:
    prefix = f"{model}_{protocol}_"
    return sorted(path.name[len(prefix) :] for path in results_dir.glob(f"{prefix}*") if path.is_dir())


def discover_eda_metrics(eda_dir: Path, datasets: list[str]) -> tuple[str, ...]:
    """Read available x-axis metrics from the EDA sample_stats.csv column headers."""
    metrics: list[str] = []
    seen: set[str] = set()
    for dataset in datasets:
        stats_path = eda_dir / dataset / "sample_stats.csv"
        if not stats_path.exists():
            continue
        for column in pd.read_csv(stats_path, nrows=0).columns:
            if column != "sample_id" and column not in seen:
                seen.add(column)
                metrics.append(column)
    return tuple(metrics)


def load_dataset_points(
    results_dir: Path,
    eda_dir: Path,
    model: str,
    dataset: str,
    protocol: str,
    eda_metrics: tuple[str, ...],
    derived_metrics: tuple[str, ...],
) -> pd.DataFrame:
    metrics_path = result_dir(results_dir, model, dataset, protocol) / "per_sample_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    stats_path = eda_dir / dataset / "sample_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    wanted = {"sample_id", *eda_metrics}
    results = pd.read_csv(metrics_path)
    stats = pd.read_csv(stats_path, usecols=lambda column: column in wanted)
    for metric in eda_metrics:
        if metric not in stats.columns:
            stats[metric] = math.nan
    merged = results.merge(stats, on="sample_id", how="left")
    if derived_metrics:
        merged = compute_derived_metrics(merged, derived_metrics)
    return merged


def plot_label(metric: str, labels: dict[str, str] | None = None) -> str:
    """Axis label: the supplied plot label, else derived from the metric name."""
    if labels and metric in labels:
        return labels[metric]
    return metric.replace("_", " ").capitalize()


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


def corr_value(points: pd.DataFrame, x_metric: str, y_metric: str, method: str) -> float:
    """Correlation between x and y under the requested coefficient (NaN if undefined)."""
    pts = points[[x_metric, y_metric]]
    if method == "log_pearson":
        pts = pts[pts[x_metric] > 0].copy()
        pts[x_metric] = pts[x_metric].map(math.log10)
    if len(pts) < 2 or pts[x_metric].nunique() < 2 or pts[y_metric].nunique() < 2:
        return float("nan")
    if method == "spearman":
        return pts.corr(method="spearman").iloc[0, 1]
    if method in ("pearson", "log_pearson"):
        return pts.corr(method="pearson").iloc[0, 1]
    raise NotImplementedError(f"Correlation method not implemented: {method}")


def corr_label(points: pd.DataFrame, x_metric: str, y_metric: str, log_x: bool) -> str:
    method = "log_pearson" if log_x else "pearson"
    value = corr_value(points, x_metric, y_metric, method)
    return "r=NA" if math.isnan(value) else f"r={value:.2f}"


def apply_x_axis_limits(axis: Any, log_x: bool, x_min: float, x_max: float) -> None:
    if log_x:
        axis.set_xscale("log")
        axis.set_xlim(right=x_max)
    else:
        axis.set_xlim(x_min, x_max)


def apply_y_axis_limits(axis: Any, y_metric: str) -> None:
    # Temporarily let fp/fn auto-scale per subplot instead of a fixed y-range.
    return
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
    plot_labels: dict[str, str],
) -> None:
    plt = setup_matplotlib()

    datasets = list(data_by_dataset)
    nrows = math.ceil(len(datasets) / ncols)
    # Temporarily let fp/fn scale per-subplot instead of sharing one y-axis.
    sharey = y_metric not in ("fp_per_gt", "fn_per_gt")
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.2 * nrows),
        sharey=sharey,
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
        axis.set_xlabel(plot_label(x_metric, plot_labels))
        apply_y_axis_limits(axis, y_metric)
        axis.grid(True, alpha=0.3)

    for axis in axes.ravel()[len(datasets) :]:
        axis.set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(plot_label(y_metric, plot_labels))

    fig.suptitle(
        f"{model} ({protocol}) - {plot_label(y_metric, plot_labels)} vs {plot_label(x_metric, plot_labels)}"
    )
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
    plot_labels: dict[str, str],
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
    axis.set_xlabel(plot_label(x_metric, plot_labels))
    axis.set_ylabel(plot_label(y_metric, plot_labels))
    apply_y_axis_limits(axis, y_metric)
    axis.set_title(
        f"{model} ({protocol}) - {plot_label(y_metric, plot_labels)} vs {plot_label(x_metric, plot_labels)}"
    )
    axis.legend(fontsize=7, markerscale=1.5, loc="best")
    axis.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmaps(
    data_by_dataset: dict[str, pd.DataFrame],
    output_path: Path,
    y_metrics: tuple[str, ...],
    eda_metrics: tuple[str, ...],
    method: str,
    model: str,
    protocol: str,
    plot_labels: dict[str, str],
) -> None:
    plt = setup_matplotlib()

    datasets = list(data_by_dataset)
    fig, axes = plt.subplots(
        len(y_metrics),
        1,
        figsize=(0.6 * len(datasets) + 3, 2.4 * len(y_metrics) + 1),
        squeeze=False,
        constrained_layout=True,
    )

    image = None
    for axis, y_metric in zip(axes.ravel(), y_metrics):
        matrix = [
            [
                corr_value(finite_points(data_by_dataset[dataset], x_metric, y_metric, False), x_metric, y_metric, method)
                for dataset in datasets
            ]
            for x_metric in eda_metrics
        ]
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        axis.set_yticks(range(len(eda_metrics)), [plot_label(metric, plot_labels) for metric in eda_metrics])
        axis.set_xticks(
            range(len(datasets)),
            [DATASET_LABELS.get(dataset, dataset.upper()) for dataset in datasets],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.tick_params(axis="x", bottom=False)
        axis.set_title(plot_label(y_metric, plot_labels))
        axis.grid(False)
        for row, x_metric in enumerate(eda_metrics):
            for col in range(len(datasets)):
                value = matrix[row][col]
                if not math.isnan(value):
                    axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6)

    fig.colorbar(image, ax=axes.ravel().tolist(), label=f"{plot_label(method)} r", shrink=0.6)
    fig.suptitle(f"{model} ({protocol}) - {plot_label(method)} correlation")
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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--result-labels", default="", help="Comma-separated axis labels for --result-metrics.")
    parser.add_argument(
        "--derived-metrics",
        default="",
        help="Comma-separated derived (y-axis) metrics, e.g. fn_per_gt,fp_per_gt.",
    )
    parser.add_argument("--derived-labels", default="", help="Comma-separated axis labels for --derived-metrics.")
    parser.add_argument(
        "--eda-metrics",
        default="target_bbox_area_ratio",
        help=(
            "Comma-separated EDA (x-axis) metrics or 'all'. "
            "Available metrics are read from the EDA sample_stats.csv column headers."
        ),
    )
    parser.add_argument("--eda-labels", default="", help="Comma-separated axis labels for --eda-metrics.")
    parser.add_argument(
        "--plot-kind",
        choices=["per_dataset", "overlay", "both"],
        default="per_dataset",
    )
    parser.add_argument("--scatter", type=str2bool, default=True, help="Generate scatter figures (true/false).")
    parser.add_argument("--heatmap", type=str2bool, default=True, help="Generate correlation heatmaps (true/false).")
    parser.add_argument(
        "--correlations",
        default="log_pearson",
        help=f"Comma-separated heatmap correlation coefficients. Available: {', '.join(CORRELATION_METHODS)}.",
    )
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log-scaled x-axis and append _logx to output filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result_metrics = parse_csv_arg(args.result_metrics, RESULT_METRICS, "result metric")
    derived_metrics = split_csv(args.derived_metrics)
    y_metrics = (*result_metrics, *derived_metrics)
    requested_datasets = parse_datasets_arg(args.datasets)
    datasets = list(requested_datasets) if requested_datasets else discover_datasets(args.results_dir, args.model, args.protocol)
    if not datasets:
        raise SystemExit(f"No datasets found for {args.model}_{args.protocol}_* in {args.results_dir}.")

    available_eda_metrics = discover_eda_metrics(args.eda_dir, datasets)
    if not available_eda_metrics:
        raise SystemExit(f"No EDA sample_stats.csv columns found under {args.eda_dir}.")
    eda_metrics = parse_csv_arg(args.eda_metrics, available_eda_metrics, "EDA metric")
    plot_labels = {
        **labels_for(result_metrics, args.result_labels),
        **labels_for(derived_metrics, args.derived_labels),
        **labels_for(eda_metrics, args.eda_labels),
    }

    data_by_dataset: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        try:
            data_by_dataset[dataset] = load_dataset_points(
                args.results_dir,
                args.eda_dir,
                args.model,
                dataset,
                args.protocol,
                eda_metrics,
                derived_metrics,
            )
        except FileNotFoundError as error:
            print(f"Skipping {dataset}: {error}", file=sys.stderr)

    if not data_by_dataset:
        raise SystemExit("No datasets had both result metrics and EDA sample_stats.csv files.")

    correlations = parse_csv_arg(args.correlations, CORRELATION_METHODS, "correlation method")

    plot_kinds = ("per_dataset", "overlay") if args.plot_kind == "both" else (args.plot_kind,)
    scatter_total = len(y_metrics) * len(eda_metrics) * len(plot_kinds) if args.scatter else 0
    heatmap_total = len(correlations) if args.heatmap else 0
    total = scatter_total + heatmap_total
    index = 0

    if args.scatter:
        for y_metric in y_metrics:
            for x_metric in eda_metrics:
                for plot_kind in plot_kinds:
                    index += 1
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
                            plot_labels,
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
                            plot_labels,
                        )
                    print(f"[{index}/{total}] wrote {path}", flush=True)

    if args.heatmap:
        for method in correlations:
            index += 1
            path = args.output_dir / f"results_vs_eda_heatmap_{args.protocol}_{args.model}_{method}.png"
            plot_correlation_heatmaps(
                data_by_dataset,
                path,
                y_metrics,
                eda_metrics,
                method,
                args.model,
                args.protocol,
                plot_labels,
            )
            print(f"[{index}/{total}] wrote {path}", flush=True)
    print("")


if __name__ == "__main__":
    main()
