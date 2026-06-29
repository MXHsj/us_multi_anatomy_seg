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

# Font size (points) for the correlation values printed inside heatmap cells.
HEATMAP_CELL_FONTSIZE = 5

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
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "figure.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
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


def build_groups(data_by_dataset: dict[str, pd.DataFrame], per_dataset: bool) -> dict[str, pd.DataFrame]:
    """Prepare data for plotting: combine all images, or keep one DataFrame per dataset.

    This is the only place ``per_dataset`` is used. Everything downstream just plots whatever
    DataFrame it is handed, oblivious to whether it is pooled or a single dataset.
    """
    if per_dataset:
        return {DATASET_LABELS.get(dataset, dataset.upper()): df for dataset, df in data_by_dataset.items()}
    return {"pooled": pd.concat(data_by_dataset.values(), ignore_index=True)}


def heat_text_color(plt: Any, value: float) -> str:
    """Black or white annotation text for an RdBu_r cell in the [-1, 1] correlation range."""
    red, green, blue, _ = plt.get_cmap("RdBu_r")((value + 1.0) / 2.0)
    return "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"


def plot_scatter(
    groups: dict[str, pd.DataFrame],
    output_path: Path,
    y_metric: str,
    eda_metrics: tuple[str, ...],
    title: str,
    log_x: bool,
    plot_labels: dict[str, str],
) -> None:
    """Scatter each group as its own row: columns = EDA metrics (x), y = the result metric.

    Spans a US-Letter-width page (8.5 in); every group adds one row of fixed height, so a single
    pooled group is one row and the per-dataset case stacks all datasets in one figure.
    """
    plt = setup_matplotlib()
    import seaborn as sns

    nrows, ncols = len(groups), len(eda_metrics)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8.5, 1.5 * nrows),
        sharey=True, squeeze=False, constrained_layout=True,
    )
    for row, (group, df) in enumerate(groups.items()):
        bottom_row = row == nrows - 1
        for axis, x_metric in zip(axes[row], eda_metrics):
            points = finite_points(df, x_metric, y_metric, log_x)
            sns.regplot(
                data=points,
                x=x_metric,
                y=y_metric,
                ax=axis,
                logx=log_x,
                ci=None,
                truncate=True,
                scatter_kws={"s": 2, "alpha": 0.5, "edgecolor": "none", "color": "#1f77b4"},
                line_kws={"color": "#d62728", "linewidth": 0.6},
            )
            if log_x:
                # Log x can't include 0; span the observed positive range instead of a fixed [0, 1].
                axis.set_xscale("log")
            else:
                # Both Dice and the EDA metrics live on a 0-1 scale: share it so plots are comparable.
                axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_title(corr_label(points, x_metric, y_metric, log_x))
            axis.set_xlabel(plot_label(x_metric, plot_labels) if bottom_row else "")
            axis.set_ylabel("")
            axis.grid(True, alpha=0.3)
        axes[row, 0].set_ylabel(f"{group}\n(n={len(df):,})")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    y_metrics: tuple[str, ...],
    eda_metrics: tuple[str, ...],
    method: str,
    title: str,
    plot_labels: dict[str, str],
) -> None:
    """Correlation heatmap of the given data: rows = EDA metrics, columns = result metrics."""
    plt = setup_matplotlib()

    matrix = [
        [corr_value(finite_points(df, x_metric, y_metric, False), x_metric, y_metric, method) for y_metric in y_metrics]
        for x_metric in eda_metrics
    ]

    fig, axis = plt.subplots(figsize=(3.0, 2.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_yticks(range(len(eda_metrics)), [plot_label(metric, plot_labels) for metric in eda_metrics])
    axis.set_xticks(
        range(len(y_metrics)),
        [plot_label(metric, plot_labels) for metric in y_metrics],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axis.tick_params(axis="x", bottom=False)
    axis.grid(False)
    for row in range(len(eda_metrics)):
        for col in range(len(y_metrics)):
            value = matrix[row][col]
            if not math.isnan(value):
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=HEATMAP_CELL_FONTSIZE, color=heat_text_color(plt, value))

    fig.colorbar(image, ax=axis, label=f"{plot_label(method)} r", shrink=0.8)
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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
        "--per-dataset",
        type=str2bool,
        default=False,
        help="false: combine all images into one figure. true: one figure per dataset.",
    )
    parser.add_argument(
        "--plot-format",
        default="png",
        choices=["png", "svg", "pdf"],
        help="File format for the saved figures. Default: png.",
    )
    parser.add_argument("--scatter", type=str2bool, default=True, help="Generate scatter figures (true/false).")
    parser.add_argument("--heatmap", type=str2bool, default=True, help="Generate correlation heatmaps (true/false).")
    parser.add_argument(
        "--correlations",
        default="log_pearson",
        help=f"Comma-separated heatmap correlation coefficients. Available: {', '.join(CORRELATION_METHODS)}.",
    )
    parser.add_argument(
        "--log-x",
        type=str2bool,
        default=False,
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

    # Prepare the data: one pooled DataFrame, or one per dataset. The plotters below are
    # oblivious to which — they just plot whatever DataFrame they are handed.
    groups = build_groups(data_by_dataset, args.per_dataset)
    suffix = "_logx" if args.log_x else ""

    # One name for the whole grouping: the single group's name when pooled, else "by_dataset".
    scope = next(iter(groups)) if len(groups) == 1 else "by_dataset"

    total = (len(y_metrics) if args.scatter else 0) + len(groups) * (len(correlations) if args.heatmap else 0)
    index = 0

    if args.scatter:
        # All groups go into one figure (one row per group).
        for y_metric in y_metrics:
            index += 1
            title = f"{args.model} ({args.protocol}) - {scope} - {plot_label(y_metric, plot_labels)}"
            path = args.output_dir / f"results_vs_eda_scatter_{args.protocol}_{args.model}_{scope}_{y_metric}{suffix}.{args.plot_format}"
            plot_scatter(groups, path, y_metric, eda_metrics, title, args.log_x, plot_labels)
            print(f"[{index}/{total}] wrote {path}", flush=True)

    if args.heatmap:
        for group, df in groups.items():
            for method in correlations:
                index += 1
                title = f"{args.model} ({args.protocol}) - {group} - {plot_label(method)} (n={len(df):,})"
                path = args.output_dir / f"results_vs_eda_heatmap_{args.protocol}_{args.model}_{group}_{method}.{args.plot_format}"
                plot_heatmap(df, path, y_metrics, eda_metrics, method, title, plot_labels)
                print(f"[{index}/{total}] wrote {path}", flush=True)
    print("")


if __name__ == "__main__":
    main()
