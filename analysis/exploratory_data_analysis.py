from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
from skimage import measure

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.common import bbox_from_mask, connected_component_masks
from datasets.label_text import concept_for
from datasets.loader import add_dataset_args, build_decoder_from_args


GROUP_KEYS = (
    "split",
    "category",
    "pathology",
    "birads",
    "label",
    "anatomy",
    "target_class_name",
    "labels",
)
STAT_COLUMNS = (
    "target_area_pixels",
    "target_area_fraction",
    "bbox_width",
    "bbox_height",
    "bbox_area_pixels",
    "bbox_area_fraction",
    "target_bbox_area_ratio",
    "aspect_ratio_feret",
    "circularity",
    "convexity",
    "solidity",
)

SHAPE_MIN_AREA = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis for a registered ultrasound dataset."
    )
    add_dataset_args(parser)
    parser.set_defaults(dataset="busi")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of decoded samples to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Defaults to analysis/eda/<dataset>.",
    )
    return parser.parse_args()


def metadata_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "+".join(str(item) for item in value if str(item))
    return str(value)


def infer_target_label(dataset_key: str, metadata: dict[str, object]) -> str:
    if metadata.get("target_class_name"):
        return str(metadata["target_class_name"])

    if metadata.get("labels"):
        return metadata_value(metadata["labels"])

    for key in ("label", "anatomy", "target_class_id"):
        label = metadata.get(key)
        if label is None or str(label) == "":
            continue
        try:
            return concept_for(dataset_key, label=label)
        except (KeyError, ValueError):
            return str(label)

    try:
        return concept_for(dataset_key)
    except (KeyError, ValueError):
        return dataset_key


def _convex_hull_points(component: np.ndarray) -> np.ndarray:
    hull = measure.regionprops(measure.label(component))[0].image_convex
    ys, xs = np.nonzero(hull)
    if xs.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack([xs.astype(float), ys.astype(float)])


def feret_aspect_ratio(component: np.ndarray) -> float:
    points = _convex_hull_points(component)
    if len(points) < 2:
        return float("nan")

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except Exception:
        hull_points = points

    spans: list[float] = []
    n_points = len(hull_points)
    for i in range(n_points):
        p0 = hull_points[i]
        p1 = hull_points[(i + 1) % n_points]
        edge = p1 - p0
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= 0.0:
            continue
        direction = edge / edge_length
        normal = np.array([-direction[1], direction[0]])
        projections = points @ normal
        spans.append(float(projections.max() - projections.min() + 1.0))

    if not spans:
        return float("nan")

    max_feret = 0.0
    for i in range(n_points):
        deltas = hull_points[i + 1 :] - hull_points[i]
        if len(deltas):
            max_feret = max(max_feret, float(np.sqrt(np.sum(deltas * deltas, axis=1)).max()) + 1.0)
    min_feret = min(spans)
    if max_feret <= 0.0:
        return float("nan")
    return max(0.0, min(1.0, min_feret / max_feret))


def component_shape_descriptors(component: np.ndarray) -> dict[str, float]:
    props = measure.regionprops(measure.label(component))[0]
    area = float(props.area)
    convex_area = float(props.area_convex)
    perimeter = float(measure.perimeter(component, neighborhood=8))
    convex_perimeter = float(measure.perimeter(props.image_convex, neighborhood=8))

    circularity = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0.0 else float("nan")
    convexity = (convex_perimeter / perimeter) if perimeter > 0.0 else float("nan")
    solidity = (area / convex_area) if convex_area > 0.0 else float("nan")
    if math.isfinite(circularity):
        circularity = max(0.0, min(1.0, circularity))
    if math.isfinite(convexity):
        convexity = max(0.0, min(1.0, convexity))
    if math.isfinite(solidity):
        solidity = max(0.0, min(1.0, solidity))

    return {
        "aspect_ratio_feret": feret_aspect_ratio(component),
        "circularity": circularity,
        "convexity": convexity,
        "solidity": solidity,
    }


def box_descriptors(component: np.ndarray, image_area: int) -> dict[str, float | int]:
    """Per-bounding-box metrics for one connected component (keys == STAT_COLUMNS)."""
    component_area = int(component.sum())
    bbox = bbox_from_mask(component)
    if bbox is None:
        bbox_width = bbox_height = bbox_area = 0
    else:
        x_min, y_min, x_max, y_max = [int(value) for value in bbox]
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        bbox_area = bbox_width * bbox_height
    return {
        "target_area_pixels": component_area,
        "target_area_fraction": component_area / image_area,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area_pixels": bbox_area,
        "bbox_area_fraction": bbox_area / image_area,
        "target_bbox_area_ratio": component_area / bbox_area if bbox_area > 0 else float("nan"),
        **component_shape_descriptors(component),
    }


def sample_record(dataset_key: str, sample: Any) -> dict[str, Any]:
    """One image record: shared image-level fields plus a `boxes` list (one per bounding box)."""
    mask = (np.asarray(sample.mask) > 0).astype(np.uint8)
    height, width = mask.shape[:2]
    image_height, image_width = np.asarray(sample.image).shape[:2]
    image_area = max(height * width, 1)
    metadata = dict(sample.metadata or {})

    components = connected_component_masks(mask, min_area=SHAPE_MIN_AREA)
    boxes = [
        {"box_index": index, **box_descriptors(component, image_area)}
        for index, component in enumerate(components)
    ]

    record: dict[str, Any] = {
        "dataset": dataset_key,
        "sample_id": sample.sample_id,
        "image_height": int(image_height),
        "image_width": int(image_width),
        "mask_height": int(height),
        "mask_width": int(width),
        "target_label": infer_target_label(dataset_key, metadata),
        "has_target": bool(boxes),
    }
    for key in GROUP_KEYS:
        record[key] = metadata_value(metadata.get(key))
    record["boxes"] = boxes
    return record


def collect_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    decoder = build_decoder_from_args(args)
    dataset_key = args.dataset.lower()
    total: int | None = None
    if hasattr(decoder, "count_samples"):
        try:
            total = decoder.count_samples(max_samples=args.max_samples)
        except Exception:
            total = None

    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(decoder.iter_samples(max_samples=args.max_samples), start=1):
        records.append(sample_record(dataset_key, sample))
        if idx == 1 or idx % 1000 == 0 or (total is not None and idx == total):
            if total is None:
                print(f"Decoded {idx} samples...", flush=True)
            else:
                pct = 100.0 * idx / max(total, 1)
                print(f"Decoded {idx}/{total} samples ({pct:.1f}%)...", flush=True)
    return records


def all_boxes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every image record's boxes into a single list of box dicts."""
    return [box for record in records for box in record["boxes"]]


def finite_values(boxes: list[dict[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for box in boxes:
        try:
            value = float(box[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def count_non_empty(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    counts = Counter(str(row[column]) for row in rows if str(row.get(column, "")))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_summary(dataset_key: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    image_sizes = Counter(
        f"{record['image_width']}x{record['image_height']}" for record in records
    )
    metadata_group_counts = {
        key: count_non_empty(records, key)
        for key in GROUP_KEYS
        if count_non_empty(records, key)
    }
    boxes = all_boxes(records)

    return {
        "dataset": dataset_key,
        "num_images": len(records),
        "num_boxes": len(boxes),
        "num_empty_masks": sum(1 for record in records if not record["has_target"]),
        "target_class_counts": count_non_empty(records, "target_label"),
        "target_class_count": len(count_non_empty(records, "target_label")),
        "metadata_group_counts": metadata_group_counts,
        "image_size_counts": dict(
            sorted(image_sizes.items(), key=lambda item: (-item[1], item[0]))
        ),
        "stats": {
            column: summarize_values(finite_values(boxes, column))
            for column in STAT_COLUMNS
        },
    }


def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def setup_matplotlib_cache() -> None:
    mpl_config_dir = Path("analysis") / ".mplconfig"
    xdg_cache_dir = Path("analysis") / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir.resolve()))


def display_counts(summary: dict[str, Any]) -> tuple[str, dict[str, int]]:
    groups = summary["metadata_group_counts"]
    if "category" in groups:
        return "Category counts", groups["category"]
    if "pathology" in groups:
        return "Pathology counts", groups["pathology"]
    return "Target counts", summary["target_class_counts"]


def format_stat(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def plot_dashboard(records: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> None:
    setup_matplotlib_cache()
    boxes = all_boxes(records)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
        }
    )

    fig = plt.figure(figsize=(12.8, 9.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.58])
    ax_headline = fig.add_subplot(grid[0, 0])
    ax_counts = fig.add_subplot(grid[0, 1])
    ax_sizes = fig.add_subplot(grid[0, 2])
    ax_area = fig.add_subplot(grid[1, 0])
    ax_bbox = fig.add_subplot(grid[1, 1])
    ax_ratio = fig.add_subplot(grid[1, 2])
    ax_table = fig.add_subplot(grid[2, :])

    dataset_label = summary["dataset"].upper()
    widths = [int(record["image_width"]) for record in records]
    heights = [int(record["image_height"]) for record in records]
    image_size_range = (
        f"W {min(widths)}-{max(widths)}, H {min(heights)}-{max(heights)}"
        if records
        else "n/a"
    )

    ax_headline.axis("off")
    headline = (
        f"{dataset_label} EDA\n\n"
        f"Images: {summary['num_images']}\n"
        f"Boxes: {summary['num_boxes']}\n"
        f"Empty masks: {summary['num_empty_masks']}\n"
        f"Image size range: {image_size_range}\n"
        f"Target classes: {summary['target_class_count']}"
    )
    ax_headline.text(
        0.02,
        0.96,
        headline,
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.5,
        transform=ax_headline.transAxes,
    )

    counts_title, counts = display_counts(summary)
    labels = list(counts)
    values = [counts[label] for label in labels]
    ax_counts.bar(labels, values, color="#4C72B0", alpha=0.75)
    ax_counts.set_title(counts_title)
    ax_counts.set_ylabel("Images")
    ax_counts.tick_params(axis="x", rotation=30)

    ax_sizes.scatter(widths, heights, s=18, alpha=0.45, color="#55A868", edgecolors="none")
    ax_sizes.set_title("Image size distribution")
    ax_sizes.set_xlabel("Width")
    ax_sizes.set_ylabel("Height")

    target_area_fraction = finite_values(boxes, "target_area_fraction")
    bbox_area_fraction = finite_values(boxes, "bbox_area_fraction")
    target_bbox_ratio = finite_values(boxes, "target_bbox_area_ratio")
    hist_bins = min(30, max(8, int(math.sqrt(max(len(boxes), 1)))))

    ax_area.hist(target_area_fraction, bins=hist_bins, color="#C44E52", alpha=0.75)
    ax_area.set_title("Target area fraction")
    ax_area.set_xlabel("Box area / image area")
    ax_area.set_ylabel("Boxes")

    ax_bbox.hist(bbox_area_fraction, bins=hist_bins, color="#8172B3", alpha=0.75)
    ax_bbox.set_title("BBox area fraction")
    ax_bbox.set_xlabel("BBox area / image area")
    ax_bbox.set_ylabel("Boxes")

    ax_ratio.hist(target_bbox_ratio, bins=hist_bins, color="#CCB974", alpha=0.82)
    ax_ratio.set_title("Target coverage within bbox")
    ax_ratio.set_xlabel("Box area / bbox area")
    ax_ratio.set_ylabel("Boxes")

    ax_table.axis("off")
    stat_rows = []
    for label, column in (
        ("Target frac", "target_area_fraction"),
        ("BBox frac", "bbox_area_fraction"),
        ("Target/BBox", "target_bbox_area_ratio"),
        ("Solidity", "solidity"),
    ):
        stats = summary["stats"][column]
        stat_rows.append(
            [
                label,
                format_stat(stats["mean"]),
                format_stat(stats["std"]),
                format_stat(stats["min"]),
                format_stat(stats["max"]),
            ]
        )

    table = ax_table.table(
        cellText=stat_rows,
        colLabels=["Metric", "Mean", "SD", "Min", "Max"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.22, 0.14, 0.14, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.55)
    ax_table.set_title("Summary statistics", pad=12)

    fig.suptitle("Exploratory Dataset Analysis", fontsize=14, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("analysis") / "eda" / args.dataset.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(args)
    if not records:
        raise SystemExit(f"No samples decoded for dataset '{args.dataset}'.")

    summary = build_summary(args.dataset.lower(), records)
    write_json(records, output_dir / "sample_stats.json")
    write_json(summary, output_dir / "dataset_summary.json")
    plot_dashboard(records, summary, output_dir / "eda_summary.png")

    print(f"Wrote {len(records)} image records to {output_dir / 'sample_stats.json'}")
    print(f"Wrote summary to {output_dir / 'dataset_summary.json'}")
    print(f"Wrote dashboard to {output_dir / 'eda_summary.png'}")


if __name__ == "__main__":
    main()
