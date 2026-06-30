from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import secrets
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "analysis" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "analysis" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from skimage import measure

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.common import ensure_three_channels, normalize_to_uint8
from datasets.loader import add_dataset_args, build_decoder_from_args


RESULTS_DIR = ROOT / "results"


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def optional_seed(value: str) -> int | None:
    if value.strip() == "":
        return None
    return int(value)


def optional_text(value: str) -> str | None:
    value = value.strip()
    return value or None


def folder_text(value: str) -> str:
    return value.strip().replace("/", "_")


def load_default_datasets() -> str:
    labels_path = ROOT / "datasets" / "datasets.json"
    return ",".join(json.loads(labels_path.read_text())["labels"])


def load_dataset_labels() -> dict[str, str]:
    labels_path = ROOT / "datasets" / "datasets.json"
    return json.loads(labels_path.read_text())["labels"]


def result_dir(model: str, dataset: str, protocol: str, results_dir: Path) -> Path:
    return results_dir / f"{model}_{protocol}_{dataset}"


def load_results(
    model: str,
    dataset: str,
    protocol: str,
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folder = result_dir(model, dataset, protocol, results_dir)
    metrics_path = folder / "per_sample_metrics.csv"
    summary_path = folder / "summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return pd.read_csv(metrics_path), summary


def sample_threshold_rows(
    df: pd.DataFrame,
    metric: str,
    lower: float,
    upper: float,
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in results.")
    values = pd.to_numeric(df[metric], errors="coerce")
    rows = df.loc[(values >= lower) & (values < upper)].copy()
    if len(rows) > count:
        positions = rng.choice(len(rows), size=count, replace=False)
        rows = rows.iloc[positions].copy()
    return rows.reset_index(drop=True)


def repo_path(path: str | Path | None) -> str:
    if path in (None, ""):
        return ""
    path = Path(path)
    return str(path if path.is_absolute() else ROOT / path)


def load_decoder(dataset: str, dataset_root: str = ""):
    parser = argparse.ArgumentParser(add_help=False)
    add_dataset_args(parser, include_camus=True)
    args = parser.parse_args([])
    args.dataset = dataset
    args.dataset_root = repo_path(dataset_root)
    args.no_auto_download = True
    return build_decoder_from_args(args)


def load_samples(decoder, sample_ids: list[str]) -> list[Any]:
    samples = []
    missing = []
    for sample_id in sample_ids:
        try:
            samples.append(decoder.load_sample(sample_id))
        except Exception:
            missing.append(sample_id)

    if not missing:
        return samples

    found = {str(sample.sample_id): sample for sample in samples}
    missing_set = set(missing)
    for sample in decoder.iter_samples(max_samples=None):
        sample_id = str(sample.sample_id)
        if sample_id in missing_set:
            found[sample_id] = sample
            missing_set.remove(sample_id)
        if not missing_set:
            break
    if missing_set:
        raise KeyError(f"Could not load sample_id(s): {sorted(missing_set)}")
    return [found[sample_id] for sample_id in sample_ids]


def run_model_on_samples(model: str, samples: list[Any], summary: dict[str, Any], **kwargs):
    if model == "ultrasam":
        module = importlib.import_module("benchmarks.ultrasam_inference")
        defaults = {
            "checkpoint": summary.get("checkpoint", "work_dir/UltraSam/UltraSam.pth"),
            "checkpoint_url": summary.get("checkpoint_url", module.DEFAULT_ULTRASAM_CHECKPOINT_URL),
            "ultrasam_dir": summary.get("ultrasam_dir") or "UltraSam",
            "config": summary.get("config", module.DEFAULT_ULTRASAM_CONFIG),
            "box_padding": int(summary.get("box_padding", 0)),
            "bbox_mode": summary.get("bbox_mode", "individual"),
            "batch_size": int(summary.get("batch_size", 4)),
            "num_workers": 0,
        }
        defaults.update(kwargs)
        return module.run_ultrasam_on_samples(samples, **defaults)

    if model == "medsam":
        module = importlib.import_module("benchmarks.medsam_inference")
        defaults = {
            "checkpoint": summary.get("checkpoint", "work_dir/MedSAM/medsam_vit_b.pth"),
            "box_padding": int(summary.get("box_padding", 0)),
            "bbox_mode": summary.get("bbox_mode", "individual"),
            "bbox_jitter_prob": 0.0,
        }
        defaults.update(kwargs)
        return module.run_medsam_on_samples(samples, **defaults)

    if model == "samus":
        module = importlib.import_module("benchmarks.samus_inference")
        defaults = {
            "checkpoint": summary.get("checkpoint", "work_dir/SAMUS/ckp/SAMUS.pth"),
            "sam_ckpt": summary.get("sam_ckpt", "work_dir/SAMUS/checkpoints/sam_vit_b_01ec64.pth"),
            "box_padding": int(summary.get("box_padding", 0)),
            "bbox_mode": summary.get("bbox_mode", "individual"),
            "batch_size": int(summary.get("batch_size", 16)),
            "num_workers": 0,
        }
        defaults.update(kwargs)
        return module.run_samus_on_samples(samples, **defaults)

    raise ValueError(f"Unsupported model: {model}")


def _draw_gt_outline(axis, gt_mask: np.ndarray, color: str, linewidth: float) -> None:
    for contour in measure.find_contours(gt_mask.astype(float), 0.5):
        axis.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def _draw_bboxes(axis, bbox, color: str, linewidth: float) -> None:
    for x0, y0, x1, y1 in np.asarray(bbox).reshape(-1, 4):
        axis.add_patch(
            Rectangle(
                (float(x0), float(y0)),
                float(x1 - x0 + 1),
                float(y1 - y0 + 1),
                edgecolor=color,
                facecolor=(0, 0, 0, 0),
                linewidth=linewidth,
            )
        )


def _bbox_union(bbox: np.ndarray) -> tuple[float, float, float, float] | None:
    boxes = np.asarray(bbox, dtype=float).reshape(-1, 4)
    if boxes.size == 0:
        return None
    return (
        float(np.min(boxes[:, 0])),
        float(np.min(boxes[:, 1])),
        float(np.max(boxes[:, 2])),
        float(np.max(boxes[:, 3])),
    )


def _crop_bounds_from_bbox(
    bbox: np.ndarray,
    image_shape: tuple[int, ...],
    width_fraction: float,
    height_fraction: float,
    target_width_fraction: float,
) -> tuple[int, int, int, int] | None:
    union = _bbox_union(bbox)
    if union is None:
        return None

    height, width = image_shape[:2]
    x0, y0, x1, y1 = union
    target_width = max(x1 - x0 + 1.0, 1.0)
    target_height = max(y1 - y0 + 1.0, 1.0)
    target_center_x = (x0 + x1) / 2.0
    target_center_y = (y0 + y1) / 2.0

    inset_aspect = (width * width_fraction) / max(height * height_fraction, 1e-6)
    crop_width = max(
        target_width / target_width_fraction,
        (target_height / target_width_fraction) * inset_aspect,
    )
    crop_height = crop_width / inset_aspect
    crop_width = min(max(crop_width, 1.0), float(width))
    crop_height = min(max(crop_height, 1.0), float(height))

    crop_x0 = int(round(target_center_x - crop_width / 2.0))
    crop_y0 = int(round(target_center_y - crop_height / 2.0))
    crop_x0 = min(max(crop_x0, 0), max(width - int(round(crop_width)), 0))
    crop_y0 = min(max(crop_y0, 0), max(height - int(round(crop_height)), 0))
    return crop_x0, crop_y0, min(crop_x0 + int(round(crop_width)), width), min(
        crop_y0 + int(round(crop_height)), height
    )


def _draw_roblus_inset(
    axis,
    image: np.ndarray,
    bbox: np.ndarray,
    *,
    gt_mask: np.ndarray | None = None,
    pred_mask: np.ndarray | None = None,
    draw_annotations: bool = False,
    draw_bbox: bool = True,
    pred_color: str,
    pred_alpha: float,
    show_image_background: bool,
    gt_outline_color: str,
    gt_outline_width: float,
    bbox_color: str,
    bbox_linewidth: float,
) -> None:
    width_fraction = 0.60
    height_fraction = 0.50
    crop_bounds = _crop_bounds_from_bbox(bbox, image.shape, width_fraction, height_fraction, 0.85)
    if crop_bounds is None:
        return

    height, width = image.shape[:2]
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_bounds
    inset_axis = axis.inset_axes(
        [
            width - 0.5 - width * width_fraction,
            height - 0.5 - height * height_fraction,
            width * width_fraction,
            height * height_fraction,
        ],
        transform=axis.transData,
    )
    if show_image_background:
        inset_axis.imshow(image[crop_y0:crop_y1, crop_x0:crop_x1], cmap="gray", aspect="auto")
    else:
        black = np.zeros((crop_y1 - crop_y0, crop_x1 - crop_x0, 3), dtype=np.uint8)
        inset_axis.imshow(black, zorder=-5, aspect="auto")

    if pred_mask is not None:
        pred_crop = pred_mask[crop_y0:crop_y1, crop_x0:crop_x1]
        overlay = np.zeros((*pred_crop.shape, 4), dtype=float)
        overlay[..., :3] = np.array(to_rgb(pred_color))
        overlay[..., 3] = (pred_crop > 0) * (pred_alpha if show_image_background else 1.0)
        inset_axis.imshow(overlay, zorder=0, aspect="auto")

    if draw_annotations and gt_mask is not None:
        _draw_gt_outline(
            inset_axis,
            gt_mask[crop_y0:crop_y1, crop_x0:crop_x1],
            gt_outline_color,
            gt_outline_width * 1.8,
        )
    if draw_annotations and draw_bbox:
        relative_bbox = np.asarray(bbox, dtype=float).reshape(-1, 4).copy()
        relative_bbox[:, [0, 2]] -= crop_x0
        relative_bbox[:, [1, 3]] -= crop_y0
        _draw_bboxes(inset_axis, relative_bbox, bbox_color, bbox_linewidth * 1.8)

    inset_axis.set_xticks([])
    inset_axis.set_yticks([])
    for spine in inset_axis.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#ffb347")
        spine.set_linewidth(1.0)
        spine.set_linestyle("dotted")


def _plot_result_pair(
    image_ax,
    overlay_ax,
    result: Any,
    sample_metrics: dict[str, Any],
    *,
    dataset: str,
    pred_color: str,
    pred_alpha: float,
    show_image_background: bool,
    gt_outline_color: str,
    bbox_color: str,
) -> None:
    image_ax.axis("off")
    overlay_ax.axis("off")

    image = ensure_three_channels(normalize_to_uint8(result.image))
    pred_rgb = np.array(to_rgb(pred_color))
    is_roblus = str(getattr(result.sample, "dataset", "")).lower() == "roblus"

    image_ax.imshow(image, cmap="gray", aspect="auto")
    _draw_bboxes(image_ax, result.bbox, bbox_color, 0.8)
    image_ax.text(
        0.02,
        0.96,
        dataset,
        transform=image_ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
    )

    if is_roblus:
        _draw_roblus_inset(
            image_ax,
            image,
            result.bbox,
            draw_annotations=True,
            draw_bbox=True,
            pred_color=pred_color,
            pred_alpha=pred_alpha,
            show_image_background=True,
            gt_outline_color=gt_outline_color,
            gt_outline_width=0.9,
            bbox_color=bbox_color,
            bbox_linewidth=0.8,
        )

    if show_image_background:
        overlay_ax.imshow(image, cmap="gray", aspect="auto")
        overlay = np.zeros((*result.pred_mask.shape, 4), dtype=float)
        overlay[..., :3] = pred_rgb
        overlay[..., 3] = (result.pred_mask > 0) * pred_alpha
    else:
        overlay_ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=overlay_ax.transAxes,
                facecolor="black",
                edgecolor="none",
                zorder=-10,
            )
        )
        overlay = np.zeros((*result.pred_mask.shape, 3), dtype=float)
        overlay[result.pred_mask > 0] = pred_rgb
    overlay_ax.imshow(overlay, zorder=0, aspect="auto")
    _draw_gt_outline(overlay_ax, result.gt_mask, gt_outline_color, 0.9)

    if is_roblus:
        _draw_roblus_inset(
            overlay_ax,
            image,
            result.bbox,
            gt_mask=result.gt_mask,
            pred_mask=result.pred_mask,
            draw_annotations=True,
            draw_bbox=False,
            pred_color=pred_color,
            pred_alpha=pred_alpha,
            show_image_background=show_image_background,
            gt_outline_color=gt_outline_color,
            gt_outline_width=0.9,
            bbox_color=bbox_color,
            bbox_linewidth=0.8,
        )

    csv_dice = sample_metrics.get("dice", math.nan)
    csv_assd = sample_metrics.get("assd", math.nan)
    overlay_ax.text(
        0.02,
        0.96,
        f"Dice: {csv_dice * 100:.2f}%\nASSD: {csv_assd:.2f}",
        transform=overlay_ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
    )


def plot_dataset_example(
    columns: list[tuple[str, Any, dict[str, Any]]],
    save_path: Path,
    *,
    pred_color: str,
    pred_alpha: float,
    show_image_background: bool,
    gt_outline_color: str,
    bbox_color: str,
) -> None:
    if not columns:
        return

    ncols = len(columns)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(3.6 * ncols, 6.0),
        gridspec_kw={"height_ratios": [2.85, 3.15]},
        squeeze=False,
    )

    for col, (dataset, result, sample_metrics) in enumerate(columns):
        _plot_result_pair(
            axes[0, col],
            axes[1, col],
            result,
            sample_metrics,
            dataset=dataset,
            pred_color=pred_color,
            pred_alpha=pred_alpha,
            show_image_background=show_image_background,
            gt_outline_color=gt_outline_color,
            bbox_color=bbox_color,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save qualitative threshold figures.")
    parser.add_argument("--model", default="ultrasam")
    parser.add_argument("--protocol", default="gt_bbox")
    parser.add_argument("--datasets", default=load_default_datasets())
    parser.add_argument("--result-metric", default="dice")
    parser.add_argument("--lower-threshold", required=True)
    parser.add_argument("--upper-threshold", required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=optional_seed, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "figures" / "qualitative")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--pred-color", default="#ff4a43")
    parser.add_argument("--pred-alpha", type=float, default=0.45)
    parser.add_argument("--show-image-background", type=str2bool, default=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-id", type=optional_text, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lower_threshold = float(args.lower_threshold)
    upper_threshold = float(args.upper_threshold)
    if args.samples_per_dataset <= 0:
        raise SystemExit("--samples-per-dataset must be positive.")
    if lower_threshold >= upper_threshold:
        raise SystemExit("--lower-threshold must be smaller than --upper-threshold.")

    seed = args.seed if args.seed is not None else secrets.randbelow(2**32)
    rng = np.random.default_rng(seed)
    datasets = split_csv(args.datasets)
    dataset_labels = load_dataset_labels()
    threshold_folder = (
        args.output_dir
        / f"{args.result_metric}_{folder_text(args.lower_threshold)}_{folder_text(args.upper_threshold)}"
    )

    print(f"Seed: {seed}")
    run_config: dict[str, Any] = {
        "model": args.model,
        "protocol": args.protocol,
        "result_metric": args.result_metric,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "num_examples": args.samples_per_dataset,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "datasets": datasets,
        "selected_sample_ids": {},
        "figures": {},
        "skipped": {},
    }

    examples_by_dataset: dict[str, list[tuple[Any, dict[str, Any]]]] = {}

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        try:
            df, summary = load_results(args.model, dataset, args.protocol, args.results_dir)
            rows = sample_threshold_rows(
                df,
                args.result_metric,
                lower_threshold,
                upper_threshold,
                args.samples_per_dataset,
                rng,
            )
        except (FileNotFoundError, KeyError) as exc:
            print(f"  Skipping: {exc}")
            run_config["skipped"][dataset] = str(exc)
            continue

        if rows.empty:
            print("  Skipping: no matching samples")
            run_config["skipped"][dataset] = "no matching samples"
            continue

        sample_ids = rows["sample_id"].astype(str).tolist()
        run_config["selected_sample_ids"][dataset] = sample_ids
        csv_metrics = rows.assign(sample_id=sample_ids).set_index("sample_id").to_dict("index")

        try:
            decoder = load_decoder(dataset, summary.get("dataset_root", ""))
            samples = load_samples(decoder, sample_ids)
            device = args.device or (f"cuda:{args.gpu_id}" if args.gpu_id is not None else None)
            model_kwargs = {"device": device or summary.get("device", "cuda:0")}
            results = run_model_on_samples(args.model, samples, summary, **model_kwargs)
        except Exception as exc:
            print(f"  Skipping: inference failed: {exc}")
            run_config["skipped"][dataset] = f"inference failed: {exc}"
            continue
        if not results:
            print("  Skipping: inference returned no results")
            run_config["skipped"][dataset] = "inference returned no results"
            continue

        examples_by_dataset[dataset] = [
            (result, csv_metrics.get(str(result.sample.sample_id), {})) for result in results
        ]

    num_figures = args.samples_per_dataset if examples_by_dataset else 0

    for index in range(num_figures):
        columns = [
            (dataset_labels.get(dataset, dataset.upper()), examples[index][0], examples[index][1])
            for dataset, examples in examples_by_dataset.items()
            if index < len(examples)
        ]
        if not columns:
            continue

        save_path = threshold_folder / (
            f"{args.model}_{args.protocol}_{args.result_metric}_"
            f"{folder_text(args.lower_threshold)}_{folder_text(args.upper_threshold)}_"
            f"seed{seed}_example{index + 1}.png"
        )
        plot_dataset_example(
            columns,
            save_path,
            pred_color=args.pred_color,
            pred_alpha=args.pred_alpha,
            show_image_background=args.show_image_background,
            gt_outline_color="white",
            bbox_color="yellow",
        )
        run_config["figures"][f"example_{index + 1}"] = str(save_path)
        print(f"  Saved: {save_path}")

    threshold_folder.mkdir(parents=True, exist_ok=True)
    config_path = threshold_folder / "run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
