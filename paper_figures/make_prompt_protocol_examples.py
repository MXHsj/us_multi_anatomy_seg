from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from benchmarks.metrics import SegmentationMetrics
from benchmarks.medsam_inference import load_medsam_model, medsam_inference
from benchmarks.samus_inference import (
    build_model as build_samus_model,
    foreground_click_xy,
    prepare_samus_tensor,
    samus_point_inference,
    scale_click_to_model_space,
)
from datasets.breast_BLUSG import BreastBLUSGDecoder
from datasets.common import (
    bbox_from_mask,
    ensure_three_channels,
    normalize_to_uint8,
    prepare_medsam_image,
)


# Consistent colors across all prompt protocols.
GT_MASK_COLOR = np.array([0, 176, 80], dtype=np.float32)      # green
PRED_MASK_COLOR = np.array([214, 39, 40], dtype=np.float32)   # red
GT_BBOX_COLOR = (255, 212, 0)                                # yellow
ALT_BBOX_COLOR = (255, 122, 0)                               # orange
POINT_COLOR = (0, 229, 255)                                  # cyan
TP_COLOR = np.array([31, 158, 68], dtype=np.uint8)            # green
FP_COLOR = np.array([214, 39, 40], dtype=np.uint8)            # red
FN_COLOR = np.array([38, 100, 235], dtype=np.uint8)           # blue


@dataclass
class Example:
    sample_id: str
    image: np.ndarray
    gt_mask: np.ndarray
    bbox: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create same-size, borderless PNG elements for Methods-section "
            "segmentation target and prompting protocol examples."
        )
    )
    parser.add_argument("--sample-id", default="case027")
    parser.add_argument(
        "--comparison-sample-id",
        default="case027",
        help=(
            "BLUSG sample used for the standalone GT-versus-prediction map. "
            "Defaults to the same sample used by the prompt-protocol tiles."
        ),
    )
    parser.add_argument("--dataset-root", default=str(ROOT / "datasets/BLUSG"))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "paper_figures/prompt_protocol_examples"),
    )
    parser.add_argument(
        "--medsam-checkpoint",
        default=str(ROOT / "work_dir/MedSAM/medsam_vit_b.pth"),
    )
    parser.add_argument(
        "--samus-checkpoint",
        default=str(ROOT / "work_dir/SAMUS/ckp/SAMUS.pth"),
    )
    parser.add_argument(
        "--samus-base-checkpoint",
        default=str(ROOT / "work_dir/SAMUS/checkpoints/sam_vit_b_01ec64.pth"),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda", "cuda:0"])
    parser.add_argument("--jitter-fraction", type=float, default=0.20)
    parser.add_argument("--jitter-seed", type=int, default=7)
    parser.add_argument("--text-model", default="medsam3", choices=["medsam3", "medicalsam3"])
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        requested = "cuda:0"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return torch.device("cpu")
    return device


def load_example(dataset_root: str | Path, sample_id: str) -> Example:
    decoder = BreastBLUSGDecoder(root=dataset_root, include_other=True)
    for sample in decoder.iter_samples(max_samples=None):
        if sample.sample_id == sample_id:
            image = normalize_to_uint8(ensure_three_channels(sample.image))
            gt_mask = (sample.mask > 0).astype(np.uint8)
            bbox = bbox_from_mask(gt_mask)
            if bbox is None:
                raise ValueError(f"Sample {sample_id} has an empty mask.")
            return Example(sample.sample_id, image, gt_mask, bbox)
    raise KeyError(f"Could not find sample_id={sample_id!r} under {dataset_root}.")


def jitter_bbox(
    bbox: np.ndarray,
    height: int,
    width: int,
    fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bbox.astype(float)
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    dx = rng.uniform(-fraction * bw, fraction * bw)
    dy = rng.uniform(-fraction * bh, fraction * bh)
    scale = rng.uniform(1.0 - fraction * 0.5, 1.0 + fraction * 0.5)
    cx = (x0 + x1) / 2.0 + dx
    cy = (y0 + y1) / 2.0 + dy
    new_bw = bw * scale
    new_bh = bh * scale
    out = np.array(
        [
            np.clip(cx - new_bw / 2.0, 0, width - 1),
            np.clip(cy - new_bh / 2.0, 0, height - 1),
            np.clip(cx + new_bw / 2.0, 0, width - 1),
            np.clip(cy + new_bh / 2.0, 0, height - 1),
        ],
        dtype=np.int32,
    )
    out[2] = max(out[2], out[0] + 1)
    out[3] = max(out[3], out[1] + 1)
    return out


def run_medsam(example: Example, checkpoint: str | Path, device: torch.device) -> np.ndarray:
    from segment_anything import sam_model_registry

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")

    model = load_medsam_model(sam_model_registry, checkpoint, device)
    height, width = example.gt_mask.shape
    image_1024 = prepare_medsam_image(example.image)
    image_tensor = (
        torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    box_1024 = example.bbox[None, :] / np.array([width, height, width, height]) * 1024
    with torch.no_grad():
        image_embedding = model.image_encoder(image_tensor)
        pred = medsam_inference(model, image_embedding, box_1024, height, width)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred.astype(np.uint8)


def run_samus(
    example: Example,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    device: torch.device,
) -> tuple[np.ndarray, tuple[int, int]]:
    # SAMUS is lightweight for one-sample CPU inference and safer on macOS CPU.
    samus_device = "cuda:0" if device.type == "cuda" else "cpu"
    args = argparse.Namespace(
        checkpoint=str(checkpoint),
        no_auto_download_checkpoint=True,
        checkpoint_file_id="",
        sam_ckpt=str(base_checkpoint),
        device=samus_device,
        encoder_input_size=256,
        low_image_size=128,
        vit_name="vit_b",
    )
    model = build_samus_model(args)
    height, width = example.gt_mask.shape
    click_xy = foreground_click_xy(example.gt_mask)
    if click_xy is None:
        raise ValueError(f"Sample {example.sample_id} has no foreground point.")
    click_256 = scale_click_to_model_space(
        click_xy,
        src_height=height,
        src_width=width,
        dst_size=256,
    )
    image_tensor = prepare_samus_tensor(example.image, size=256, device=samus_device).unsqueeze(0)
    pred = samus_point_inference(model, image_tensor, click_256[None, :, :], [(height, width)])[0]
    del model
    if samus_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pred.astype(np.uint8), click_xy


def read_metric_row(path: Path, sample_id: str) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("sample_id") == sample_id:
                return row
    return None


def text_visualization_path(text_model: str, sample_id: str) -> Path:
    if text_model == "medsam3":
        return ROOT / "results" / "medsam3_label_prompt" / "blusg" / "visualizations" / f"{sample_id}.png"
    return ROOT / "results" / "medicalsam3_label_prompt" / "blusg" / "visualizations" / f"{sample_id}.png"


def text_metric_path(text_model: str) -> Path:
    if text_model == "medsam3":
        return ROOT / "results" / "medsam3_label_prompt" / "blusg" / "per_sample_metrics.csv"
    return ROOT / "results" / "medicalsam3_label_prompt" / "blusg" / "per_sample_metrics.csv"


def crop_text_prediction_panel(vis_path: Path) -> Image.Image:
    """Crop the prediction image panel from the saved 1x3 text benchmark figure."""
    image = Image.open(vis_path).convert("RGB")
    width, height = image.size
    panel = image.crop((int(width * 2 / 3), 0, width, height))
    arr = np.asarray(panel)
    # Remove title/margin area by finding non-white rendered image content below
    # the title band.
    search = arr[55:, :, :]
    nonwhite = np.any(search < 248, axis=-1)
    ys, xs = np.where(nonwhite)
    if len(xs) == 0 or len(ys) == 0:
        return panel
    pad = 6
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, panel.width - 1)
    y0 = max(int(ys.min()) + 55 - pad, 0)
    y1 = min(int(ys.max()) + 55 + pad, panel.height - 1)
    return panel.crop((x0, y0, x1 + 1, y1 + 1))


def extract_text_prediction_mask(vis_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Recover a text-prompt mask from the saved red prediction overlay.

    Raw text-prompt prediction masks are not currently saved in ``results/``.
    The saved benchmark figure is a real code output produced from the predicted
    mask, so this recovers the red overlay and resizes it to the raw image size.
    """
    panel = crop_text_prediction_panel(vis_path)
    panel = panel.resize((target_shape[1], target_shape[0]), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(panel).astype(np.int16)
    red_dominance = arr[..., 0] - np.maximum(arr[..., 1], arr[..., 2])
    return (red_dominance > 6).astype(np.uint8)


def bbox_region_mask(shape: tuple[int, int], bbox: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    mask[max(y0, 0) : min(y1 + 1, shape[0]), max(x0, 0) : min(x1 + 1, shape[1])] = 1
    return mask


def alpha_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    base = normalize_to_uint8(ensure_three_channels(image)).astype(np.float32)
    out = base.copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def white_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = 255
    return out


def draw_bboxes(
    image: np.ndarray,
    boxes: list[tuple[np.ndarray, tuple[int, int, int], int]],
) -> np.ndarray:
    pil = Image.fromarray(normalize_to_uint8(ensure_three_channels(image)))
    draw = ImageDraw.Draw(pil)
    for bbox, color, width in boxes:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        for offset in range(width):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)
    return np.asarray(pil)


def draw_point(image: np.ndarray, point_xy: tuple[int, int], radius: int = 7) -> np.ndarray:
    pil = Image.fromarray(normalize_to_uint8(ensure_three_channels(image)))
    draw = ImageDraw.Draw(pil)
    x, y = point_xy
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=POINT_COLOR,
        outline=(0, 0, 0),
        width=2,
    )
    return np.asarray(pil)


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    rgb = np.full((*gt.shape, 3), 255, dtype=np.uint8)
    rgb[np.logical_and(gt_b, pred_b)] = TP_COLOR
    rgb[np.logical_and(~gt_b, pred_b)] = FP_COLOR
    rgb[np.logical_and(gt_b, ~pred_b)] = FN_COLOR
    return rgb


def resize_binary_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(np.uint8)
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (target_shape[1], target_shape[0]),
        resample=Image.Resampling.NEAREST,
    )
    return (np.asarray(resized) > 0).astype(np.uint8)


def gt_prediction_map(
    gt: np.ndarray,
    pred: np.ndarray,
    target_shape: tuple[int, int],
    prediction_opacity: float = 0.80,
) -> np.ndarray:
    """Render GT as green fill with a transparent red prediction overlay.

    Unlike ``error_map``, this encodes the two masks directly and does not
    partition pixels into TP/FP/FN categories. The prediction is rendered with
    20% transparency so the GT remains visible in overlapping regions.
    """
    gt_b = resize_binary_mask(gt, target_shape).astype(bool)
    pred_b = resize_binary_mask(pred, target_shape).astype(bool)

    rgb = np.full((*target_shape, 3), 255, dtype=np.uint8)
    rgb[gt_b] = GT_MASK_COLOR.astype(np.uint8)
    rgb_float = rgb.astype(np.float32)
    rgb_float[pred_b] = (
        (1.0 - prediction_opacity) * rgb_float[pred_b]
        + prediction_opacity * PRED_MASK_COLOR
    )
    return np.clip(rgb_float, 0, 255).astype(np.uint8)


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalize_to_uint8(image)).save(path)


def clean_previous_outputs(out_dir: Path) -> None:
    for pattern in (
        "protocol_*.png",
        "protocol_*.svg",
        "panels/*.png",
        "panels/*.json",
        "elements/*.png",
    ):
        for path in out_dir.glob(pattern):
            path.unlink()


def save_element_tiles(
    out_dir: Path,
    example: Example,
    medsam_pred: np.ndarray,
    samus_pred: np.ndarray,
    text_pred: np.ndarray,
    jittered_bbox: np.ndarray,
    click_xy: tuple[int, int],
    comparison_gt: np.ndarray,
    comparison_pred: np.ndarray,
) -> dict[str, str]:
    elements = out_dir / "elements"
    elements.mkdir(parents=True, exist_ok=True)

    gt_overlay = alpha_overlay(example.image, example.gt_mask, GT_MASK_COLOR)
    text_constrained = np.logical_and(
        text_pred.astype(bool),
        bbox_region_mask(example.gt_mask.shape, example.bbox),
    ).astype(np.uint8)

    tiles: dict[str, np.ndarray] = {
        "bbox_01_raw_bmode.png": example.image,
        "bbox_02_raw_gt_mask.png": white_mask(example.gt_mask),
        "bbox_03_gt_mask_with_gt_bbox.png": draw_bboxes(
            white_mask(example.gt_mask),
            [(example.bbox, GT_BBOX_COLOR, 3)],
        ),
        "bbox_04_gt_overlay_with_gt_bbox.png": draw_bboxes(
            gt_overlay,
            [(example.bbox, GT_BBOX_COLOR, 3)],
        ),
        "bbox_05_bbox_jitter_example.png": draw_bboxes(
            gt_overlay,
            [(example.bbox, GT_BBOX_COLOR, 3), (jittered_bbox, ALT_BBOX_COLOR, 3)],
        ),
        "bbox_06_medsam_pred_overlay.png": alpha_overlay(example.image, medsam_pred, PRED_MASK_COLOR),
        "bbox_07_error_map.png": error_map(example.gt_mask, medsam_pred),
        "point_01_raw_bmode.png": example.image,
        "point_02_raw_gt_mask.png": white_mask(example.gt_mask),
        "point_03_gt_overlay_with_point.png": draw_point(gt_overlay, click_xy),
        "point_04_samus_pred_overlay.png": alpha_overlay(example.image, samus_pred, PRED_MASK_COLOR),
        "point_05_error_map.png": error_map(example.gt_mask, samus_pred),
        "text_01_raw_bmode.png": example.image,
        "text_02_raw_gt_mask.png": white_mask(example.gt_mask),
        "text_03_gt_overlay_for_text_prompt.png": gt_overlay,
        "text_04_text_prompt_pred_overlay.png": alpha_overlay(example.image, text_pred, PRED_MASK_COLOR),
        "text_05_text_prompt_pred_with_gt_bbox_constraint.png": draw_bboxes(
            alpha_overlay(example.image, text_constrained, PRED_MASK_COLOR),
            [(example.bbox, GT_BBOX_COLOR, 3)],
        ),
        "text_06_error_map.png": error_map(example.gt_mask, text_pred),
        "text_07_constrained_error_map.png": error_map(example.gt_mask, text_constrained),
        "comparison_01_gt_pred_overlay.png": gt_prediction_map(
            comparison_gt,
            comparison_pred,
            target_shape=example.gt_mask.shape,
        ),
    }

    saved: dict[str, str] = {}
    for name, tile in tiles.items():
        path = elements / name
        save_png(path, tile)
        saved[name] = str(path.relative_to(ROOT))
    return saved


def relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(out_dir)

    example = load_example(args.dataset_root, args.sample_id)
    height, width = example.gt_mask.shape
    jittered = jitter_bbox(example.bbox, height, width, args.jitter_fraction, args.jitter_seed)
    device = resolve_device(args.device)

    print(f"Loaded BLUSG sample {example.sample_id}: image={example.image.shape}, mask={example.gt_mask.shape}")
    print(f"Using device for MedSAM: {device}")
    medsam_pred = run_medsam(example, args.medsam_checkpoint, device)
    print("MedSAM prediction generated.")
    comparison_example = load_example(args.dataset_root, args.comparison_sample_id)
    if comparison_example.sample_id == example.sample_id:
        comparison_pred = medsam_pred
    else:
        comparison_pred = run_medsam(comparison_example, args.medsam_checkpoint, device)
    print(
        "Standalone GT/prediction comparison generated for "
        f"{comparison_example.sample_id}."
    )
    samus_pred, click_xy = run_samus(example, args.samus_checkpoint, args.samus_base_checkpoint, device)
    print(f"SAMUS prediction generated with click={click_xy}.")

    text_vis_path = text_visualization_path(args.text_model, example.sample_id)
    if not text_vis_path.exists():
        raise FileNotFoundError(f"Saved text-prompt visualization not found: {text_vis_path}")
    text_pred = extract_text_prediction_mask(text_vis_path, example.gt_mask.shape)
    text_metric_row = read_metric_row(text_metric_path(args.text_model), example.sample_id)

    saved_tiles = save_element_tiles(
        out_dir=out_dir,
        example=example,
        medsam_pred=medsam_pred,
        samus_pred=samus_pred,
        text_pred=text_pred,
        jittered_bbox=jittered,
        click_xy=click_xy,
        comparison_gt=comparison_example.gt_mask,
        comparison_pred=comparison_pred,
    )

    metadata = {
        "sample_id": example.sample_id,
        "dataset": "BLUSG",
        "raw_data_source": relative(args.dataset_root),
        "tile_size_width_height": [int(width), int(height)],
        "color_palette": {
            "gt_mask_overlay": "#00B050",
            "pred_mask_overlay": "#D62728",
            "gt_bbox": "#FFD400",
            "jittered_or_alternative_bbox": "#FF7A00",
            "point_prompt": "#00E5FF",
            "error_true_positive": "#1F9E44",
            "error_false_positive": "#D62728",
            "error_false_negative": "#2664EB",
            "comparison_gt_fill": "#00B050",
            "comparison_prediction_fill": "#D62728 at 80% opacity",
        },
        "element_pngs": saved_tiles,
        "medsam_exact_single_sample_inference": {
            "checkpoint": relative(args.medsam_checkpoint),
            "metrics": SegmentationMetrics().compute(example.gt_mask, medsam_pred),
        },
        "samus_exact_single_sample_inference": {
            "checkpoint": relative(args.samus_checkpoint),
            "metrics": SegmentationMetrics().compute(example.gt_mask, samus_pred),
            "click_xy": list(click_xy),
        },
        "standalone_gt_prediction_comparison": {
            "sample_id": comparison_example.sample_id,
            "model": "MedSAM",
            "checkpoint": relative(args.medsam_checkpoint),
            "metrics": SegmentationMetrics().compute(
                comparison_example.gt_mask,
                comparison_pred,
            ),
            "rendering": (
                "green GT fill with 20% transparent red prediction fill on white"
            ),
        },
        "text_prompt_output": {
            "text_model": args.text_model,
            "visualization_source": str(text_vis_path.relative_to(ROOT)),
            "benchmark_metrics_source": str(text_metric_path(args.text_model).relative_to(ROOT)),
            "benchmark_dice": float(text_metric_row["dice"]) if text_metric_row else None,
            "benchmark_iou": float(text_metric_row["iou"]) if text_metric_row else None,
            "extracted_mask_metrics": SegmentationMetrics().compute(example.gt_mask, text_pred),
            "concept": text_metric_row.get("concept", "breast tumor") if text_metric_row else "breast tumor",
            "note": (
                "Raw text-prompt prediction masks are not stored in results. "
                "The text-prompt tile/error map is recovered from the saved benchmark "
                "visualization's red prediction overlay."
            ),
        },
        "error_map_legend": {
            "green": "true positive",
            "red": "false positive",
            "blue": "false negative",
            "white": "true negative/background",
        },
    }
    (out_dir / "protocol_example_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved individual PNG elements to {out_dir / 'elements'}")


if __name__ == "__main__":
    main()
