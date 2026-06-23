from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.common import (
    bboxes_from_mask,
    ensure_three_channels,
    normalize_to_uint8,
    prepare_medsam_image,
)
from datasets.loader import add_dataset_args, build_decoder_from_args
from benchmarks.eval_utils import (
    InferenceResult,
    METRIC_FIELDNAMES,
    TargetVisualizationCollector,
    build_metric_row,
    count_source_iterations,
    count_unique_source_samples,
    evenly_spaced_zero_based_indices,
    format_max_samples,
    parse_max_samples,
    summarize_metric_rows,
    summarize_target_class_metrics,
)
from benchmarks.metrics import SegmentationMetrics


DEFAULT_MEDSAM_CHECKPOINT_REPO_ID = "GleghornLab/medsam-vit-b"
DEFAULT_MEDSAM_CHECKPOINT_FILENAME = "medsam_vit_b.pth"


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(
    current: int,
    total: int | None,
    evaluated: int,
    skipped: int,
    sample_id: str,
    elapsed_s: float,
    bar_width: int = 24,
) -> None:
    if total and total > 0:
        frac = min(max(current / total, 0.0), 1.0)
        filled = min(int(round(frac * bar_width)), bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        total_str = str(total)
        progress = f"[{bar}] {frac * 100:5.1f}%"
    else:
        total_str = "?"
        progress = "[" + "?" * bar_width + "]"

    sample_label = sample_id if len(sample_id) <= 48 else f"...{sample_id[-45:]}"
    avg_rate = evaluated / elapsed_s if elapsed_s > 0 and evaluated > 0 else 0.0
    print(
        "\r"
        f"{progress} {current}/{total_str} | evaluated={evaluated} | skipped={skipped} "
        f"| avg={avg_rate:.2f} samples/s | elapsed={format_duration(elapsed_s)} "
        f"| last={sample_label}",
        end="",
        flush=True,
    )


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_torch_device(requested_device: str) -> torch.device:
    requested = torch.device(requested_device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        if mps_is_available():
            print(
                f"Requested device '{requested_device}' but CUDA is unavailable; using 'mps'.",
                flush=True,
            )
            return torch.device("mps")
        print(
            f"Requested device '{requested_device}' but CUDA is unavailable; using 'cpu'.",
            flush=True,
        )
        return torch.device("cpu")
    if requested.type == "mps" and not mps_is_available():
        if torch.cuda.is_available():
            print(
                f"Requested device '{requested_device}' but MPS is unavailable; using 'cuda:0'.",
                flush=True,
            )
            return torch.device("cuda:0")
        print(
            f"Requested device '{requested_device}' but MPS is unavailable; using 'cpu'.",
            flush=True,
        )
        return torch.device("cpu")
    return requested


def ensure_checkpoint(
    checkpoint: str | Path,
    auto_download: bool,
    repo_id: str,
    filename: str,
    revision: str,
) -> Path:
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        return checkpoint
    if not auto_download:
        raise FileNotFoundError(
            f"MedSAM checkpoint not found: {checkpoint}. "
            "Place medsam_vit_b.pth under work_dir/MedSAM or pass --checkpoint."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required to auto-download the MedSAM checkpoint. "
            "Install requirements or pass --no-auto-download-checkpoint and provide --checkpoint."
        ) from exc

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    )
    shutil.copy2(downloaded, checkpoint)
    return checkpoint


def load_medsam_model(sam_model_registry, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = sam_model_registry["vit_b"](checkpoint=None)
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def medsam_inference(medsam_model, img_embed: torch.Tensor, box_1024: np.ndarray, H: int, W: int) -> np.ndarray:
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    pred = (low_res_pred > 0.5).astype(np.uint8)
    if pred.ndim == 3:
        pred = pred.max(axis=0)
    return pred


def jitter_bbox(
    bbox: np.ndarray, jitter_frac: float, height: int, width: int
) -> np.ndarray:
    if jitter_frac <= 0:
        return bbox
    bbox_arr = np.asarray(bbox)
    w = bbox_arr[..., 2] - bbox_arr[..., 0]
    h = bbox_arr[..., 3] - bbox_arr[..., 1]
    dx = np.random.uniform(-jitter_frac * w, jitter_frac * w)
    dy = np.random.uniform(-jitter_frac * h, jitter_frac * h)
    new_x0 = np.clip(bbox_arr[..., 0] + dx, 0, width - 1)
    new_y0 = np.clip(bbox_arr[..., 1] + dy, 0, height - 1)
    new_x1 = np.clip(bbox_arr[..., 2] + dx, 0, width - 1)
    new_y1 = np.clip(bbox_arr[..., 3] + dy, 0, height - 1)
    return np.stack([new_x0, new_y0, new_x1, new_y1], axis=-1).astype(np.int32)


def get_total_iterations(decoder, max_samples: int | None) -> int | None:
    count_fn = getattr(decoder, "count_samples", None)
    if callable(count_fn):
        return count_fn(max_samples=max_samples)
    return max_samples


def evenly_spaced_indices(total: int | None, count: int) -> set[int]:
    if count <= 0:
        return set()
    if total is None or total <= 0:
        return set(range(count))
    save_count = min(count, total)
    return {int(round(idx)) for idx in np.linspace(0, total - 1, save_count)}


def save_vis(
    vis_dir: Path,
    sample_id: str,
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    bbox: np.ndarray,
):
    import matplotlib.pyplot as plt

    vis_dir.mkdir(parents=True, exist_ok=True)
    name = sample_id.replace("/", "_")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Image")

    ax[1].imshow(image)
    ax[1].imshow(gt_mask, alpha=0.45, cmap="Greens")
    for box in np.asarray(bbox).reshape(-1, 4):
        ax[1].add_patch(
            plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                edgecolor="yellow",
                facecolor=(0, 0, 0, 0),
                linewidth=2,
            )
        )
    ax[1].set_title("GT + Box Prompt")

    ax[2].imshow(image)
    ax[2].imshow(pred_mask, alpha=0.45, cmap="Reds")
    ax[2].set_title("MedSAM Prediction")

    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(vis_dir / f"{name}.png", dpi=140)
    plt.close(fig)


def run_medsam_on_samples(
    samples,
    *,
    device: str = "cuda:0",
    checkpoint: str | Path = "work_dir/MedSAM/medsam_vit_b.pth",
    checkpoint_repo_id: str = DEFAULT_MEDSAM_CHECKPOINT_REPO_ID,
    checkpoint_filename: str = DEFAULT_MEDSAM_CHECKPOINT_FILENAME,
    checkpoint_revision: str = "main",
    no_auto_download_checkpoint: bool = False,
    box_padding: int = 0,
    bbox_mode: str = "union",
    bbox_jitter_prob: float = 0.0,
    bbox_jitter_fraction: float = 0.2,
) -> list[InferenceResult]:
    """Run the normal MedSAM GT-box pipeline on already-decoded samples."""
    try:
        from segment_anything import sam_model_registry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "segment_anything is required for MedSAM inference."
        ) from exc

    resolved_checkpoint = ensure_checkpoint(
        checkpoint=str(checkpoint),
        auto_download=not no_auto_download_checkpoint,
        repo_id=checkpoint_repo_id,
        filename=checkpoint_filename,
        revision=checkpoint_revision,
    )
    torch_device = resolve_torch_device(device)
    model = load_medsam_model(sam_model_registry, resolved_checkpoint, torch_device)
    metrics_calculator = SegmentationMetrics()
    results: list[InferenceResult] = []

    for sample in samples:
        image_3c = ensure_three_channels(sample.image)
        height, width = image_3c.shape[:2]
        gt_mask = (sample.mask > 0).astype(np.uint8)
        bboxes = bboxes_from_mask(
            gt_mask,
            padding=box_padding,
            mode=bbox_mode,
        )
        if bboxes is None:
            continue
        if bbox_jitter_fraction > 0 and np.random.rand() < bbox_jitter_prob:
            bboxes = jitter_bbox(bboxes, bbox_jitter_fraction, height, width)

        image_1024 = prepare_medsam_image(image_3c)
        image_1024_tensor = (
            torch.tensor(image_1024)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(torch_device)
        )
        box_1024 = bboxes / np.array([width, height, width, height]) * 1024

        tic = time.perf_counter()
        with torch.no_grad():
            image_embedding = model.image_encoder(image_1024_tensor)
        pred_mask = medsam_inference(model, image_embedding, box_1024, height, width)
        infer_ms = (time.perf_counter() - tic) * 1000.0
        metric_bbox = bboxes[0] if bbox_mode == "union" and len(bboxes) == 1 else bboxes
        metrics = metrics_calculator.compute(gt_mask, pred_mask)
        results.append(
            InferenceResult(
                sample=sample,
                image=ensure_three_channels(normalize_to_uint8(image_3c)),
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                bbox=metric_bbox,
                metrics=metrics,
                infer_ms=infer_ms,
            )
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MedSAM inference with GT box prompts")
    add_dataset_args(parser, include_camus=True)
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument(
        "--no-auto-download-checkpoint",
        action="store_true",
        help="Require an existing local MedSAM checkpoint instead of downloading it.",
    )
    parser.add_argument(
        "--checkpoint-repo-id",
        type=str,
        default=DEFAULT_MEDSAM_CHECKPOINT_REPO_ID,
        help="Hugging Face model repo used when the MedSAM checkpoint is missing.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default=DEFAULT_MEDSAM_CHECKPOINT_FILENAME,
        help="Checkpoint filename inside --checkpoint-repo-id.",
    )
    parser.add_argument(
        "--checkpoint-revision",
        type=str,
        default="main",
        help="Hugging Face checkpoint repo revision.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--max-samples",
        type=parse_max_samples,
        default=None,
        metavar="N|all",
        help="Evaluate a positive integer cap or use 'all' for the full dataset. Defaults to 'all'.",
    )
    parser.add_argument("--box-padding", type=int, default=0)
    parser.add_argument(
        "--bbox-mode",
        choices=("union", "individual"),
        default="union",
        help=(
            "Use one bbox around the full target mask (union, default) or one bbox "
            "per connected component larger than 15 pixels (individual)."
        ),
    )
    parser.add_argument(
        "--bbox-jitter-prob",
        type=float,
        default=0.0,
        help="Whether to apply bbox degradation (0.0 = never, 1.0 = always, intermediate values = probability of degrading each sample)",
    )
    parser.add_argument(
        "--bbox-jitter-fraction",
        type=float,
        default=0.2,
        help="Max jitter as fraction of bbox size (per axis)",
    )
    parser.add_argument("--save-vis", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="results/medsam_test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder = build_decoder_from_args(args)
    total_iterations = get_total_iterations(decoder, args.max_samples)
    vis_indices = evenly_spaced_indices(total_iterations, args.save_vis)
    total_source_iterations = count_source_iterations(decoder, args.max_samples)
    source_vis_indices = evenly_spaced_zero_based_indices(total_source_iterations, args.save_vis)
    target_vis_collector = TargetVisualizationCollector(
        vis_dir=out_dir / "visualizations",
        source_indices=source_vis_indices,
        model_label="MedSAM",
    )

    try:
        from segment_anything import sam_model_registry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "segment_anything is required for benchmarking. "
            "Install it in your environment before running this script."
        ) from exc

    checkpoint = ensure_checkpoint(
        checkpoint=args.checkpoint,
        auto_download=not args.no_auto_download_checkpoint,
        repo_id=args.checkpoint_repo_id,
        filename=args.checkpoint_filename,
        revision=args.checkpoint_revision,
    )

    device = resolve_torch_device(args.device)
    model = load_medsam_model(sam_model_registry, checkpoint, device)

    rows = []
    skipped = 0
    metrics_calculator = SegmentationMetrics()
    benchmark_tic = time.perf_counter()
    total_label = str(total_iterations) if total_iterations is not None else "all available"
    print(f"Running MedSAM benchmark for dataset '{args.dataset}' on {total_label} samples...")

    for idx, sample in enumerate(decoder.iter_samples(max_samples=args.max_samples)):
        image_3c = ensure_three_channels(sample.image)
        H, W = image_3c.shape[:2]

        gt_mask = (sample.mask > 0).astype(np.uint8)
        bboxes = bboxes_from_mask(
            gt_mask,
            padding=args.box_padding,
            mode=args.bbox_mode,
        )
        if bboxes is None:
            skipped += 1
            print_progress(
                current=idx + 1,
                total=total_iterations,
                evaluated=len(rows),
                skipped=skipped,
                sample_id=sample.sample_id,
                elapsed_s=time.perf_counter() - benchmark_tic,
            )
            continue
        if (
            args.bbox_jitter_fraction > 0
            and np.random.rand() < args.bbox_jitter_prob
        ):
            bboxes = jitter_bbox(bboxes, args.bbox_jitter_fraction, H, W)
        
        image_1024 = prepare_medsam_image(image_3c)
        image_1024_tensor = (
            torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )
        box_np = bboxes
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        tic = time.perf_counter()
        with torch.no_grad():
            image_embedding = model.image_encoder(image_1024_tensor)
        pred = medsam_inference(model, image_embedding, box_1024, H, W)
        infer_ms = (time.perf_counter() - tic) * 1000.0
        metric_bbox = bboxes[0] if args.bbox_mode == "union" and len(bboxes) == 1 else bboxes

        metrics = metrics_calculator.compute(gt_mask, pred)

        rows.append(
            build_metric_row(
                sample=sample,
                height=H,
                width=W,
                bbox=metric_bbox,
                metrics=metrics,
                infer_ms=infer_ms,
            )
        )

        handled_target_vis = target_vis_collector.add_if_selected(
            sample=sample,
            image=image_3c,
            gt_mask=gt_mask,
            pred_mask=pred,
            bbox=metric_bbox,
            dice=metrics["dice"],
            iou=metrics["iou"],
        )
        if not handled_target_vis and idx in vis_indices:
            save_vis(
                vis_dir=out_dir / "visualizations",
                sample_id=sample.sample_id,
                image=image_3c,
                gt_mask=gt_mask,
                pred_mask=pred,
                bbox=metric_bbox,
            )

        print_progress(
            current=idx + 1,
            total=total_iterations,
            evaluated=len(rows),
            skipped=skipped,
            sample_id=sample.sample_id,
            elapsed_s=time.perf_counter() - benchmark_tic,
        )

    if rows or skipped:
        print()

    target_vis_collector.flush_pending()

    metrics_csv = out_dir / "per_sample_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=METRIC_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        summary = {
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "max_samples": format_max_samples(args.max_samples),
            "bbox_mode": args.bbox_mode,
            "box_padding": args.box_padding,
            "num_evaluated": len(rows),
            "num_skipped_empty_mask": skipped,
            **summarize_metric_rows(rows),
        }
        target_class_metrics = summarize_target_class_metrics(rows)
        if target_class_metrics:
            summary["target_mode"] = "class_instance"
            summary["num_source_samples"] = count_unique_source_samples(rows)
            summary["target_class_metrics"] = target_class_metrics
    else:
        summary = {
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "max_samples": format_max_samples(args.max_samples),
            "bbox_mode": args.bbox_mode,
            "box_padding": args.box_padding,
            "num_evaluated": 0,
            "num_skipped_empty_mask": skipped,
            "error": "No valid samples were evaluated.",
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Benchmark finished")
    print(json.dumps(summary, indent=2))
    print(f"Saved per-sample metrics to: {metrics_csv}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
