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
    bbox_from_mask,
    dice_score,
    ensure_three_channels,
    iou_score,
    prepare_medsam_image,
)
from datasets.loader import add_dataset_args, build_decoder_from_args


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
    return (low_res_pred > 0.5).astype(np.uint8)


def jitter_bbox(
    bbox: np.ndarray, jitter_frac: float, height: int, width: int
) -> np.ndarray:
    if jitter_frac <= 0:
        return bbox
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    dx = np.random.uniform(-jitter_frac * w, jitter_frac * w)
    dy = np.random.uniform(-jitter_frac * h, jitter_frac * h)
    new_x0 = np.clip(bbox[0] + dx, 0, width - 1)
    new_y0 = np.clip(bbox[1] + dy, 0, height - 1)
    new_x1 = np.clip(bbox[2] + dx, 0, width - 1)
    new_y1 = np.clip(bbox[3] + dy, 0, height - 1)
    return np.array([new_x0, new_y0, new_x1, new_y1], dtype=np.int32)


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
    ax[1].add_patch(
        plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
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
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--box-padding", type=int, default=0)
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
    benchmark_tic = time.perf_counter()
    total_label = str(total_iterations) if total_iterations is not None else "all available"
    print(f"Running MedSAM benchmark for dataset '{args.dataset}' on {total_label} samples...")

    for idx, sample in enumerate(decoder.iter_samples(max_samples=args.max_samples)):
        image_3c = ensure_three_channels(sample.image)
        H, W = image_3c.shape[:2]

        gt_mask = (sample.mask > 0).astype(np.uint8)
        bbox = bbox_from_mask(gt_mask, padding=args.box_padding)
        if bbox is None:
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
            bbox = jitter_bbox(bbox, args.bbox_jitter_fraction, H, W)
        
        image_1024 = prepare_medsam_image(image_3c)
        image_1024_tensor = (
            torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )
        box_np = bbox[None, :]
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        tic = time.perf_counter()
        with torch.no_grad():
            image_embedding = model.image_encoder(image_1024_tensor)
        pred = medsam_inference(model, image_embedding, box_1024, H, W)
        infer_ms = (time.perf_counter() - tic) * 1000.0

        dice = float(dice_score(gt_mask, pred))
        iou = float(iou_score(gt_mask, pred))

        rows.append(
            {
                "sample_id": sample.sample_id,
                "height": H,
                "width": W,
                "bbox": bbox.tolist(),
                "dice": dice,
                "iou": iou,
                "infer_ms": infer_ms,
            }
        )

        if idx in vis_indices:
            save_vis(
                vis_dir=out_dir / "visualizations",
                sample_id=sample.sample_id,
                image=image_3c,
                gt_mask=gt_mask,
                pred_mask=pred,
                bbox=bbox,
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

    metrics_csv = out_dir / "per_sample_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "height", "width", "bbox", "dice", "iou", "infer_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        dice_vals = np.array([r["dice"] for r in rows], dtype=np.float32)
        iou_vals = np.array([r["iou"] for r in rows], dtype=np.float32)
        ms_vals = np.array([r["infer_ms"] for r in rows], dtype=np.float32)
        summary = {
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "num_evaluated": len(rows),
            "num_skipped_empty_mask": skipped,
            "dice_mean": float(dice_vals.mean()),
            "dice_std": float(dice_vals.std()),
            "iou_mean": float(iou_vals.mean()),
            "iou_std": float(iou_vals.std()),
            "infer_ms_mean": float(ms_vals.mean()),
            "infer_ms_std": float(ms_vals.std()),
        }
    else:
        summary = {
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
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
