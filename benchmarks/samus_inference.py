from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import re
import time
from pathlib import Path
import sys
import urllib.parse
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform

ROOT_DIR = Path(__file__).resolve().parents[1]
SAMUS_DIR = ROOT_DIR / "work_dir" / "SAMUS"

for path in (ROOT_DIR, SAMUS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets.common import (
    bbox_from_mask,
    dice_score,
    ensure_three_channels,
    iou_score,
    normalize_to_uint8,
)
from datasets.loader import add_dataset_args, build_decoder_from_args
from models.segment_anything_samus.build_sam_us import samus_model_registry


DEFAULT_SAMUS_CHECKPOINT_FILE_ID = "1nQjMAvbPeolNpCxQyU_HTiOiB5704pkH"


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(
    current: int,
    total: int,
    evaluated: int,
    skipped: int,
    sample_id: str,
    elapsed_s: float,
    recent_batch_size: int | None = None,
    recent_batch_s: float | None = None,
) -> None:
    total_str = str(total) if total > 0 else "?"
    sample_label = sample_id if len(sample_id) <= 48 else f"...{sample_id[-45:]}"
    avg_rate = evaluated / elapsed_s if elapsed_s > 0 and evaluated > 0 else 0.0
    parts = [
        f"Progress {current}/{total_str}",
        f"evaluated={evaluated}",
        f"skipped={skipped}",
        f"avg={avg_rate:.2f} samples/s",
        f"elapsed={format_duration(elapsed_s)}",
    ]
    if recent_batch_size is not None and recent_batch_s is not None and recent_batch_s > 0:
        batch_rate = recent_batch_size / recent_batch_s
        parts.append(f"batch={recent_batch_size} @ {batch_rate:.2f} samples/s")
    parts.append(f"last={sample_label}")
    print(
        "\r" + " | ".join(parts),
        end="",
        flush=True,
    )


def get_total_iterations(decoder, max_samples: int | None) -> int | None:
    count_fn = getattr(decoder, "count_samples", None)
    if callable(count_fn):
        return count_fn(max_samples=max_samples)
    return max_samples


def iter_samples_with_workers(decoder, max_samples: int | None, num_workers: int):
    if num_workers <= 0:
        yield from decoder.iter_samples(max_samples=max_samples)
        return

    info_iter_fn = getattr(decoder, "iter_sample_infos", None)
    load_fn = getattr(decoder, "load_sample_from_info", None)
    if not callable(info_iter_fn) or not callable(load_fn):
        yield from decoder.iter_samples(max_samples=max_samples)
        return

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(load_fn, info_iter_fn(max_samples=max_samples))


def prepare_samus_tensor(image: np.ndarray, size: int) -> torch.Tensor:
    image_uint8 = normalize_to_uint8(image)
    image_3c = ensure_three_channels(image_uint8)
    image_rs = transform.resize(
        image_3c,
        (size, size),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    image_rs /= 255.0
    return torch.from_numpy(image_rs).permute(2, 0, 1).contiguous()


def scale_bbox_to_model_space(
    bbox: np.ndarray,
    src_height: int,
    src_width: int,
    dst_size: int,
) -> np.ndarray:
    scale = np.array(
        [
            dst_size / max(src_width, 1),
            dst_size / max(src_height, 1),
            dst_size / max(src_width, 1),
            dst_size / max(src_height, 1),
        ],
        dtype=np.float32,
    )
    scaled = bbox.astype(np.float32) * scale
    return np.clip(scaled, 0, dst_size - 1)


def load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _http_get(url: str) -> tuple[bytes, dict[str, str], str]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
        return body, headers, response.geturl()


def download_samus_checkpoint(destination: Path, file_id: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    warning_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    body, headers, _ = _http_get(warning_url)

    content_type = headers.get("content-type", "")
    disposition = headers.get("content-disposition", "")
    if "application/octet-stream" in content_type or "attachment" in disposition.lower():
        download_url = warning_url
    else:
        html = body.decode("utf-8", errors="replace")
        confirm_match = re.search(r'name="confirm" value="([^"]+)"', html)
        uuid_match = re.search(r'name="uuid" value="([^"]+)"', html)
        if not confirm_match:
            raise RuntimeError("Could not extract Google Drive confirmation token for SAMUS checkpoint.")

        query = {
            "id": file_id,
            "export": "download",
            "confirm": confirm_match.group(1),
        }
        if uuid_match:
            query["uuid"] = uuid_match.group(1)
        download_url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(query)

    request = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    return destination


def ensure_samus_checkpoint(path: Path, auto_download: bool, file_id: str) -> Path:
    if path.exists():
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"SAMUS checkpoint not found: {path}. "
            "Pass --checkpoint to a valid file or omit --no-auto-download-checkpoint."
        )

    print(f"SAMUS checkpoint not found at {path}. Downloading from upstream SAMUS release...")
    return download_samus_checkpoint(path, file_id)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    base_checkpoint = resolve_repo_path(args.sam_ckpt)
    base_checkpoint_arg = str(base_checkpoint) if base_checkpoint.exists() else None

    model = samus_model_registry["vit_b"](args=args, checkpoint=base_checkpoint_arg)

    trained_checkpoint = ensure_samus_checkpoint(
        path=resolve_repo_path(args.checkpoint),
        auto_download=not args.no_auto_download_checkpoint,
        file_id=args.checkpoint_file_id,
    )

    state_dict = load_checkpoint_state_dict(trained_checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def samus_box_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    boxes_256: np.ndarray,
    original_sizes: list[tuple[int, int]],
) -> list[np.ndarray]:
    input_size = image_tensor.shape[-2:]
    image_tensor = image_tensor.to(model.device)
    processed = torch.stack([model.preprocess(img) for img in image_tensor], dim=0)
    image_embeddings = model.image_encoder(processed)

    box_torch = torch.as_tensor(boxes_256, dtype=torch.float32, device=model.device)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    predictions = []
    for idx, original_size in enumerate(original_sizes):
        masks = model.postprocess_masks(
            low_res_masks[idx : idx + 1],
            input_size=input_size,
            original_size=original_size,
        )
        probs = torch.sigmoid(masks)[0, 0]
        predictions.append((probs.detach().cpu().numpy() > 0.5).astype(np.uint8))
    return predictions


def save_vis(
    vis_dir: Path,
    sample_id: str,
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    bbox: np.ndarray,
) -> None:
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
    ax[2].set_title("SAMUS Prediction")

    for axis in ax:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(vis_dir / f"{name}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SAMUS inference with GT box prompts")
    add_dataset_args(parser, include_camus=False)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dir/SAMUS/ckp/SAMUS.pth",
        help="Path to trained SAMUS weights.",
    )
    parser.add_argument(
        "--no-auto-download-checkpoint",
        action="store_true",
        help="Require an existing local SAMUS checkpoint instead of downloading it.",
    )
    parser.add_argument(
        "--checkpoint-file-id",
        type=str,
        default=DEFAULT_SAMUS_CHECKPOINT_FILE_ID,
        help="Google Drive file id for the upstream SAMUS checkpoint.",
    )
    parser.add_argument(
        "--sam-ckpt",
        type=str,
        default="work_dir/SAMUS/checkpoints/sam_vit_b_01ec64.pth",
        help="Optional base SAM ViT-B checkpoint loaded before the trained SAMUS weights if present.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of background worker threads for sample loading/preprocessing.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples to evaluate. Defaults to the full dataset.",
    )
    parser.add_argument("--box-padding", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="results/samus_test")
    parser.add_argument(
        "--encoder-input-size",
        dest="encoder_input_size",
        type=int,
        default=256,
        help="Input image size used by the SAMUS encoder.",
    )
    parser.add_argument(
        "--low-image-size",
        dest="low_image_size",
        type=int,
        default=128,
        help="Low-resolution mask size expected by SAMUS configs.",
    )
    parser.add_argument("--vit-name", type=str, default="vit_b")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder = build_decoder_from_args(args)
    model = build_model(args)
    total_iterations = get_total_iterations(decoder, args.max_samples)

    rows = []
    skipped = 0
    benchmark_tic = time.perf_counter()
    total_label = str(total_iterations) if total_iterations is not None else "all available"
    print(
        f"Running SAMUS benchmark for dataset '{args.dataset}' on "
        f"{total_label} samples with batch_size={args.batch_size} "
        f"and num_workers={args.num_workers}..."
    )

    pending = []
    sample_iter = iter_samples_with_workers(
        decoder=decoder,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    for idx, sample in enumerate(sample_iter, start=1):
        image_3c = ensure_three_channels(sample.image)
        height, width = image_3c.shape[:2]

        gt_mask = (sample.mask > 0).astype(np.uint8)
        bbox = bbox_from_mask(gt_mask, padding=args.box_padding)
        if bbox is None:
            skipped += 1
            print_progress(
                idx,
                total_iterations or 0,
                len(rows),
                skipped,
                sample.sample_id,
                elapsed_s=time.perf_counter() - benchmark_tic,
            )
            continue

        image_tensor = prepare_samus_tensor(image_3c, size=args.encoder_input_size)
        scaled_bbox = scale_bbox_to_model_space(
            bbox=bbox,
            src_height=height,
            src_width=width,
            dst_size=args.encoder_input_size,
        )
        pending.append(
            {
                "idx": idx,
                "sample": sample,
                "image_uint8": normalize_to_uint8(image_3c),
                "image_tensor": image_tensor,
                "height": height,
                "width": width,
                "gt_mask": gt_mask,
                "bbox": bbox,
                "scaled_bbox": scaled_bbox,
            }
        )

        if len(pending) < args.batch_size:
            continue

        batch_image_tensor = torch.stack([entry["image_tensor"] for entry in pending], dim=0)
        batch_boxes = np.stack([entry["scaled_bbox"] for entry in pending], axis=0)
        batch_original_sizes = [(entry["height"], entry["width"]) for entry in pending]

        tic = time.perf_counter()
        batch_preds = samus_box_inference(
            model=model,
            image_tensor=batch_image_tensor,
            boxes_256=batch_boxes,
            original_sizes=batch_original_sizes,
        )
        batch_elapsed_s = time.perf_counter() - tic
        batch_infer_ms = batch_elapsed_s * 1000.0 / len(pending)

        for entry, pred in zip(pending, batch_preds):
            dice = float(dice_score(entry["gt_mask"], pred))
            iou = float(iou_score(entry["gt_mask"], pred))

            rows.append(
                {
                    "sample_id": entry["sample"].sample_id,
                    "height": entry["height"],
                    "width": entry["width"],
                    "bbox": entry["bbox"].tolist(),
                    "dice": dice,
                    "iou": iou,
                    "infer_ms": batch_infer_ms,
                }
            )

            if entry["idx"] <= args.save_vis:
                save_vis(
                    vis_dir=out_dir / "visualizations",
                    sample_id=entry["sample"].sample_id,
                    image=entry["image_uint8"],
                    gt_mask=entry["gt_mask"],
                    pred_mask=pred,
                    bbox=entry["bbox"],
                )
        last_entry = pending[-1]
        print_progress(
            last_entry["idx"],
            total_iterations or 0,
            len(rows),
            skipped,
            last_entry["sample"].sample_id,
            elapsed_s=time.perf_counter() - benchmark_tic,
            recent_batch_size=len(pending),
            recent_batch_s=batch_elapsed_s,
        )

        pending = []

    if pending:
        batch_image_tensor = torch.stack([entry["image_tensor"] for entry in pending], dim=0)
        batch_boxes = np.stack([entry["scaled_bbox"] for entry in pending], axis=0)
        batch_original_sizes = [(entry["height"], entry["width"]) for entry in pending]

        tic = time.perf_counter()
        batch_preds = samus_box_inference(
            model=model,
            image_tensor=batch_image_tensor,
            boxes_256=batch_boxes,
            original_sizes=batch_original_sizes,
        )
        batch_elapsed_s = time.perf_counter() - tic
        batch_infer_ms = batch_elapsed_s * 1000.0 / len(pending)

        for entry, pred in zip(pending, batch_preds):
            dice = float(dice_score(entry["gt_mask"], pred))
            iou = float(iou_score(entry["gt_mask"], pred))

            rows.append(
                {
                    "sample_id": entry["sample"].sample_id,
                    "height": entry["height"],
                    "width": entry["width"],
                    "bbox": entry["bbox"].tolist(),
                    "dice": dice,
                    "iou": iou,
                    "infer_ms": batch_infer_ms,
                }
            )

            if entry["idx"] <= args.save_vis:
                save_vis(
                    vis_dir=out_dir / "visualizations",
                    sample_id=entry["sample"].sample_id,
                    image=entry["image_uint8"],
                    gt_mask=entry["gt_mask"],
                    pred_mask=pred,
                    bbox=entry["bbox"],
                )
        last_entry = pending[-1]
        print_progress(
            last_entry["idx"],
            total_iterations or 0,
            len(rows),
            skipped,
            last_entry["sample"].sample_id,
            elapsed_s=time.perf_counter() - benchmark_tic,
            recent_batch_size=len(pending),
            recent_batch_s=batch_elapsed_s,
        )

    if rows or skipped:
        print()

    metrics_csv = out_dir / "per_sample_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "height", "width", "bbox", "dice", "iou", "infer_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        dice_vals = np.array([row["dice"] for row in rows], dtype=np.float32)
        iou_vals = np.array([row["iou"] for row in rows], dtype=np.float32)
        ms_vals = np.array([row["infer_ms"] for row in rows], dtype=np.float32)
        summary = {
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "checkpoint": args.checkpoint,
            "sam_ckpt": args.sam_ckpt,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
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
            "checkpoint": args.checkpoint,
            "sam_ckpt": args.sam_ckpt,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
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
