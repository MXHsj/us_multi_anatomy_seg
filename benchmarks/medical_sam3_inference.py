from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.common import ensure_three_channels, normalize_to_uint8
from datasets.label_text import LABEL_TEXT, concept_for
from datasets.loader import add_dataset_args, build_decoder_from_args
from benchmarks.eval_utils import (
    METRIC_FIELDNAMES,
    build_metric_row,
    count_unique_source_samples,
    format_max_samples,
    parse_max_samples,
    summarize_metric_rows,
    summarize_target_class_metrics,
)
from benchmarks.metrics import SegmentationMetrics


# Medical SAM3's `sam3` package must be installed in the environment (mirroring how the
# MedSAM engine imports the installed `segment_anything` package). This engine only
# imports from it -- it does not reference any local source clone. The fine-tuned 2D
# checkpoint is auto-downloaded from Hugging Face.
DEFAULT_MEDICAL_SAM3_CHECKPOINT = "work_dir/MedicalSAM3/checkpoint_2D.pt"
DEFAULT_MEDICAL_SAM3_CHECKPOINT_REPO_ID = "Chongcong/Medical-SAM3"
DEFAULT_MEDICAL_SAM3_CHECKPOINT_FILENAME = "checkpoint_2D.pt"
# CLIP BPE vocab vendored into this repo so the engine is self-contained. The sam3
# package's bundled copy lives under the clone's assets/ (outside the package), so a
# non-editable install does not ship it; we fall back to this vendored file.
DEFAULT_BPE_VOCAB = ROOT_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"

# Per-sample CSV gains two text-specific columns on top of the shared schema.
# The "bbox" column from METRIC_FIELDNAMES is inert here (text has no box) and is
# written empty so the file stays schema-compatible with the box engines.
TEXT_METRIC_FIELDNAMES = [*METRIC_FIELDNAMES, "concept", "text_score"]


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


def resolve_device_str(requested_device: str) -> str:
    """Medical SAM3's build_sam3_image_model only understands 'cuda' / 'cpu'."""
    requested = str(requested_device).lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        print(
            f"Requested device '{requested_device}' but CUDA is unavailable; using 'cpu'.",
            flush=True,
        )
        return "cpu"
    return "cpu"


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
            f"Medical SAM3 checkpoint not found: {checkpoint}. "
            f"Download '{filename}' from https://huggingface.co/{repo_id} and place it "
            "there, or pass --checkpoint."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required to auto-download the Medical SAM3 checkpoint."
        ) from exc

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    )
    shutil.copy2(downloaded, checkpoint)
    return checkpoint


def load_medical_sam3_processor(
    checkpoint: Path,
    device: str,
    confidence_threshold: float,
    bpe_path: str | None = None,
):
    """Build the SAM3 image model from the installed `sam3` package and wrap it."""
    try:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Medical SAM3 'sam3' package is required for this benchmark and is not "
            "installed. Install it from https://github.com/AIM-Research-Lab/Medical-SAM3 "
            "into this environment (e.g. `pip install -r requirements.txt && "
            "pip install -e \".[train]\"`), like `segment_anything` is for the MedSAM engine."
        ) from exc

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # The fine-tuned 2D checkpoint carries the full model (vision + text + decoder),
    # so we never fetch the base facebook/sam3 weights (load_from_HF=False).
    # bpe_path=None lets the installed package use its bundled BPE vocabulary.
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
    )
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


def _run_label(args: argparse.Namespace, dataset: str) -> str:
    """Resolve the active label for selectable single-target datasets from CLI args.

    Text prompting needs exactly one concept per run for these datasets, so a merged
    multi-label selection is rejected rather than silently collapsed.
    """
    if dataset == "roblus":
        labels = [v.strip() for v in args.roblus_labels.split(",") if v.strip()]
        flag = "--roblus-labels"
    elif dataset == "oku":
        labels = [v.strip() for v in args.oku_anatomy.split(",") if v.strip()]
        flag = "--oku-anatomy"
    else:
        raise SystemExit(
            f"Dataset '{dataset}' has per-class concepts but no per-sample label "
            "metadata and no known CLI label flag."
        )
    if len(labels) != 1:
        raise SystemExit(
            f"Text path needs exactly one concept for '{dataset}'; {flag} selected "
            f"{labels}. Run one class per evaluation (class-specific reporting)."
        )
    return labels[0]


def resolve_concept(args: argparse.Namespace, sample) -> str:
    """Map a decoded sample to its canonical text concept (never from the mask)."""
    dataset = args.dataset.lower()
    entry = LABEL_TEXT.get(dataset)
    if isinstance(entry, str):
        return concept_for(dataset)
    if not isinstance(entry, dict):
        raise KeyError(f"No label->text concept registered for dataset '{dataset}'.")

    metadata = getattr(sample, "metadata", None) or {}
    # CAMUS varies per sample: one binary target per class id (e.g. 1=LV, 3=LA).
    class_id = metadata.get("target_class_id")
    if class_id in entry:
        return concept_for(dataset, class_id)
    # AULID records its (run-level) label per sample.
    label = metadata.get("label")
    if label in entry:
        return concept_for(dataset, label)
    # Remaining selectable datasets (roblus, oku) take their label from CLI args.
    return concept_for(dataset, _run_label(args, dataset))


@torch.no_grad()
def predict_text(processor, state: dict, concept: str) -> tuple[np.ndarray | None, float]:
    """Run text-prompted inference and return the highest-confidence mask + score."""
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(prompt=concept, state=state)

    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None or scores is None or masks.shape[0] == 0:
        return None, 0.0

    best_idx = int(torch.argmax(scores).item())
    # masks are [N, 1, H, W] bool, already interpolated to the original image size.
    mask = masks[best_idx].squeeze().detach().cpu().numpy().astype(np.uint8)
    score = float(scores[best_idx].item())
    return mask, score


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    mask = np.squeeze(mask)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    resized = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    return (np.array(resized) > 127).astype(np.uint8)


def save_text_vis(
    vis_dir: Path,
    sample_id: str,
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    concept: str,
    score: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir.mkdir(parents=True, exist_ok=True)
    name = sample_id.replace("/", "_")

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Image")

    ax[1].imshow(image)
    ax[1].imshow(gt_mask, alpha=0.45, cmap="Greens")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(image)
    ax[2].imshow(pred_mask, alpha=0.45, cmap="Reds")
    ax[2].set_title(f"Medical SAM3 text='{concept}' (s={score:.2f})")

    for axis in ax:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(vis_dir / f"{name}.png", dpi=140)
    plt.close(fig)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text-prompted Medical SAM3 benchmark (concept from label taxonomy)."
    )
    add_dataset_args(parser, include_camus=True)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_MEDICAL_SAM3_CHECKPOINT)
    parser.add_argument(
        "--bpe-path",
        type=str,
        default="",
        help="BPE vocab path. Defaults to the vendored assets/bpe_simple_vocab_16e6.txt.gz "
        "(falls back to the sam3 package's bundled vocab if absent).",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="",
        help="Override the per-class concept from datasets/label_text.py with this single "
        "prompt for ALL samples (e.g. 'object'). Use a distinct --output-dir to keep the run separate.",
    )
    parser.add_argument(
        "--no-auto-download-checkpoint",
        action="store_true",
        help="Require an existing local checkpoint instead of downloading it.",
    )
    parser.add_argument(
        "--checkpoint-repo-id",
        type=str,
        default=DEFAULT_MEDICAL_SAM3_CHECKPOINT_REPO_ID,
        help="Hugging Face model repo used when the checkpoint is missing.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default=DEFAULT_MEDICAL_SAM3_CHECKPOINT_FILENAME,
        help="Checkpoint filename inside --checkpoint-repo-id.",
    )
    parser.add_argument("--checkpoint-revision", type=str, default="main")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum detection confidence (Medical SAM3 default is 0.1).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max-samples",
        type=parse_max_samples,
        default=None,
        metavar="N|all",
        help="Evaluate a positive integer cap or 'all' for the full dataset.",
    )
    parser.add_argument("--save-vis", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="results/medicalsam3_text_prompt")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder = build_decoder_from_args(args)
    total_iterations = get_total_iterations(decoder, args.max_samples)
    vis_indices = evenly_spaced_indices(total_iterations, args.save_vis)

    checkpoint = ensure_checkpoint(
        checkpoint=args.checkpoint,
        auto_download=not args.no_auto_download_checkpoint,
        repo_id=args.checkpoint_repo_id,
        filename=args.checkpoint_filename,
        revision=args.checkpoint_revision,
    )
    device = resolve_device_str(args.device)
    if args.bpe_path:
        bpe_path = args.bpe_path
    elif DEFAULT_BPE_VOCAB.exists():
        bpe_path = str(DEFAULT_BPE_VOCAB)
    else:
        bpe_path = None  # let the sam3 package use its own bundled vocab
    processor = load_medical_sam3_processor(
        checkpoint=checkpoint,
        device=device,
        confidence_threshold=args.confidence_threshold,
        bpe_path=bpe_path,
    )

    from PIL import Image

    rows: list[dict] = []
    skipped = 0
    metrics_calculator = SegmentationMetrics()
    benchmark_tic = time.perf_counter()
    total_label = str(total_iterations) if total_iterations is not None else "all available"
    print(
        f"Running Medical SAM3 text benchmark for dataset '{args.dataset}' "
        f"on {total_label} samples (device={device})..."
    )

    # Encode each source image once and reuse the state across its per-class targets
    # (e.g. CAMUS yields LV and LA for the same frame). The ViT encode dominates cost.
    cached_key: str | None = None
    cached_state: dict | None = None
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else None
    )

    for idx, sample in enumerate(decoder.iter_samples(max_samples=args.max_samples)):
        image_3c = ensure_three_channels(normalize_to_uint8(sample.image))
        H, W = image_3c.shape[:2]
        gt_mask = (sample.mask > 0).astype(np.uint8)

        # Mirror the box engines: skip empty-GT frames (there a box could not be built).
        if not gt_mask.any():
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

        concept = args.text_prompt or resolve_concept(args, sample)
        metadata = getattr(sample, "metadata", None) or {}
        source_key = str(metadata.get("source_sample_id") or sample.sample_id)

        tic = time.perf_counter()
        if autocast is not None:
            autocast.__enter__()
        try:
            if source_key != cached_key:
                cached_state = processor.set_image(Image.fromarray(image_3c))
                cached_key = source_key
            pred, score = predict_text(processor, cached_state, concept)
        finally:
            if autocast is not None:
                autocast.__exit__(None, None, None)
        infer_ms = (time.perf_counter() - tic) * 1000.0

        if pred is None:
            pred = np.zeros((H, W), dtype=np.uint8)
        elif pred.shape != (H, W):
            pred = resize_mask(pred, (H, W))

        metrics = metrics_calculator.compute(gt_mask, pred)
        row = build_metric_row(
            sample=sample,
            height=H,
            width=W,
            bbox=np.array([], dtype=np.int32),  # inert for the text path
            metrics=metrics,
            infer_ms=infer_ms,
        )
        row["concept"] = concept
        row["text_score"] = score
        rows.append(row)

        if idx in vis_indices:
            save_text_vis(
                vis_dir=out_dir / "visualizations",
                sample_id=sample.sample_id,
                image=image_3c,
                gt_mask=gt_mask,
                pred_mask=pred,
                concept=concept,
                score=score,
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
        writer = csv.DictWriter(f, fieldnames=TEXT_METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        summary = {
            "model": "medicalsam3",
            "protocol": "text",
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "checkpoint": str(checkpoint),
            "device": device,
            "confidence_threshold": args.confidence_threshold,
            "text_prompt_override": args.text_prompt or None,
            "max_samples": format_max_samples(args.max_samples),
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
            "model": "medicalsam3",
            "protocol": "text",
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "max_samples": format_max_samples(args.max_samples),
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
