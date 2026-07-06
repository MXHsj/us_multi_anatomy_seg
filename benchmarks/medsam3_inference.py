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
import os
from dotenv import load_dotenv
import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.common import bbox_from_mask, ensure_three_channels, normalize_to_uint8
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

SRC_DIR = ROOT_DIR / "work_dir" / "MedSAM3" / "source_code"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

DEFAULT_MEDSAM3_CHECKPOINT = ROOT_DIR / "work_dir/MedSAM3/best_lora_weights.pt"
DEFAULT_MEDSAM3_CHECKPOINT_REPO_ID = "lal-Joey/MedSAM3_v1"
DEFAULT_MEDSAM3_CHECKPOINT_FILENAME = "best_lora_weights.pt"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "full_lora_config.yaml"
DEFAULT_RESOLUTION = 1008

TEXT_METRIC_FIELDNAMES = [*METRIC_FIELDNAMES, "concept", "text_score"]

ENV_FILE_PATH = ROOT_DIR / ".env"
assert os.path.exists(ENV_FILE_PATH), f".env file not found at path {ENV_FILE_PATH}"
load_dotenv(ENV_FILE_PATH)
assert os.environ.get("HF_TOKEN"), f"HF_TOKEN must be set in .env file before using medsam3"


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
    """Use either'cuda' / 'cpu'. MPS is not supported."""
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
            f"MedSAM3 checkpoint not found: {checkpoint}. "
            f"Download '{filename}' from https://huggingface.co/{repo_id} and place it "
            "there, or pass --checkpoint."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required to auto-download the MedSAM3 checkpoint."
        ) from exc

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    )
    shutil.copy2(downloaded, checkpoint)
    return checkpoint


class MedSAM3TextInference:
    """MedSAM3 text-prompted inference wrapper for benchmarking."""

    def __init__(
        self,
        checkpoint: Path,
        config_path: Path,
        device: str,
        resolution: int = DEFAULT_RESOLUTION,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
    ):
        from sam3.model_builder import build_sam3_image_model
        from sam3.train.data.sam3_image_dataset import (
            Datapoint,
            Image as SAMImage,
            FindQueryLoaded,
            InferenceMetadata,
        )
        from sam3.train.data.collator import collate_fn_api
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI,
            RandomResizeAPI,
            ToTensorAPI,
            NormalizeAPI,
        )

        # Store SAM3 classes for later use
        self.Datapoint = Datapoint
        self.SAMImage = SAMImage
        self.FindQueryLoaded = FindQueryLoaded
        self.InferenceMetadata = InferenceMetadata
        self.collate_fn_api = collate_fn_api
        self.copy_data_to_device = copy_data_to_device

        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print(f"🔧 Initializing MedSAM3 + LoRA...")
        print(f"   Device: {self.device}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {confidence_threshold}")

        # Build base model
        print("📦 Building SAM3 model...")
        model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path=os.path.join(str(SRC_DIR), "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
            eval_mode=True,
        )

        # Apply LoRA configuration
        print("🔗 Applying LoRA configuration...")
        lora_cfg = config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        model = apply_lora_to_model(model, lora_config)

        # Load LoRA weights
        print(f"💾 Loading LoRA weights from {checkpoint}...")
        load_lora_weights(model, str(checkpoint))
        model.to(self.device)
        model.eval()
        self.model = model

        # Setup transforms
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        print("✅ MedSAM3 + LoRA ready for inference!\n")

    def create_datapoint(
        self,
        pil_image,
        text_prompt: str,
        bbox_xyxy: np.ndarray | None = None,
    ):
        """Create a SAM3 datapoint from a PIL image, a text prompt and an optional box.

        ``bbox_xyxy`` is given in denormalized XYXY pixel coordinates of the original
        image. The SAM3 transforms (RandomResizeAPI + NormalizeAPI) scale it to the model
        resolution and convert it to normalized CxCyWH, mirroring the training pipeline.
        """
        w, h = pil_image.size

        sam_image = self.SAMImage(data=pil_image, objects=[], size=[h, w])

        input_bbox = None
        input_bbox_label = None
        if bbox_xyxy is not None:
            input_bbox = torch.as_tensor(
                np.asarray(bbox_xyxy, dtype=np.float32)
            ).view(1, 4)
            # The collator requires a per-box label; 1 marks a positive (inclusion) box.
            input_bbox_label = torch.ones(1, dtype=torch.long)

        query = self.FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=input_bbox,
            input_bbox_label=input_bbox_label,
            inference_metadata=self.InferenceMetadata(
                coco_image_id=0,
                original_image_id=0,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )

        return self.Datapoint(find_queries=[query], images=[sam_image])

    @torch.no_grad()
    def predict(
        self,
        pil_image,
        text_prompt: str,
        bbox_xyxy: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, float]:
        """Run prompted inference and return the highest-confidence mask + score.

        When ``bbox_xyxy`` is provided, the box is supplied to the detector alongside the
        text concept (the paper's MedSAM-3 "T+I" protocol); otherwise this is pure text.
        """
        from torchvision.ops import nms
        import torch.nn.functional as F

        # Create datapoint
        datapoint = self.create_datapoint(pil_image, text_prompt, bbox_xyxy=bbox_xyxy)

        # Apply transforms
        datapoint = self.transform(datapoint)

        # Collate into batch
        batch = self.collate_fn_api([datapoint], dict_key="input")["input"]

        # Move to device
        batch = self.copy_data_to_device(batch, self.device, non_blocking=True)

        # Forward pass
        outputs = self.model(batch)

        # Post-process outputs
        last_output = outputs[-1]
        pred_logits = last_output["pred_logits"]  # [batch, num_queries, num_classes]
        pred_boxes = last_output["pred_boxes"]  # [batch, num_queries, 4]
        pred_masks = last_output.get("pred_masks", None)  # [batch, num_queries, H, W]

        # Get probabilities
        out_probs = pred_logits.sigmoid()

        # Get scores
        scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

        # Filter by threshold
        keep = scores > self.confidence_threshold
        num_keep = keep.sum().item()

        if num_keep == 0:
            return None, 0.0

        # Get boxes and convert from cxcywh to xyxy
        boxes_cxcywh = pred_boxes[0, keep]
        kept_scores = scores[keep]
        cx, cy, w_box, h_box = boxes_cxcywh.unbind(-1)

        # Convert to xyxy and scale to original image size
        orig_w, orig_h = pil_image.size
        x1 = (cx - w_box / 2) * orig_w
        y1 = (cy - h_box / 2) * orig_h
        x2 = (cx + w_box / 2) * orig_w
        y2 = (cy + h_box / 2) * orig_h

        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # Apply NMS
        keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
        if len(keep_nms) == 0:
            return None, 0.0

        kept_scores = kept_scores[keep_nms]

        # Get best mask
        if pred_masks is not None:
            best_idx = keep_nms[torch.argmax(kept_scores)]
            # Get mask indices that survived thresholding
            keep_indices = torch.where(keep)[0]
            mask_idx = keep_indices[best_idx]

            mask_small = pred_masks[0, mask_idx].sigmoid() > 0.5

            # Resize to original size
            mask_resized = (
                F.interpolate(
                    mask_small.unsqueeze(0).unsqueeze(0).float(),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                > 0.5
            )

            mask_np = mask_resized.cpu().numpy().astype(np.uint8)
            score = float(kept_scores.max().item())
            return mask_np, score

        return None, 0.0


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




def apply_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    scale_range: tuple[float, float] | None = None,
    shift_range: tuple[float, float] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply geometric augmentations to image and mask consistently.
    
    Args:
        image: Input image (H, W, C)
        mask: Ground truth mask (H, W)
        scale_range: (min_scale, max_scale) for random scaling, e.g., (0.8, 1.2)
        shift_range: (max_horizontal_shift, max_vertical_shift) as fraction of image size, e.g., (0.1, 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        Augmented image and mask
    """
    if seed is not None:
        np.random.seed(seed)
    
    H, W = image.shape[:2]
    aug_image = image.copy()
    aug_mask = mask.copy()
    
    # Apply scaling
    if scale_range is not None:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        new_h, new_w = int(H * scale), int(W * scale)
        aug_image = cv2.resize(aug_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        aug_mask = cv2.resize(aug_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Crop or pad to original size
        if scale > 1.0:
            # Crop center
            start_h = (new_h - H) // 2
            start_w = (new_w - W) // 2
            aug_image = aug_image[start_h:start_h + H, start_w:start_w + W]
            aug_mask = aug_mask[start_h:start_h + H, start_w:start_w + W]
        else:
            # Pad to original size
            pad_h = (H - new_h) // 2
            pad_w = (W - new_w) // 2
            aug_image = cv2.copyMakeBorder(
                aug_image,
                pad_h, H - new_h - pad_h,
                pad_w, W - new_w - pad_w,
                cv2.BORDER_CONSTANT,
                value=0
            )
            aug_mask = cv2.copyMakeBorder(
                aug_mask,
                pad_h, H - new_h - pad_h,
                pad_w, W - new_w - pad_w,
                cv2.BORDER_CONSTANT,
                value=0
            )
    
    # Apply translation (shift)
    if shift_range is not None:
        shift_h = int(np.random.uniform(-shift_range[1], shift_range[1]) * H)
        shift_w = int(np.random.uniform(-shift_range[0], shift_range[0]) * W)
        
        M = np.float32([[1, 0, shift_w], [0, 1, shift_h]])
        aug_image = cv2.warpAffine(aug_image, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aug_mask = cv2.warpAffine(aug_mask, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return aug_image, aug_mask


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
    bbox: np.ndarray | None = None,
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
    if bbox is not None and len(bbox) == 4:
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
    else:
        ax[1].set_title("Ground Truth")

    ax[2].imshow(image)
    ax[2].imshow(pred_mask, alpha=0.45, cmap="Reds")
    title = f"MedSAM3 text='{concept}'"
    if bbox is not None and len(bbox) == 4:
        title += "+box"
    ax[2].set_title(f"{title} (s={score:.2f})")

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
        description="MedSAM3 benchmark (concept from label taxonomy). Pure text by default; "
        "pass --use-gt-bbox to add a ground-truth box prompt (the paper's 'T+I' protocol)."
    )
    add_dataset_args(parser, include_camus=True)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_MEDSAM3_CHECKPOINT)
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help=f"Path to LoRA config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Input resolution (default: 1008)",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5)",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="",
        help="Override the per-class concept from datasets/label_text.py with this single "
        "prompt for ALL samples (e.g. 'object'). Use a distinct --output-dir to keep the run separate.",
    )
    parser.add_argument(
        "--use-gt-bbox",
        action="store_true",
        help="Also feed a bounding box derived from the GT mask alongside the text concept "
        "(the paper's MedSAM-3 'T+I' protocol). Without this flag the run is pure text.",
    )
    parser.add_argument(
        "--box-padding",
        type=int,
        default=0,
        help="Pixels of padding added around the GT-derived box prompt (only with --use-gt-bbox).",
    )
    parser.add_argument(
        "--no-auto-download-checkpoint",
        action="store_true",
        help="Require an existing local checkpoint instead of downloading it.",
    )
    parser.add_argument(
        "--checkpoint-repo-id",
        type=str,
        default=DEFAULT_MEDSAM3_CHECKPOINT_REPO_ID,
        help="Hugging Face model repo used when the checkpoint is missing.",
    )
    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        default=DEFAULT_MEDSAM3_CHECKPOINT_FILENAME,
        help="Checkpoint filename inside --checkpoint-repo-id.",
    )
    parser.add_argument("--checkpoint-revision", type=str, default="main")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5).",
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
    parser.add_argument("--output-dir", type=str, default= ROOT_DIR / "results/medsam3_text_prompt")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply augmentation to images and masks during inference.",
    )
    parser.add_argument(
        "--aug-scale-range",
        type=str,
        default="0.8,1.2",
        help="Scale range for augmentation as 'min,max' (default: 0.8,1.2).",
    )
    parser.add_argument(
        "--aug-shift-range",
        type=str,
        default="0.1,0.1",
        help="Shift range for augmentation as 'horizontal,vertical' fraction (default: 0.1,0.1).",
    )
    parser.add_argument(
        "--aug-seed",
        type=int,
        default=None,
        help="Random seed for augmentation (default: None for random).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse augmentation parameters
    aug_scale_range = None
    aug_shift_range = None
    if args.augment:
        try:
            scale_parts = [float(x.strip()) for x in args.aug_scale_range.split(",")]
            if len(scale_parts) != 2:
                raise ValueError("--aug-scale-range must have exactly 2 values")
            aug_scale_range = tuple(scale_parts)
        except Exception as e:
            raise ValueError(f"Invalid --aug-scale-range: {e}")
        
        try:
            shift_parts = [float(x.strip()) for x in args.aug_shift_range.split(",")]
            if len(shift_parts) != 2:
                raise ValueError("--aug-shift-range must have exactly 2 values")
            aug_shift_range = tuple(shift_parts)
        except Exception as e:
            raise ValueError(f"Invalid --aug-shift-range: {e}")

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
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    processor = MedSAM3TextInference(
        checkpoint=checkpoint,
        config_path=config_path,
        device=device,
        resolution=args.resolution,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )

    from PIL import Image

    rows: list[dict] = []
    skipped = 0
    metrics_calculator = SegmentationMetrics()
    benchmark_tic = time.perf_counter()
    total_label = str(total_iterations) if total_iterations is not None else "all available"
    protocol = "text+bbox" if args.use_gt_bbox else "text"
    print(
        f"Running MedSAM3 {protocol} benchmark for dataset '{args.dataset}' "
        f"on {total_label} samples (device={device})..."
    )

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    )

    for idx, sample in enumerate(decoder.iter_samples(max_samples=args.max_samples)):
        image_3c = ensure_three_channels(normalize_to_uint8(sample.image))
        H, W = image_3c.shape[:2]
        gt_mask = (sample.mask > 0).astype(np.uint8)

        # Apply augmentation if enabled
        if args.augment:
            aug_seed = args.aug_seed + idx if args.aug_seed is not None else None
            image_3c, gt_mask = apply_augmentation(
                image_3c,
                gt_mask,
                scale_range=aug_scale_range,
                shift_range=aug_shift_range,
                seed=aug_seed,
            )

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

        # In the T+I protocol, derive the box prompt from the GT mask. An empty mask
        # is already skipped above, so bbox_from_mask returns a valid box here.
        bbox = None
        if args.use_gt_bbox:
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

        concept = args.text_prompt or resolve_concept(args, sample)
        pil_image = Image.fromarray(image_3c)

        tic = time.perf_counter()
        if autocast is not None:
            autocast.__enter__()
        try:
            pred, score = processor.predict(pil_image, concept, bbox_xyxy=bbox)
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
            # Box prompt for the T+I path; an empty array (inert) for pure text.
            bbox=bbox if bbox is not None else np.array([], dtype=np.int32),
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
        writer = csv.DictWriter(f, fieldnames=TEXT_METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        summary = {
            "model": "medsam3",
            "protocol": protocol,
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "checkpoint": str(checkpoint),
            "config": str(config_path),
            "device": device,
            "resolution": args.resolution,
            "confidence_threshold": args.confidence_threshold,
            "nms_iou_threshold": args.nms_iou_threshold,
            "text_prompt_override": args.text_prompt or None,
            "use_gt_bbox": args.use_gt_bbox,
            "box_padding": args.box_padding if args.use_gt_bbox else None,
            "augmentation_enabled": args.augment,
            "aug_scale_range": args.aug_scale_range if args.augment else None,
            "aug_shift_range": args.aug_shift_range if args.augment else None,
            "aug_seed": args.aug_seed if args.augment else None,
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
            "model": "medsam3",
            "protocol": protocol,
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