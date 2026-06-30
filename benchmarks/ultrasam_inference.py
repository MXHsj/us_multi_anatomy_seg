from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage import io

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarks.eval_utils import (
    InferenceResult,
    METRIC_FIELDNAMES,
    TargetVisualizationCollector,
    build_box_metric_record,
    build_metric_row,
    count_source_iterations,
    count_unique_source_samples,
    evenly_spaced_zero_based_indices,
    format_max_samples,
    parse_max_samples,
    summarize_metric_rows,
    summarize_target_class_metrics,
)
from benchmarks.metrics import METRIC_NAMES, SegmentationMetrics
from datasets.common import (
    bbox_from_binary_mask,
    bbox_masks_from_mask,
    ensure_three_channels,
    normalize_to_uint8,
)
from datasets.loader import add_dataset_args, build_decoder_from_args


DEFAULT_ULTRASAM_REPO = "https://github.com/CAMMA-public/UltraSam"
DEFAULT_ULTRASAM_CHECKPOINT_URL = (
    "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"
)
DEFAULT_ULTRASAM_CONFIG = "configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py"

# Thin line-like targets where centerline Dice (clDice) replaces overlap Dice (still saved as "dice").
CENTERLINE_DICE_DATASETS = {"umud"}


def resolve_torch_device(requested_device: str) -> torch.device:
    requested = torch.device(requested_device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(
            f"Requested device '{requested_device}' but CUDA is unavailable; using 'cpu'.",
            flush=True,
        )
        return torch.device("cpu")
    return requested


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
) -> None:
    total_str = str(total) if total > 0 else "?"
    sample_label = sample_id if len(sample_id) <= 48 else f"...{sample_id[-45:]}"
    avg_rate = evaluated / elapsed_s if elapsed_s > 0 and evaluated > 0 else 0.0
    print(
        "\r"
        f"Progress {current}/{total_str} | evaluated={evaluated} | skipped={skipped} "
        f"| avg={avg_rate:.2f} samples/s | elapsed={format_duration(elapsed_s)} "
        f"| last={sample_label}",
        end="",
        flush=True,
    )


def default_ultrasam_dir() -> Path:
    env_dir = os_environ_path("ULTRASAM_DIR")
    if env_dir is not None:
        return env_dir
    opt_dir = Path("/opt/UltraSam")
    if opt_dir.exists():
        return opt_dir
    return ROOT_DIR / "work_dir" / "UltraSam"


def os_environ_path(name: str) -> Path | None:
    import os

    value = os.environ.get(name)
    return Path(value) if value else None


def resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def ensure_ultrasam_source(path: Path, auto_clone: bool, repo_url: str) -> Path:
    if (path / "configs" / "UltraSAM").exists() and (path / "endosam").exists():
        return path
    if not auto_clone:
        raise FileNotFoundError(
            f"UltraSam source tree not found at {path}. "
            "Build the devcontainer, set ULTRASAM_DIR, pass --ultrasam-dir, "
            "or use --auto-clone-source."
        )

    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(path)],
        check=True,
    )
    return path


def ensure_checkpoint(path: Path, auto_download: bool, url: str) -> Path:
    if path.exists():
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"UltraSAM checkpoint not found: {path}. "
            "Pass --checkpoint to a valid file or omit --no-auto-download-checkpoint."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"UltraSAM checkpoint not found at {path}. Downloading from {url}...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return path


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "sample"


def mask_to_coco_rle(mask: np.ndarray) -> dict[str, Any]:
    try:
        from pycocotools import mask as mask_utils
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycocotools is required for UltraSAM COCO export. "
            "Install the UltraSAM/OpenMMLab devcontainer dependencies."
        ) from exc

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# Extensions mmcv.imread (the UltraSAM LoadImageFromFile pipeline) can decode.
# NIfTI volumes (e.g. CAMUS .nii/.nii.gz) are not loadable and must be exported
# as decoded PNGs instead of referenced by their raw path.
_MMCV_LOADABLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def raw_image_path_from_sample(sample: Any) -> Path | None:
    metadata = getattr(sample, "metadata", {}) or {}
    for key in ("raw_image_path", "image_path", "source_image_path"):
        value = metadata.get(key)
        if value:
            path = Path(str(value))
            if path.exists() and path.suffix.lower() in _MMCV_LOADABLE_SUFFIXES:
                return path.resolve()
    return None


def resolve_coco_image_path(coco_dir: Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return coco_dir / path


def export_decoder_to_coco(
    decoder: Any,
    max_samples: int | None,
    box_padding: int,
    bbox_mode: str,
    export_dir: Path,
) -> tuple[Path, list[dict[str, Any]], int]:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    image_dir = export_dir / "images"
    export_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    skipped = 0

    for sample in decoder.iter_samples(max_samples=max_samples):
        mask = np.asarray(sample.mask).astype(np.uint8)
        component_masks = bbox_masks_from_mask(mask, mode=bbox_mode, min_area=16)
        bboxes = [
            bbox
            for component_mask in component_masks
            if (bbox := bbox_from_binary_mask(component_mask, padding=box_padding)) is not None
        ]
        if not bboxes:
            skipped += 1
            continue

        image_id = len(images) + 1
        height, width = mask.shape[:2]
        image_uint8 = ensure_three_channels(normalize_to_uint8(sample.image))
        raw_image_path = raw_image_path_from_sample(sample)
        if raw_image_path is None:
            image_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"images/{image_id:06d}_{safe_stem(sample.sample_id)}.png"
            io.imsave(export_dir / file_name, image_uint8, check_contrast=False)
        else:
            file_name = str(raw_image_path)

        for component_mask, bbox in zip(component_masks, bboxes):
            ann_id = len(annotations) + 1
            x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
            coco_bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": coco_bbox,
                    "area": int(component_mask.sum()),
                    "iscrowd": 0,
                    "segmentation": mask_to_coco_rle(component_mask),
                }
            )
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "height": int(height),
                "width": int(width),
            }
        )
        prompt_bboxes = np.stack(bboxes).astype(np.int32)
        record_bbox = (
            prompt_bboxes[0]
            if bbox_mode == "union" and len(prompt_bboxes) == 1
            else prompt_bboxes
        )
        records.append(
            {
                "image_id": image_id,
                "sample": sample,
                "image": image_uint8,
                "mask": mask,
                "bbox": record_bbox,
            }
        )

    ann_path = export_dir / "annotations.json"
    with ann_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "object"}],
            },
            handle,
        )
    return ann_path, records, skipped


def load_existing_coco_export_records(
    decoder: Any,
    max_samples: int | None,
    box_padding: int,
    bbox_mode: str,
    export_dir: Path,
) -> tuple[Path, list[dict[str, Any]], int] | None:
    ann_path = export_dir / "annotations.json"
    if not ann_path.exists():
        return None

    with ann_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if len(images) == 0 or len(annotations) == 0:
        return None
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        annotations_by_image_id.setdefault(int(annotation.get("image_id", -1)), []).append(annotation)

    records: list[dict[str, Any]] = []
    skipped = 0
    for sample in decoder.iter_samples(max_samples=max_samples):
        mask = np.asarray(sample.mask).astype(np.uint8)
        component_masks = bbox_masks_from_mask(mask, mode=bbox_mode, min_area=16)
        bboxes = [
            bbox
            for component_mask in component_masks
            if (bbox := bbox_from_binary_mask(component_mask, padding=box_padding)) is not None
        ]
        if not bboxes:
            skipped += 1
            continue

        image_id = len(records) + 1
        if image_id > len(images):
            return None
        image_info = images[image_id - 1]
        raw_image_path = raw_image_path_from_sample(sample)
        if raw_image_path is None:
            expected_name = f"images/{image_id:06d}_{safe_stem(sample.sample_id)}.png"
        else:
            expected_name = str(raw_image_path)
        image_path = resolve_coco_image_path(export_dir, expected_name)
        if image_info.get("id") != image_id or image_info.get("file_name") != expected_name:
            return None
        if not image_path.exists():
            return None
        expected_coco_bboxes = []
        for bbox in bboxes:
            x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
            expected_coco_bboxes.append([x0, y0, x1 - x0 + 1, y1 - y0 + 1])
        existing_annotations = annotations_by_image_id.get(image_id, [])
        existing_coco_bboxes = [annotation.get("bbox") for annotation in existing_annotations]
        if existing_coco_bboxes != expected_coco_bboxes:
            return None

        prompt_bboxes = np.stack(bboxes).astype(np.int32)
        record_bbox = (
            prompt_bboxes[0]
            if bbox_mode == "union" and len(prompt_bboxes) == 1
            else prompt_bboxes
        )
        records.append(
            {
                "image_id": image_id,
                "sample": sample,
                "image": ensure_three_channels(normalize_to_uint8(sample.image)),
                "mask": mask,
                "bbox": record_bbox,
            }
        )

    if len(records) != len(images):
        return None
    print(f"Reusing existing COCO prompt dataset at {export_dir}", flush=True)
    return ann_path, records, skipped


def import_ultrasam_modules(ultrasam_dir: Path) -> None:
    for path in (ultrasam_dir, ultrasam_dir / "datasets"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def apply_ultrasam_runtime_patches() -> None:
    """Apply runtime patches normally installed by UltraSAM's MMEngine hooks."""
    import torch.nn.functional as F
    from endosam.models.utils.custom_functional import multi_head_attention_forward

    F.multi_head_attention_forward = multi_head_attention_forward


def load_ultrasam_model(
    ultrasam_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    import_ultrasam_modules(ultrasam_dir)

    try:
        from mmengine.config import Config
        from mmengine.runner.checkpoint import load_checkpoint
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "UltraSAM requires OpenMMLab packages: mmengine, mmcv, mmdet, and mmpretrain. "
            "Rebuild the devcontainer or install the UltraSAM dependencies from the README."
        ) from exc

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    custom_imports = cfg.get("custom_imports")
    if custom_imports:
        import_modules_from_strings(**custom_imports)
    apply_ultrasam_runtime_patches()

    model = MODELS.build(cfg.model)
    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects the
    # mmengine objects (e.g. HistoryBuffer) pickled into the UltraSAM checkpoint.
    # mmengine's load_checkpoint doesn't expose the kwarg, so force weights_only=False
    # for this trusted checkpoint by temporarily patching torch.load.
    original_torch_load = torch.load

    def _torch_load_weights_only_false(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_weights_only_false
    try:
        load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    finally:
        torch.load = original_torch_load
    model.to(device)
    model.eval()
    return model, cfg


def build_ultrasam_dataloader(
    cfg: Any,
    coco_dir: Path,
    ann_path: Path,
    batch_size: int,
    num_workers: int,
):
    try:
        from mmengine.runner import Runner
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("mmengine is required to build the UltraSAM dataloader.") from exc

    cfg.test_dataloader.batch_size = batch_size
    cfg.test_dataloader.num_workers = num_workers
    cfg.test_dataloader.pin_memory = True
    cfg.test_dataloader.persistent_workers = num_workers > 0
    if num_workers > 0:
        cfg.test_dataloader.prefetch_factor = 4
    cfg.test_dataloader.dataset.data_root = str(coco_dir)
    cfg.test_dataloader.dataset.ann_file = ann_path.name
    cfg.test_dataloader.dataset.data_prefix = {"img": ""}
    cfg.test_dataloader.dataset.test_mode = True
    if "test_evaluator" in cfg and hasattr(cfg.test_evaluator, "ann_file"):
        cfg.test_evaluator.ann_file = str(ann_path)
    return Runner.build_dataloader(cfg.test_dataloader)


def tensor_mask_to_numpy(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    if hasattr(mask, "detach"):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = np.asarray(arr).astype(bool)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape != shape:
        raise ValueError(f"Predicted mask shape mismatch: {arr.shape} vs {shape}")
    return arr.astype(np.uint8)


def read_box_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read the crash-safe per-box JSONL stream (one record per finished image).

    This per-box JSONL is the source of truth for resuming. Unparseable trailing lines (from a crash
    mid-write) are skipped.
    """
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def read_metric_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read prior per-sample rows back, coercing metric columns to float for summary stats."""
    if not csv_path.exists():
        return []
    numeric = {*METRIC_NAMES, "infer_ms"}
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in numeric:
                try:
                    row[column] = float(row[column])
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def write_pending_annotations(ann_path: Path, keep_image_ids: set[int], out_path: Path) -> Path:
    """Write a COCO annotations file with only the given image ids (original ids preserved)."""
    coco = json.loads(ann_path.read_text())
    coco["images"] = [image for image in coco["images"] if int(image["id"]) in keep_image_ids]
    coco["annotations"] = [ann for ann in coco["annotations"] if int(ann["image_id"]) in keep_image_ids]
    out_path.write_text(json.dumps(coco))
    return out_path


def dedup_by_sample_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one record per sample_id (last wins), preserving order."""
    return list({record["sample_id"]: record for record in records}.values())


@torch.no_grad()
def iter_ultrasam_predictions(model: Any, dataloader: Any, records_by_image_id: dict[int, Any]):
    use_amp = next(model.parameters()).is_cuda
    for data in dataloader:
        started = time.perf_counter()
        with torch.autocast("cuda", enabled=use_amp):
            outputs = model.test_step(data)
        infer_ms = (time.perf_counter() - started) * 1000.0 / max(len(outputs), 1)
        for data_sample in outputs:
            image_id = int(getattr(data_sample, "img_id", data_sample.metainfo["img_id"]))
            record = records_by_image_id[image_id]
            pred_instances = data_sample.pred_instances
            # One predicted mask per prompt box, in prompt order (see SAM mask decoder split by
            # len(gt_instances)); per_box_masks[i] aligns with the i-th GT component of the mask.
            per_box_masks = [
                tensor_mask_to_numpy(pred_instances.masks[mask_idx], record["mask"].shape)
                for mask_idx in range(len(pred_instances))
            ]
            pred_mask = np.zeros_like(record["mask"], dtype=np.uint8)
            for box_mask in per_box_masks:
                pred_mask |= box_mask
            yield record, pred_mask, per_box_masks, infer_ms


class _SampleListDecoder:
    def __init__(self, samples: list[Any]):
        self.samples = samples

    def iter_samples(self, max_samples: int | None = None):
        samples = self.samples if max_samples is None else self.samples[:max_samples]
        yield from samples


def run_ultrasam_on_samples(
    samples,
    *,
    device: str = "cuda:0",
    batch_size: int = 4,
    num_workers: int = 4,
    checkpoint: str | Path = "work_dir/UltraSam/UltraSam.pth",
    checkpoint_url: str = DEFAULT_ULTRASAM_CHECKPOINT_URL,
    no_auto_download_checkpoint: bool = False,
    ultrasam_dir: str | Path | None = None,
    ultrasam_repo: str = DEFAULT_ULTRASAM_REPO,
    auto_clone_source: bool = False,
    config: str | Path = DEFAULT_ULTRASAM_CONFIG,
    box_padding: int = 0,
    bbox_mode: str = "individual",
) -> list[InferenceResult]:
    """Run the normal UltraSAM GT-box pipeline on already-decoded samples."""
    sample_list = list(samples)
    if not sample_list:
        return []

    resolved_ultrasam_dir = ensure_ultrasam_source(
        resolve_repo_path(ultrasam_dir or default_ultrasam_dir()),
        auto_clone=auto_clone_source,
        repo_url=ultrasam_repo,
    )
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = resolved_ultrasam_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"UltraSAM config not found: {config_path}")

    checkpoint_path = ensure_checkpoint(
        resolve_repo_path(checkpoint),
        auto_download=not no_auto_download_checkpoint,
        url=checkpoint_url,
    )
    torch_device = resolve_torch_device(device)

    with tempfile.TemporaryDirectory(prefix="ultrasam_sample_inference_") as tmp_dir:
        coco_dir = Path(tmp_dir) / "coco_export"
        ann_path, records, _ = export_decoder_to_coco(
            decoder=_SampleListDecoder(sample_list),
            max_samples=None,
            box_padding=box_padding,
            bbox_mode=bbox_mode,
            export_dir=coco_dir,
        )
        if not records:
            return []

        model, cfg = load_ultrasam_model(
            ultrasam_dir=resolved_ultrasam_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=torch_device,
        )
        dataloader = build_ultrasam_dataloader(
            cfg=cfg,
            coco_dir=coco_dir,
            ann_path=ann_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        records_by_image_id = {int(record["image_id"]): record for record in records}
        metrics_calculator = SegmentationMetrics()
        results: list[InferenceResult] = []
        for record, pred_mask, _per_box_masks, infer_ms in iter_ultrasam_predictions(
            model,
            dataloader,
            records_by_image_id,
        ):
            gt_mask = record["mask"]
            metrics = metrics_calculator.compute(gt_mask, pred_mask)
            results.append(
                InferenceResult(
                    sample=record["sample"],
                    image=record["image"],
                    gt_mask=gt_mask,
                    pred_mask=pred_mask,
                    bbox=record["bbox"],
                    metrics=metrics,
                    infer_ms=infer_ms,
                )
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Test UltraSAM inference with GT box prompts")
    add_dataset_args(parser, include_camus=True)
    parser.add_argument("--checkpoint", type=str, default="work_dir/UltraSam/UltraSam.pth")
    parser.add_argument(
        "--no-auto-download-checkpoint",
        action="store_true",
        help="Require an existing local UltraSAM checkpoint instead of downloading it.",
    )
    parser.add_argument(
        "--checkpoint-url",
        type=str,
        default=DEFAULT_ULTRASAM_CHECKPOINT_URL,
        help="URL used when the UltraSAM checkpoint is missing.",
    )
    parser.add_argument(
        "--ultrasam-dir",
        type=str,
        default=str(default_ultrasam_dir()),
        help="Path to the cloned CAMMA-public/UltraSam source tree.",
    )
    parser.add_argument(
        "--ultrasam-repo",
        type=str,
        default=DEFAULT_ULTRASAM_REPO,
        help="Git repo used by --auto-clone-source.",
    )
    parser.add_argument(
        "--auto-clone-source",
        action="store_true",
        help="Clone CAMMA-public/UltraSam into --ultrasam-dir if it is missing.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_ULTRASAM_CONFIG,
        help="UltraSam MMDetection config path, relative to --ultrasam-dir unless absolute.",
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
        "--centerline-tolerance",
        type=float,
        default=0.0,
        help=(
            "Euclidean pixel tolerance for centerline Dice (clDice) on line-like datasets "
            f"({', '.join(sorted(CENTERLINE_DICE_DATASETS))}). 0 (default) = strict clDice."
        ),
    )
    parser.add_argument(
        "--bbox-mode",
        choices=("union", "individual"),
        default="individual",
        help=(
            "Use one bbox around the full target mask (union) or one bbox "
            "per connected component larger than 15 pixels (individual, default)."
        ),
    )
    # batch_size=4 fits 1024x1024 SAM-encoder activations in ~10GB and is the
    # throughput optimum on 16GB cards (e.g. RTX 5080). Larger batches (8) saturate
    # 16GB VRAM, spill into system RAM over PCIe, and run ~5x slower; throughput
    # does not improve past 4 even on 24GB cards. Override with --batch-size.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-vis", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/ultrasam_test")
    args = parser.parse_args()

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coco_dir = output_dir / "coco_export"

    ultrasam_dir = ensure_ultrasam_source(
        resolve_repo_path(args.ultrasam_dir),
        auto_clone=args.auto_clone_source,
        repo_url=args.ultrasam_repo,
    )
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ultrasam_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"UltraSAM config not found: {config_path}")

    checkpoint = ensure_checkpoint(
        resolve_repo_path(args.checkpoint),
        auto_download=not args.no_auto_download_checkpoint,
        url=args.checkpoint_url,
    )
    device = resolve_torch_device(args.device)

    decoder = build_decoder_from_args(args)
    source_total = count_source_iterations(decoder, args.max_samples)
    existing_export = load_existing_coco_export_records(
        decoder=decoder,
        max_samples=args.max_samples,
        box_padding=args.box_padding,
        bbox_mode=args.bbox_mode,
        export_dir=coco_dir,
    )
    if existing_export is None:
        print("Exporting decoded samples to COCO prompt dataset...", flush=True)
        ann_path, records, export_skipped = export_decoder_to_coco(
            decoder=decoder,
            max_samples=args.max_samples,
            box_padding=args.box_padding,
            bbox_mode=args.bbox_mode,
            export_dir=coco_dir,
        )
    else:
        ann_path, records, export_skipped = existing_export
    if not records:
        raise RuntimeError("No non-empty masks were exported for UltraSAM evaluation.")

    metrics_path = output_dir / "per_sample_metrics.csv"
    box_jsonl_path = output_dir / "per_box_metrics.jsonl"
    box_metrics_path = output_dir / "per_box_metrics.json"

    # Resume when the export was reused (consistent with the current args) and prior per-box results
    # exist. The streaming JSONL only exists while a run is in progress; a finished run removes it and
    # leaves a single per_box_metrics.json. So prefer the JSONL (interrupted run); otherwise fall back
    # to the finished .json. A bare .json with no .jsonl means the run already completed.
    resume = existing_export is not None and (box_jsonl_path.exists() or box_metrics_path.exists())
    if not resume:
        prior_box_records: list[dict[str, Any]] = []
    elif box_jsonl_path.exists():
        prior_box_records = dedup_by_sample_id(read_box_records(box_jsonl_path))
    else:
        prior_box_records = dedup_by_sample_id(json.loads(box_metrics_path.read_text()))
    done = {record["sample_id"] for record in prior_box_records}
    # Prior per-sample rows from the CSV, restricted to ids confirmed done by the JSONL (drop any
    # CSV row left orphaned by a crash between the CSV and JSONL writes; that sample is re-run).
    rows: list[dict[str, Any]] = [row for row in read_metric_rows(metrics_path) if row["sample_id"] in done]
    box_records: list[dict[str, Any]] = prior_box_records
    pending = [record for record in records if record["sample"].sample_id not in done]
    if done:
        print(f"Resuming: {len(done)} samples already done, {len(pending)} remaining.", flush=True)

    metrics_calc = SegmentationMetrics(
        centerline_dice=args.dataset.lower() in CENTERLINE_DICE_DATASETS,
        centerline_tolerance=args.centerline_tolerance,
    )
    start_time = time.time()

    if pending:
        model, _cfg = load_ultrasam_model(
            ultrasam_dir=ultrasam_dir,
            config_path=config_path,
            checkpoint_path=checkpoint,
            device=device,
        )
        loader_ann_path = (
            write_pending_annotations(
                ann_path,
                {int(record["image_id"]) for record in pending},
                coco_dir / "annotations_pending.json",
            )
            if len(pending) < len(records)
            else ann_path
        )
        dataloader = build_ultrasam_dataloader(
            cfg=_cfg,
            coco_dir=coco_dir,
            ann_path=loader_ann_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        records_by_image_id = {int(record["image_id"]): record for record in pending}

        vis_collector = None
        if args.save_vis > 0:
            vis_total = source_total if source_total is not None else len(records)
            source_vis_indices = evenly_spaced_zero_based_indices(vis_total, args.save_vis)
            vis_collector = TargetVisualizationCollector(
                output_dir / "visualizations",
                source_vis_indices,
                model_label="UltraSAM",
            )

        # Stream per-image results so a crash keeps the finished samples. Write the CSV row before
        # the per-box JSONL line, so a JSONL entry (the resume marker) always implies its CSV row.
        csv_handle = metrics_path.open("a" if resume else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_handle, fieldnames=METRIC_FIELDNAMES)
        if not resume:
            writer.writeheader()
        jsonl_handle = box_jsonl_path.open("a" if box_jsonl_path.exists() else "w", encoding="utf-8")

        print(f"Running UltraSAM benchmark for dataset '{args.dataset}' on {len(pending)} samples...")
        try:
            for record, pred_mask, per_box_masks, infer_ms in iter_ultrasam_predictions(
                model, dataloader, records_by_image_id
            ):
                sample = record["sample"]
                mask = record["mask"]
                bbox = record["bbox"]
                height, width = mask.shape[:2]
                metrics = metrics_calc.compute(mask, pred_mask)
                row = build_metric_row(
                    sample=sample,
                    height=height,
                    width=width,
                    bbox=bbox,
                    metrics=metrics,
                    infer_ms=infer_ms,
                )

                # Per-bounding-box metrics: score each prompt box's prediction against its own GT
                # component, recomputed here in prompt order (avoids holding components in RAM).
                gt_components = bbox_masks_from_mask(mask, mode=args.bbox_mode, min_area=16)
                prompt_boxes = np.atleast_2d(np.asarray(bbox))
                boxes_payload: list[dict[str, Any]] = []
                for box_index, gt_component in enumerate(gt_components):
                    box_pred = (
                        per_box_masks[box_index]
                        if box_index < len(per_box_masks)
                        else np.zeros_like(mask, dtype=np.uint8)
                    )
                    box_metrics = metrics_calc.compute(np.asarray(gt_component).astype(np.uint8), box_pred)
                    boxes_payload.append(
                        {
                            "box_index": box_index,
                            "bbox": prompt_boxes[box_index].tolist()
                            if box_index < len(prompt_boxes)
                            else None,
                            **{name: float(box_metrics[name]) for name in METRIC_NAMES},
                        }
                    )
                box_record = build_box_metric_record(
                    sample=sample,
                    height=height,
                    width=width,
                    infer_ms=infer_ms,
                    boxes=boxes_payload,
                )

                writer.writerow(row)
                csv_handle.flush()
                jsonl_handle.write(json.dumps(box_record) + "\n")
                jsonl_handle.flush()
                rows.append(row)
                box_records.append(box_record)

                if vis_collector is not None:
                    vis_collector.add_if_selected(
                        sample=sample,
                        image=record["image"],
                        gt_mask=mask,
                        pred_mask=pred_mask,
                        bbox=bbox,
                        dice=metrics["dice"],
                        iou=metrics["iou"],
                    )

                print_progress(
                    current=len(rows),
                    total=len(records),
                    evaluated=len(rows),
                    skipped=export_skipped,
                    sample_id=sample.sample_id,
                    elapsed_s=time.time() - start_time,
                )
        finally:
            csv_handle.close()
            jsonl_handle.close()
        print()

        if vis_collector is not None:
            vis_collector.flush_pending()
    else:
        print(f"All {len(records)} samples already evaluated; finalizing outputs.", flush=True)

    # Finalize: dedup by sample_id (a crash-orphan sample may have been re-run), then rewrite the
    # derived CSV and per-box JSON cleanly from the in-memory results. The streaming JSONL is then
    # removed, so a finished run leaves a single per_box_metrics.json (its presence == run finished).
    rows = dedup_by_sample_id(rows)
    box_records = dedup_by_sample_id(box_records)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    box_metrics_path.write_text(json.dumps(box_records, indent=2))
    box_jsonl_path.unlink(missing_ok=True)

    target_class_summary = summarize_target_class_metrics(rows)
    summary = {
        "model": "ultrasam",
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "checkpoint": args.checkpoint,
        "checkpoint_url": args.checkpoint_url,
        "ultrasam_dir": str(ultrasam_dir),
        "config": str(config_path),
        "device": str(device),
        "max_samples": format_max_samples(args.max_samples),
        "box_padding": args.box_padding,
        "bbox_mode": args.bbox_mode,
        "batch_size": args.batch_size,
        "num_evaluated": len(rows),
        "num_skipped_empty_masks": export_skipped,
        "num_source_samples_evaluated": count_unique_source_samples(rows) or None,
        **summarize_metric_rows(rows),
    }
    if target_class_summary:
        summary["target_class_metrics"] = target_class_summary

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote per-box metrics to {box_metrics_path}")


if __name__ == "__main__":
    main()
