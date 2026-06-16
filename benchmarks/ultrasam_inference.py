from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
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
from datasets.common import bbox_from_mask, ensure_three_channels, normalize_to_uint8
from datasets.loader import add_dataset_args, build_decoder_from_args


DEFAULT_ULTRASAM_REPO = "https://github.com/CAMMA-public/UltraSam"
DEFAULT_ULTRASAM_CHECKPOINT_URL = (
    "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth"
)
DEFAULT_ULTRASAM_CONFIG = "configs/UltraSAM/UltraSAM_full/UltraSAM_box_refine.py"


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


def export_decoder_to_coco(
    decoder: Any,
    max_samples: int | None,
    box_padding: int,
    export_dir: Path,
) -> tuple[Path, list[dict[str, Any]], int]:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    image_dir = export_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    skipped = 0

    for sample in decoder.iter_samples(max_samples=max_samples):
        mask = np.asarray(sample.mask).astype(np.uint8)
        bbox = bbox_from_mask(mask, padding=box_padding)
        if bbox is None:
            skipped += 1
            continue

        image_id = len(images) + 1
        ann_id = len(annotations) + 1
        height, width = mask.shape[:2]
        file_name = f"{image_id:06d}_{safe_stem(sample.sample_id)}.png"
        image_uint8 = ensure_three_channels(normalize_to_uint8(sample.image))
        io.imsave(image_dir / file_name, image_uint8, check_contrast=False)

        x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
        coco_bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": coco_bbox,
                "area": int(mask.astype(bool).sum()),
                "iscrowd": 0,
                "segmentation": mask_to_coco_rle(mask),
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
        records.append(
            {
                "image_id": image_id,
                "sample": sample,
                "image": image_uint8,
                "mask": mask,
                "bbox": bbox,
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
    export_dir: Path,
) -> tuple[Path, list[dict[str, Any]], int] | None:
    ann_path = export_dir / "annotations.json"
    image_dir = export_dir / "images"
    if not ann_path.exists() or not image_dir.exists():
        return None

    with ann_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if len(images) != len(annotations):
        return None

    records: list[dict[str, Any]] = []
    skipped = 0
    for sample in decoder.iter_samples(max_samples=max_samples):
        mask = np.asarray(sample.mask).astype(np.uint8)
        bbox = bbox_from_mask(mask, padding=box_padding)
        if bbox is None:
            skipped += 1
            continue

        image_id = len(records) + 1
        if image_id > len(images):
            return None
        image_info = images[image_id - 1]
        expected_name = f"{image_id:06d}_{safe_stem(sample.sample_id)}.png"
        image_path = image_dir / expected_name
        if image_info.get("id") != image_id or image_info.get("file_name") != expected_name:
            return None
        if not image_path.exists():
            return None

        records.append(
            {
                "image_id": image_id,
                "sample": sample,
                "image": ensure_three_channels(normalize_to_uint8(sample.image)),
                "mask": mask,
                "bbox": bbox,
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
    cfg.test_dataloader.dataset.data_prefix = {"img": "images"}
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
            if len(pred_instances) == 0:
                pred_mask = np.zeros_like(record["mask"], dtype=np.uint8)
            else:
                masks = pred_instances.masks
                scores = pred_instances.scores
                if hasattr(scores, "detach"):
                    best_idx = int(torch.argmax(scores).detach().cpu().item())
                else:
                    best_idx = int(np.argmax(np.asarray(scores)))
                pred_mask = tensor_mask_to_numpy(masks[best_idx], record["mask"].shape)
            yield record, pred_mask, infer_ms


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
        export_dir=coco_dir,
    )
    if existing_export is None:
        print("Exporting decoded samples to COCO prompt dataset...", flush=True)
        ann_path, records, export_skipped = export_decoder_to_coco(
            decoder=decoder,
            max_samples=args.max_samples,
            box_padding=args.box_padding,
            export_dir=coco_dir,
        )
    else:
        ann_path, records, export_skipped = existing_export
    if not records:
        raise RuntimeError("No non-empty masks were exported for UltraSAM evaluation.")

    model, cfg = load_ultrasam_model(
        ultrasam_dir=ultrasam_dir,
        config_path=config_path,
        checkpoint_path=checkpoint,
        device=device,
    )
    dataloader = build_ultrasam_dataloader(
        cfg=cfg,
        coco_dir=coco_dir,
        ann_path=ann_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    records_by_image_id = {int(record["image_id"]): record for record in records}
    metrics_calc = SegmentationMetrics()
    rows: list[dict[str, Any]] = []
    total = len(records)
    start_time = time.time()

    vis_collector = None
    if args.save_vis > 0:
        vis_total = source_total if source_total is not None else total
        source_vis_indices = evenly_spaced_zero_based_indices(vis_total, args.save_vis)
        vis_collector = TargetVisualizationCollector(
            output_dir / "visualizations",
            source_vis_indices,
            model_label="UltraSAM",
        )

    print(f"Running UltraSAM benchmark for dataset '{args.dataset}' on {total} samples...")
    for idx, (record, pred_mask, infer_ms) in enumerate(
        iter_ultrasam_predictions(model, dataloader, records_by_image_id),
        start=1,
    ):
        sample = record["sample"]
        mask = record["mask"]
        bbox = record["bbox"]
        height, width = mask.shape[:2]
        metrics = metrics_calc.compute(mask, pred_mask)
        rows.append(
            build_metric_row(
                sample=sample,
                height=height,
                width=width,
                bbox=bbox,
                metrics=metrics,
                infer_ms=infer_ms,
            )
        )

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
            current=idx,
            total=total,
            evaluated=len(rows),
            skipped=export_skipped,
            sample_id=sample.sample_id,
            elapsed_s=time.time() - start_time,
        )
    print()

    if vis_collector is not None:
        vis_collector.flush_pending()

    metrics_path = output_dir / "per_sample_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

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


if __name__ == "__main__":
    main()
