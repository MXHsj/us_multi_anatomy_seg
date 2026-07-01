from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.metrics import METRIC_NAMES


TARGET_METADATA_COLUMNS = [
    "source_sample_id",
    "target_uid",
    "target_class_id",
    "target_class_name",
    "target_instance_id",
    "target_color",
]

METRIC_FIELDNAMES = [
    "sample_id",
    *TARGET_METADATA_COLUMNS,
    "height",
    "width",
    "bbox",
    *METRIC_NAMES,
    "infer_ms",
]


@dataclass
class InferenceResult:
    sample: Any
    image: np.ndarray
    gt_mask: np.ndarray
    pred_mask: np.ndarray
    bbox: np.ndarray
    metrics: dict[str, float]
    infer_ms: float


def parse_max_samples(value: str | int | None) -> int | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if text == "all":
        return None

    try:
        max_samples = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--max-samples must be a positive integer or 'all'."
        ) from exc

    if max_samples <= 0:
        raise argparse.ArgumentTypeError(
            "--max-samples must be a positive integer or 'all'."
        )
    return max_samples


def format_max_samples(value: int | None) -> int | str:
    return "all" if value is None else value


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def has_target_metadata(sample: Any) -> bool:
    metadata = _metadata(sample)
    return bool(metadata.get("source_sample_id")) and bool(metadata.get("target_class_name"))


def build_metric_row(
    sample: Any,
    height: int,
    width: int,
    bbox: np.ndarray,
    metrics: dict[str, float],
    infer_ms: float,
) -> dict[str, Any]:
    metadata = _metadata(sample)
    row = {
        "sample_id": sample.sample_id,
        "height": height,
        "width": width,
        "bbox": bbox.tolist(),
        "infer_ms": infer_ms,
    }
    for metric_name in METRIC_NAMES:
        row[metric_name] = metrics[metric_name]
    for column in TARGET_METADATA_COLUMNS:
        row[column] = metadata.get(column, "")
    return row


def build_box_metric_record(
    sample: Any,
    height: int,
    width: int,
    infer_ms: float,
    boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    """One per-sample record holding metrics for each bounding box (for JSON output).

    `boxes` is a list of per-box dicts (e.g. ``{"box_index", "bbox", **METRIC_NAMES}``), one per
    prompt box, in prompt order.
    """
    metadata = _metadata(sample)
    record = {
        "sample_id": sample.sample_id,
        "height": height,
        "width": width,
        "infer_ms": infer_ms,
        "num_boxes": len(boxes),
    }
    for column in TARGET_METADATA_COLUMNS:
        record[column] = metadata.get(column, "")
    record["boxes"] = boxes
    return record


def count_source_iterations(decoder: Any, max_samples: int | None) -> int | None:
    count_fn = getattr(decoder, "count_source_samples", None)
    if callable(count_fn):
        return count_fn(max_samples=max_samples)
    return None


def evenly_spaced_zero_based_indices(total: int | None, count: int) -> set[int]:
    if count <= 0 or total is None or total <= 0:
        return set()
    save_count = min(count, total)
    return {int(round(idx)) for idx in np.linspace(0, total - 1, save_count)}


def summarize_target_class_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        class_name = str(row.get("target_class_name") or "")
        if not class_name:
            continue
        grouped.setdefault(class_name, []).append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for class_name in sorted(grouped):
        class_rows = grouped[class_name]
        class_ids = sorted(
            {
                str(row.get("target_class_id"))
                for row in class_rows
                if row.get("target_class_id") not in ("", None)
            }
        )
        summaries[class_name] = {
            "target_class_id": class_ids[0] if len(class_ids) == 1 else class_ids,
            "num_evaluated": len(class_rows),
            **summarize_metric_rows(class_rows),
        }
    return summaries


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for column in [*METRIC_NAMES, "infer_ms"]:
        mean, std = _nanmean_std([row.get(column) for row in rows])
        summary[f"{column}_mean"] = mean
        summary[f"{column}_std"] = std
    return summary


def _nanmean_std(values: list[Any]) -> tuple[float | None, float | None]:
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    arr = np.array(numeric_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(arr.mean()), float(arr.std())


def count_unique_source_samples(rows: list[dict[str, Any]]) -> int:
    source_ids = {
        str(row.get("source_sample_id"))
        for row in rows
        if row.get("source_sample_id") not in ("", None)
    }
    return len(source_ids)


class TargetVisualizationCollector:
    def __init__(
        self,
        vis_dir: Path,
        source_indices: set[int],
        model_label: str,
        num_workers: int | None = None,
    ):
        self.vis_dir = vis_dir
        self.source_indices = source_indices
        self.model_label = model_label
        self._current_source_id: str | None = None
        self._current_group: dict[str, Any] | None = None
        self._fallback_source_index = 0
        # matplotlib rendering is CPU-bound (~0.5s/figure) and the GIL serializes
        # threads, so figures are rendered across a pool of worker processes to
        # overlap with inference and saturate spare cores. "spawn" avoids forking
        # the CUDA-initialized parent (which torch warns against / can deadlock).
        if num_workers is None:
            num_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._futures: list[concurrent.futures.Future] = []

    def _enqueue_current_group(self) -> None:
        if self._current_source_id is not None and self._current_group is not None:
            self._futures.append(
                self._executor.submit(
                    _render_and_save_group,
                    self.vis_dir,
                    self.model_label,
                    self._current_source_id,
                    self._current_group,
                )
            )
        self._current_group = None

    def add_if_selected(
        self,
        sample: Any,
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        bbox: np.ndarray,
        dice: float,
        iou: float,
    ) -> bool:
        metadata = _metadata(sample)
        if has_target_metadata(sample):
            source_sample_id = str(metadata["source_sample_id"])
            source_index_raw = metadata.get("source_sample_index")
            try:
                source_index = int(source_index_raw)
            except (TypeError, ValueError):
                return True
        else:
            source_sample_id = str(getattr(sample, "sample_id", "sample"))
            source_index = self._fallback_source_index
            self._fallback_source_index += 1

        if source_sample_id != self._current_source_id:
            self._enqueue_current_group()
            self._current_source_id = source_sample_id
            self._current_group = None

        if source_index not in self.source_indices:
            return True

        if self._current_group is None:
            self._current_group = {
                "image": image,
                "targets": [],
            }
        self._current_group["targets"].append(
            {
                "class_name": str(metadata.get("target_class_name", "target")),
                "instance_id": str(metadata.get("target_instance_id", "0")),
                "target_index": metadata.get("target_index", ""),
                "color": str(metadata.get("target_color", "")),
                "gt_mask": gt_mask.copy(),
                "pred_mask": pred_mask.copy(),
                "bbox": bbox.copy(),
                "dice": dice,
                "iou": iou,
            }
        )
        return True

    def flush_pending(self) -> None:
        """Enqueue the current group and wait for all pending saves to complete."""
        self._enqueue_current_group()
        for future in concurrent.futures.as_completed(self._futures):
            try:
                future.result()
            except Exception as exc:
                print(f"[vis] save failed: {exc}", flush=True)
        self._futures.clear()
        self._executor.shutdown(wait=True)


def _render_and_save_group(
    vis_dir: Path,
    model_label: str,
    source_sample_id: str,
    group: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch, Rectangle

    targets = sorted(group["targets"], key=_target_sort_key)
    if not targets:
        return

    vis_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    titles = [
        "Image",
        "GT Targets + Box Prompts",
        f"{model_label} Predictions",
    ]
    for axis, title in zip(axes, titles):
        axis.imshow(group["image"])
        axis.set_title(title)
        axis.axis("off")

    legend_handles = []
    for target in targets:
        color = _safe_rgb(target["color"])
        if color is None:
            color = to_rgb("#d62728")
        _overlay_mask(axes[1], target["gt_mask"], color=color, alpha=0.42)
        _overlay_mask(axes[2], target["pred_mask"], color=color, alpha=0.42)
        for bbox in _iter_bboxes(target["bbox"]):
            axes[1].add_patch(
                Rectangle(
                    (bbox[0], bbox[1]),
                    max(float(bbox[2] - bbox[0]), 1.0),
                    max(float(bbox[3] - bbox[1]), 1.0),
                    edgecolor=color,
                    facecolor=(0, 0, 0, 0),
                    linewidth=1.8,
                )
            )
        label = (
            f"{target['class_name']} "
            f"D={target['dice']:.2f} I={target['iou']:.2f}"
        )
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=label))

    axes[2].legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.82,
        fontsize=7,
    )
    fig.suptitle(source_sample_id)
    fig.tight_layout()
    safe_name = source_sample_id.replace("/", "_")
    fig.savefig(vis_dir / f"{safe_name}.png", dpi=140)
    plt.close(fig)


def _safe_rgb(color: str) -> tuple[float, float, float] | None:
    if not color:
        return None
    try:
        from matplotlib.colors import to_rgb

        return to_rgb(color)
    except ValueError:
        return None


def _target_sort_key(target: dict[str, Any]) -> tuple[int, str]:
    try:
        target_index = int(target.get("target_index"))
    except (TypeError, ValueError):
        target_index = 10_000
    return target_index, str(target.get("class_name", ""))


def _overlay_mask(axis: Any, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return
    overlay = np.zeros((*mask_bool.shape, 4), dtype=np.float32)
    overlay[mask_bool, :3] = color
    overlay[mask_bool, 3] = alpha
    axis.imshow(overlay)


def _iter_bboxes(bbox: np.ndarray) -> np.ndarray:
    return np.asarray(bbox).reshape(-1, 4)
