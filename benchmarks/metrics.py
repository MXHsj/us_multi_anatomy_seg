from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


METRIC_NAMES = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "hd95",
    "assd",
    "relative_area_error",
]


@dataclass(frozen=True)
class SegmentationMetrics:
    hd_percentile: float = 95.0
    spacing: tuple[float, ...] | None = None
    centerline_dice: bool = False
    centerline_tolerance: float = 0.0

    def compute(self, gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        gt_b = np.asarray(gt).astype(bool)
        pred_b = np.asarray(pred).astype(bool)
        if gt_b.shape != pred_b.shape:
            raise ValueError(f"Mask shape mismatch: {gt_b.shape} vs {pred_b.shape}")

        tp = float(np.logical_and(gt_b, pred_b).sum())
        tn = float(np.logical_and(~gt_b, ~pred_b).sum())
        fp = float(np.logical_and(~gt_b, pred_b).sum())
        fn = float(np.logical_and(gt_b, ~pred_b).sum())
        gt_area = tp + fn
        pred_area = tp + fp

        precision = _safe_rate(tp, tp + fp, empty_value=1.0 if gt_area == 0 else 0.0)
        recall = _safe_rate(tp, tp + fn, empty_value=1.0 if pred_area == 0 else 0.0)
        specificity = _safe_rate(tn, tn + fp, empty_value=1.0)
        if self.centerline_dice:
            dice = self._centerline_dice(gt_b, pred_b)
        else:
            dice = _safe_rate(2.0 * tp, 2.0 * tp + fp + fn, empty_value=1.0)
        iou = _safe_rate(tp, tp + fp + fn, empty_value=1.0)
        hd95, assd = self._surface_distances(gt_b, pred_b)

        if gt_area > 0:
            relative_area_error = (pred_area - gt_area) / gt_area
        else:
            relative_area_error = 0.0 if pred_area == 0 else float("nan")

        return {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": 0.5 * (recall + specificity),
            "hd95": hd95,
            "assd": assd,
            "relative_area_error": relative_area_error,
        }

    def _surface_distances(self, gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        gt_any = bool(np.any(gt))
        pred_any = bool(np.any(pred))
        if not gt_any and not pred_any:
            return 0.0, 0.0

        if not gt_any or not pred_any:
            penalty = _image_diagonal(gt.shape, self.spacing)
            return penalty, penalty

        gt_surface = _surface(gt)
        pred_surface = _surface(pred)
        if not np.any(gt_surface) or not np.any(pred_surface):
            penalty = _image_diagonal(gt.shape, self.spacing)
            return penalty, penalty

        spacing = _validate_spacing(self.spacing, gt.ndim)
        distances_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
        distances_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
        pred_to_gt = distances_to_gt[pred_surface]
        gt_to_pred = distances_to_pred[gt_surface]
        distances = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float64)
        if distances.size == 0:
            return 0.0, 0.0

        return (
            float(np.percentile(distances, self.hd_percentile)),
            float(distances.mean()),
        )

    def _centerline_dice(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Centerline Dice (clDice): topology-aware overlap of each mask's skeleton with the other
        mask, optionally within ``centerline_tolerance`` Euclidean pixels. Empty GT is skipped (NaN).
        """
        if not gt.any():
            return float("nan")
        if not pred.any():
            return 0.0

        skel_gt = skeletonize(gt)
        skel_pred = skeletonize(pred)
        skel_gt_sum = float(skel_gt.sum())
        skel_pred_sum = float(skel_pred.sum())
        if skel_gt_sum == 0 or skel_pred_sum == 0:
            return 0.0

        tolerance = self.centerline_tolerance
        gt_region = gt if tolerance <= 0 else ndimage.distance_transform_edt(~gt) <= tolerance
        pred_region = pred if tolerance <= 0 else ndimage.distance_transform_edt(~pred) <= tolerance

        tprec = float(np.logical_and(skel_pred, gt_region).sum()) / skel_pred_sum
        tsens = float(np.logical_and(skel_gt, pred_region).sum()) / skel_gt_sum
        if tprec + tsens == 0:
            return 0.0
        return 2.0 * tprec * tsens / (tprec + tsens)


def _safe_rate(numerator: float, denominator: float, empty_value: float) -> float:
    if denominator == 0:
        return float(empty_value)
    return float(numerator / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, ~eroded)


def _validate_spacing(spacing: Iterable[float] | None, ndim: int) -> tuple[float, ...] | None:
    if spacing is None:
        return None
    spacing_tuple = tuple(float(value) for value in spacing)
    if len(spacing_tuple) != ndim:
        raise ValueError(f"Expected {ndim} spacing values, got {len(spacing_tuple)}")
    return spacing_tuple


def _image_diagonal(shape: tuple[int, ...], spacing: tuple[float, ...] | None) -> float:
    spacing_tuple = _validate_spacing(spacing, len(shape))
    if spacing_tuple is None:
        spacing_tuple = tuple(1.0 for _ in shape)
    extents = [
        max(float(size - 1), 1.0) * spacing_value
        for size, spacing_value in zip(shape, spacing_tuple)
    ]
    return float(np.sqrt(np.sum(np.square(extents))))
