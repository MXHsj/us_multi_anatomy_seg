from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
from skimage import io, transform


@dataclass
class DecodedSample:
    dataset: str
    sample_id: str
    image: np.ndarray
    mask: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - arr_min) / (arr_max - arr_min)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def ensure_three_channels(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        return arr[:, :, :3]
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def to_binary_mask(mask: np.ndarray, positive_labels: Optional[Iterable[int]] = None) -> np.ndarray:
    arr = np.asarray(mask)
    if positive_labels is None:
        return (arr > 0).astype(np.uint8)
    rounded = np.rint(arr).astype(np.int32)
    labels = np.array(list(positive_labels), dtype=np.int32)
    return np.isin(rounded, labels).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray, padding: int = 0) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min = max(int(xs.min()) - padding, 0)
    y_min = max(int(ys.min()) - padding, 0)
    x_max = min(int(xs.max()) + padding, mask.shape[1] - 1)
    y_max = min(int(ys.max()) + padding, mask.shape[0] - 1)
    return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)


def prepare_medsam_image(image: np.ndarray, size: int = 1024) -> np.ndarray:
    image_uint8 = normalize_to_uint8(image)
    image_3c = ensure_three_channels(image_uint8)
    image_rs = transform.resize(
        image_3c,
        (size, size),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    image_rs = image_rs.astype(np.float32)
    image_rs -= image_rs.min()
    image_rs /= max(float(image_rs.max()), 1e-8)
    return image_rs


def dice_score(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    inter = float(np.logical_and(gt_b, pred_b).sum())
    return (2.0 * inter + eps) / (float(gt_b.sum()) + float(pred_b.sum()) + eps)


def iou_score(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    inter = float(np.logical_and(gt_b, pred_b).sum())
    union = float(np.logical_or(gt_b, pred_b).sum())
    return (inter + eps) / (union + eps)


def export_samples(samples: Iterator[DecodedSample], out_dir: str | Path) -> int:
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    msk_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    count = 0
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("dataset,sample_id,image_path,mask_path,height,width,metadata\n")
        for sample in samples:
            sample_name = sample.sample_id.replace("/", "_")
            img_path = img_dir / f"{sample_name}.png"
            msk_path = msk_dir / f"{sample_name}.png"

            image_uint8 = normalize_to_uint8(sample.image)
            mask_uint8 = (sample.mask > 0).astype(np.uint8) * 255
            io.imsave(img_path, image_uint8, check_contrast=False)
            io.imsave(msk_path, mask_uint8, check_contrast=False)

            h, w = sample.mask.shape[:2]
            metadata_str = str(sample.metadata).replace(",", ";")
            f.write(
                f"{sample.dataset},{sample.sample_id},{img_path},{msk_path},{h},{w},{metadata_str}\n"
            )
            count += 1
    return count
