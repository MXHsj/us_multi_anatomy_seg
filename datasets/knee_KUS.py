from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image
from skimage import io, transform

try:
    from datasets.common import (
        DecodedSample,
        export_samples,
        normalize_to_uint8,
        to_binary_mask,
    )
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


DEFAULT_ROOT = "datasets/KUS"
# KneeUS_Ilker masks are nnUNet predictions (plans.json / videos_with_preds), not
# hand-drawn ground truth, so the cohort is excluded from the benchmark dataset.
EXCLUDED_COHORTS = {"KneeUS_Ilker"}
TARGET_CLASS_NAME = "cartilage"
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask.astype(np.uint8)
    resized = transform.resize(
        mask.astype(np.uint8),
        shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return (resized > 0).astype(np.uint8)


def _resolve_mask_path(mask_dir: Path, image_path: Path) -> Optional[Path]:
    """Resolve the mask for an image.

    Three naming conventions are used across cohorts:
      * ``X.tif``  -> mask ``X.tif.png``  (full image filename + ``.png``)
      * ``X.png``  -> mask ``X.png``      (same filename)
      * ``X.png``  -> mask ``X_mask.png`` (stem + ``_mask.png``)
    Returns ``None`` when no matching mask file exists.
    """
    for candidate in (
        mask_dir / f"{image_path.name}.png",
        mask_dir / f"{image_path.stem}.png",
        mask_dir / f"{image_path.stem}_mask.png",
    ):
        if candidate.exists():
            return candidate
    return None


class KneeKUSDecoder:
    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"KUS root '{self.root}' not found.")

    def _cohorts(self) -> list[Path]:
        return sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir()
            and path.name not in EXCLUDED_COHORTS
            and (path / "images").is_dir()
            and (path / "masks").is_dir()
        )

    def _iter_pairs(self) -> Iterator[tuple[str, str, Path, Path]]:
        for cohort_dir in self._cohorts():
            cohort = cohort_dir.name
            image_dir = cohort_dir / "images"
            mask_dir = cohort_dir / "masks"
            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                mask_path = _resolve_mask_path(mask_dir, image_path)
                if mask_path is None:
                    continue
                yield cohort, image_path.stem, image_path, mask_path

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = sum(1 for _ in self._iter_pairs())
        return min(total, max_samples) if max_samples is not None else total

    def load_sample(self, sample_id: str) -> DecodedSample:
        cohort, sep, stem = sample_id.partition("/")
        if not sep:
            raise KeyError(f"KUS sample '{sample_id}' not found.")
        cohort_dir = self.root / cohort
        if cohort in EXCLUDED_COHORTS or not cohort_dir.is_dir():
            raise KeyError(f"KUS sample '{sample_id}' not found.")

        image_dir = cohort_dir / "images"
        mask_dir = cohort_dir / "masks"
        image_path = None
        for candidate in sorted(image_dir.glob(f"{stem}.*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                image_path = candidate
                break
        if image_path is None:
            raise KeyError(f"KUS sample '{sample_id}' not found.")
        mask_path = _resolve_mask_path(mask_dir, image_path)
        if mask_path is None:
            raise KeyError(f"KUS sample '{sample_id}' not found.")

        return self._load_sample(
            cohort=cohort, stem=stem, image_path=image_path, mask_path=mask_path
        )

    def _load_sample(
        self, cohort: str, stem: str, image_path: Path, mask_path: Path
    ) -> DecodedSample:
        image = io.imread(image_path)
        # A few files are saved as multi-frame TIFF stacks (frames, H, W[, C]);
        # they carry a single 2D mask, so collapse to the first frame.
        multiframe = image.ndim == 4 or (image.ndim == 3 and image.shape[-1] not in (1, 3, 4))
        if multiframe:
            image = image[0]
        with Image.open(mask_path) as mask_image:
            raw_mask_shape = (int(mask_image.height), int(mask_image.width))
            raw_mask = np.asarray(mask_image.convert("L"))
        mask = to_binary_mask(raw_mask)
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        original_image_shape = tuple(int(v) for v in image.shape)
        original_mask_shape = raw_mask_shape
        mask = _resize_mask_nearest(mask, image.shape[:2])

        return DecodedSample(
            dataset="KUS",
            sample_id=f"{cohort}/{stem}",
            image=normalize_to_uint8(image),
            mask=mask,
            metadata={
                "cohort": cohort,
                "target_class_id": TARGET_CLASS_NAME,
                "target_class_name": TARGET_CLASS_NAME,
                "target_index": 0,
                "targets_per_source": 1,
                "raw_image_path": str(image_path),
                "raw_mask_path": str(mask_path),
                "original_image_shape": original_image_shape,
                "original_mask_shape": original_mask_shape,
                "mask_resized": original_mask_shape[:2] != original_image_shape[:2],
                "multiframe_collapsed": multiframe,
            },
        )

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for cohort, stem, image_path, mask_path in self._iter_pairs():
            yield self._load_sample(
                cohort=cohort, stem=stem, image_path=image_path, mask_path=mask_path
            )
            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode KUS knee cartilage ultrasound dataset")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    decoder = KneeKUSDecoder(root=args.root)
    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            meta = sample.metadata
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"original_mask_shape={meta['original_mask_shape']}, "
                f"mask_resized={meta['mask_resized']}"
            )
            count += 1
        print(f"Decoded {count} samples")
