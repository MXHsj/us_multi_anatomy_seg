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


IMAGE_DIR = Path("apo_imgs_v1") / "apo_images_new_model_v1"
MASK_DIR = Path("apo_masks_v1") / "apo_masks_new_model_v1"
TARGET_CLASS_ID = "aponeurosis"
TARGET_CLASS_NAME = "aponeurosis"


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


class UMUDAponeurosisDecoder:
    def __init__(self, root: str | Path = "datasets/UMUD"):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"UMUD root '{self.root}' not found.")

        self.image_dir = self.root / IMAGE_DIR
        self.mask_dir = self.root / MASK_DIR
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected '{self.image_dir}' and '{self.mask_dir}' to exist."
            )

    def _sample_stems(self) -> list[str]:
        image_stems = {path.stem for path in self.image_dir.glob("*.tif")}
        mask_stems = {path.stem for path in self.mask_dir.glob("*.tif")}
        return sorted(image_stems & mask_stems)

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = len(self._sample_stems())
        return min(total, max_samples) if max_samples is not None else total

    def load_sample(self, sample_id: str) -> DecodedSample:
        prefix, sep, stem = sample_id.partition("/")
        if sep and prefix != TARGET_CLASS_NAME:
            raise KeyError(f"UMUD sample '{sample_id}' not found.")
        if not sep:
            stem = sample_id

        image_path = self.image_dir / f"{stem}.tif"
        mask_path = self.mask_dir / f"{stem}.tif"
        if not image_path.exists() or not mask_path.exists():
            raise KeyError(f"UMUD sample '{sample_id}' not found.")

        return self._load_sample(stem=stem, image_path=image_path, mask_path=mask_path)

    def _load_sample(self, stem: str, image_path: Path, mask_path: Path) -> DecodedSample:
        image = io.imread(image_path)
        with Image.open(mask_path) as mask_image:
            mask_photometric = mask_image.tag_v2.get(262)
            raw_mask_shape = (int(mask_image.height), int(mask_image.width))
            raw_mask = np.asarray(mask_image.convert("L"))
        mask = to_binary_mask(raw_mask)
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
        original_image_shape = tuple(int(v) for v in image.shape)
        original_mask_shape = raw_mask_shape
        mask = _resize_mask_nearest(mask, image.shape[:2])

        return DecodedSample(
            dataset="UMUD",
            sample_id=f"{TARGET_CLASS_NAME}/{stem}",
            image=normalize_to_uint8(image),
            mask=mask,
            metadata={
                "target_class_id": TARGET_CLASS_ID,
                "target_class_name": TARGET_CLASS_NAME,
                "target_index": 0,
                "targets_per_source": 1,
                "raw_image_path": str(image_path),
                "raw_mask_path": str(mask_path),
                "original_image_shape": original_image_shape,
                "original_mask_shape": original_mask_shape,
                "mask_photometric": mask_photometric,
                "mask_resized": original_mask_shape[:2] != original_image_shape[:2],
            },
        )

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for stem in self._sample_stems():
            yield self._load_sample(
                stem=stem,
                image_path=self.image_dir / f"{stem}.tif",
                mask_path=self.mask_dir / f"{stem}.tif",
            )
            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode UMUD aponeurosis ultrasound dataset")
    parser.add_argument("--root", type=str, default="datasets/UMUD")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    decoder = UMUDAponeurosisDecoder(root=args.root)
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
