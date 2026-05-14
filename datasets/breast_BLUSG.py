from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8


def _mask_to_binary(mask) -> "object":
    arr = mask
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype("uint8")


class BreastBLUSGDecoder:
    def __init__(self, root: str | Path = "datasets/BLUSG", include_other: bool = True):
        self.root = Path(root)
        self.include_other = include_other

        if not self.root.exists():
            raise FileNotFoundError(f"BLUSG root '{self.root}' not found.")

    def _image_paths(self) -> list[Path]:
        candidates = sorted(self.root.glob("case*.png"))
        return [
            p
            for p in candidates
            if "_tumor" not in p.name and "_other" not in p.name
        ]

    def _mask_paths(self, stem: str) -> list[Path]:
        masks = list(self.root.glob(f"{stem}_tumor.png"))
        if self.include_other:
            masks.extend(self.root.glob(f"{stem}_other*.png"))
        return sorted(masks)

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = sum(1 for img_path in self._image_paths() if self._mask_paths(img_path.stem))
        return min(total, max_samples) if max_samples is not None else total

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for img_path in self._image_paths():
            stem = img_path.stem
            mask_paths = self._mask_paths(stem)
            if not mask_paths:
                continue

            image = io.imread(img_path)
            merged_mask = None
            for mask_path in mask_paths:
                mask = io.imread(mask_path)
                mask_bin = _mask_to_binary(mask)
                if merged_mask is None:
                    merged_mask = mask_bin
                else:
                    merged_mask = (merged_mask | mask_bin).astype("uint8")

            if merged_mask is None:
                continue

            sample = DecodedSample(
                dataset="BLUSG",
                sample_id=stem,
                image=normalize_to_uint8(image),
                mask=merged_mask,
                metadata={
                    "mask_files": [p.name for p in mask_paths],
                    "include_other": self.include_other,
                },
            )
            yield sample

            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode BLUSG breast ultrasound dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/BLUSG",
        help="Path to BLUSG raw dataset root",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of decoded samples",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="",
        help="Optional output directory for decoded PNGs + manifest",
    )
    parser.add_argument(
        "--only-tumor",
        action="store_true",
        help="Use only tumor masks, ignore other lesion masks",
    )
    args = parser.parse_args()

    decoder = BreastBLUSGDecoder(args.root, include_other=not args.only_tumor)

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"masks={len(sample.metadata['mask_files'])}"
            )
            count += 1
        print(f"Decoded {count} samples")
