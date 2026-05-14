from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


class UNSDecoder:
    def __init__(self, root: str | Path = "datasets/UNS"):
        self.root = Path(root)
        self.train_dir = self.root / "train"
        if not self.train_dir.exists():
            raise FileNotFoundError(f"UNS train directory '{self.train_dir}' not found.")

    def _image_paths(self) -> list[Path]:
        return sorted(
            (path for path in self.train_dir.glob("*.tif") if not path.stem.endswith("_mask")),
            key=lambda path: path.stem,
        )

    def _mask_path(self, image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}_mask.tif")

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = sum(1 for image_path in self._image_paths() if self._mask_path(image_path).exists())
        return min(total, max_samples) if max_samples is not None else total

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for image_path in self._image_paths():
            mask_path = self._mask_path(image_path)
            if not mask_path.exists():
                continue

            yield DecodedSample(
                dataset="UNS",
                sample_id=image_path.stem,
                image=normalize_to_uint8(io.imread(image_path)),
                mask=to_binary_mask(io.imread(mask_path)),
                metadata={
                    "split": "train",
                    "raw_image_path": str(image_path),
                    "raw_mask_path": str(mask_path),
                },
            )

            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode Ultrasound Nerve Segmentation dataset")
    parser.add_argument("--root", type=str, default="datasets/UNS")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    decoder = UNSDecoder(root=args.root)
    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            print(f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}")
            count += 1
        print(f"Decoded {count} samples")
