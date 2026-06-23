from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from skimage import io
from skimage.draw import polygon

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8


_CATEGORIES = ("Benign", "Malignant", "Normal")
_LABELS = ("mass", "liver", "outline")


def _polygon_mask(points: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not points:
        return mask
    coords = np.asarray(points, dtype=np.float32)
    rr, cc = polygon(coords[:, 1], coords[:, 0], shape=shape)
    mask[rr, cc] = 1
    return mask


class AULIDDecoder:
    def __init__(self, root: str | Path = "datasets/AULID", label: str = "mass"):
        self.root = Path(root)
        self.data_root = self.root / "AULID" if (self.root / "AULID").is_dir() else self.root
        self.label = label
        if label not in _LABELS:
            raise ValueError(f"Unsupported AULID label '{label}'. Expected one of: {_LABELS}")
        if not self.root.exists():
            raise FileNotFoundError(f"AULID root '{self.root}' not found.")

    def _image_paths(self) -> list[Path]:
        paths: list[Path] = []
        for category in _CATEGORIES:
            image_dir = self.data_root / category / "image"
            if image_dir.exists():
                paths.extend(sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem)))
        return paths

    def _mask_path(self, image_path: Path) -> Path:
        category_dir = image_path.parents[1]
        return category_dir / "segmentation" / self.label / f"{image_path.stem}.json"

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = sum(1 for image_path in self._image_paths() if self._mask_path(image_path).exists())
        return min(total, max_samples) if max_samples is not None else total

    def _load_sample(self, image_path: Path) -> Optional[DecodedSample]:
        mask_path = self._mask_path(image_path)
        if not mask_path.exists():
            return None

        image = io.imread(image_path)
        height, width = image.shape[:2]
        points = json.loads(mask_path.read_text(encoding="utf-8"))
        mask = _polygon_mask(points, shape=(height, width))
        if mask.sum() == 0:
            return None

        category = image_path.parents[1].name
        return DecodedSample(
            dataset="AULID",
            sample_id=f"{category}/{image_path.stem}",
            image=normalize_to_uint8(image),
            mask=mask,
            metadata={
                "category": category,
                "label": self.label,
                "raw_image_path": str(image_path),
                "raw_mask_path": str(mask_path),
            },
        )

    def load_sample(self, sample_id: str) -> DecodedSample:
        category, sep, stem = sample_id.partition("/")
        if sep:
            image_path = self.data_root / category / "image" / f"{stem}.jpg"
            if image_path.exists():
                sample = self._load_sample(image_path)
                if sample is not None:
                    return sample

        for image_path in self._image_paths():
            if f"{image_path.parents[1].name}/{image_path.stem}" == sample_id:
                sample = self._load_sample(image_path)
                if sample is not None:
                    return sample
        raise KeyError(f"AULID sample '{sample_id}' not found.")

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for image_path in self._image_paths():
            sample = self._load_sample(image_path)
            if sample is None:
                continue
            yield sample

            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode AULID liver ultrasound dataset")
    parser.add_argument("--root", type=str, default="datasets/AULID")
    parser.add_argument("--label", type=str, default="mass", choices=_LABELS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    decoder = AULIDDecoder(root=args.root, label=args.label)
    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            print(f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}")
            count += 1
        print(f"Decoded {count} samples")
