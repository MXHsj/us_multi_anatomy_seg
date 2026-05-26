from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


CATEGORIES = ("benign", "malignant", "normal")


def _sanitize_sample_id(category: str, stem: str) -> str:
    match = re.search(r"\((\d+)\)", stem)
    if match:
        return f"{category}_{match.group(1)}"
    return re.sub(r"[^A-Za-z0-9_]", "_", stem)


class BreastBUSIDecoder:
    def __init__(
        self,
        root: str | Path = "datasets/BUSI",
        categories: Optional[Iterable[str]] = None,
    ):
        root_path = Path(root)
        nested = root_path / "Dataset_BUSI_with_GT"
        if nested.is_dir() and not (root_path / "benign").is_dir():
            root_path = nested
        self.root = root_path

        requested = (
            [c.strip().lower() for c in categories if str(c).strip()]
            if categories
            else list(CATEGORIES)
        )
        for category in requested:
            if category not in CATEGORIES:
                raise ValueError(
                    f"Unknown BUSI category '{category}'. Expected one of {CATEGORIES}."
                )
        self.categories = requested

        missing = [c for c in self.categories if not (self.root / c).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"BUSI root '{self.root}' is missing category folder(s): {missing}"
            )

    def _image_paths(self) -> List[tuple[str, Path]]:
        entries: List[tuple[str, Path]] = []
        for category in self.categories:
            category_dir = self.root / category
            for path in sorted(category_dir.glob(f"{category} (*).png")):
                if "_mask" in path.name:
                    continue
                entries.append((category, path))
        return entries

    def _mask_paths(self, image_path: Path) -> List[Path]:
        stem = image_path.stem
        primary = image_path.with_name(f"{stem}_mask.png")
        extras = sorted(image_path.parent.glob(f"{stem}_mask_*.png"))
        masks = [p for p in [primary, *extras] if p.exists()]
        return masks

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = sum(1 for _cat, img in self._image_paths() if self._mask_paths(img))
        return min(total, max_samples) if max_samples is not None else total

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for category, image_path in self._image_paths():
            mask_paths = self._mask_paths(image_path)
            if not mask_paths:
                continue

            image = io.imread(image_path)
            merged_mask = None
            for mask_path in mask_paths:
                mask_bin = to_binary_mask(io.imread(mask_path))
                if merged_mask is None:
                    merged_mask = mask_bin
                else:
                    merged_mask = (merged_mask | mask_bin).astype("uint8")

            if merged_mask is None:
                continue

            sample = DecodedSample(
                dataset="BUSI",
                sample_id=_sanitize_sample_id(category, image_path.stem),
                image=normalize_to_uint8(image),
                mask=merged_mask,
                metadata={
                    "category": category,
                    "image_file": image_path.name,
                    "mask_files": [p.name for p in mask_paths],
                    "num_masks": len(mask_paths),
                },
            )
            yield sample

            count += 1
            if max_samples is not None and count >= max_samples:
                break


def _parse_csv_arg(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode BUSI breast ultrasound dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/BUSI",
        help="Path to BUSI raw dataset root (containing benign/malignant/normal or Dataset_BUSI_with_GT)",
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
        "--categories",
        type=str,
        default="benign,malignant",
        help="Comma-separated BUSI categories (benign, malignant, normal).",
    )
    args = parser.parse_args()

    decoder = BreastBUSIDecoder(args.root, categories=_parse_csv_arg(args.categories))

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            meta = sample.metadata
            print(
                f"{sample.sample_id}: category={meta['category']}, "
                f"image={sample.image.shape}, mask={sample.mask.shape}, masks={meta['num_masks']}"
            )
            count += 1
        print(f"Decoded {count} samples")
