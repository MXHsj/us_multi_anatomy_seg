from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator, Optional

from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_MASK_DIR_NAMES = {
    "mask",
    "masks",
    "label",
    "labels",
    "segmentation",
    "segmentations",
    "annotation",
    "annotations",
}
_MASK_SUFFIXES = ("_mask", "_masks", "_label", "_labels", "_seg", "_segmentation", "_gt")


def _is_mask_like_path(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & _MASK_DIR_NAMES:
        return True
    stem = path.stem.lower()
    return any(stem.endswith(suffix) for suffix in _MASK_SUFFIXES)


class FetusFTPDecoder:
    """Decoder for curated Fetal Planes / FTP segmentation-style materializations.

    The public FETAL_PLANES_DB is primarily a plane-classification dataset. This
    decoder therefore supports common paired image/mask layouts if masks are
    present in the curated zip, while yielding zero samples when only
    classification images/metadata are available.
    """

    def __init__(self, root: str | Path = "datasets/FTP"):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"FTP root '{self.root}' not found.")
        self.metadata_by_image = self._read_metadata()
        self._pairs = self._build_pairs()

    def _read_metadata(self) -> dict[str, dict[str, str]]:
        candidates = [
            self.root / "FETAL_PLANES_DB_data.csv",
            self.root / "fetal_planes_data.csv",
            self.root / "metadata.csv",
        ]
        metadata: dict[str, dict[str, str]] = {}
        for csv_path in candidates:
            if not csv_path.exists():
                continue
            text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
            sample = text[:2048]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(text.splitlines(), dialect=dialect)
            for row in reader:
                image_name = (
                    row.get("Image_name")
                    or row.get("image_name")
                    or row.get("filename")
                    or row.get("file_name")
                    or ""
                ).strip()
                if image_name:
                    metadata[Path(image_name).stem] = {k: v for k, v in row.items() if k}
            break
        return metadata

    def _image_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _IMAGE_EXTS
            and not _is_mask_like_path(path)
        )

    def _mask_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _IMAGE_EXTS
            and _is_mask_like_path(path)
        )

    def _build_pairs(self) -> list[tuple[Path, Path]]:
        masks = self._mask_paths()
        masks_by_stem = {path.stem.lower(): path for path in masks}
        pairs: list[tuple[Path, Path]] = []
        for image_path in self._image_paths():
            stem = image_path.stem.lower()
            candidates = [stem, *(f"{stem}{suffix}" for suffix in _MASK_SUFFIXES)]
            mask_path = next((masks_by_stem[name] for name in candidates if name in masks_by_stem), None)
            if mask_path is not None:
                pairs.append((image_path, mask_path))
        return pairs

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = len(self._pairs)
        return min(total, max_samples) if max_samples is not None else total

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for image_path, mask_path in self._pairs:
            image = io.imread(image_path)
            mask = io.imread(mask_path)
            metadata = {
                "raw_image_path": str(image_path),
                "raw_mask_path": str(mask_path),
            }
            metadata.update(self.metadata_by_image.get(image_path.stem, {}))

            yield DecodedSample(
                dataset="FTP",
                sample_id=image_path.relative_to(self.root).with_suffix("").as_posix(),
                image=normalize_to_uint8(image),
                mask=to_binary_mask(mask),
                metadata=metadata,
            )

            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode FTP / Fetal Planes segmentation pairs")
    parser.add_argument("--root", type=str, default="datasets/FTP")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    decoder = FetusFTPDecoder(root=args.root)
    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            print(f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}")
            count += 1
        print(f"Decoded {count} samples")
