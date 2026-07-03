from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image
from skimage import io, transform

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


TARGET_CLASS_ID = 1
TARGET_CLASS_NAME = "ovarian_tumor"
TARGET_COLOR = "#ffffff"


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


def _read_split_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_class_file(path: Path) -> dict[str, int]:
    classes: dict[str, int] = {}
    if not path.exists():
        return classes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        stem = Path(parts[0]).stem
        try:
            classes[stem] = int(parts[1])
        except ValueError:
            continue
    return classes


class MMOTU2DDecoder:
    def __init__(self, root: str | Path = "datasets/MMOTU"):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"MMOTU root '{self.root}' not found.")

        self.otu2d_root = self._resolve_otu2d_root()
        self.image_dir = self.otu2d_root / "images"
        self.mask_dir = self.otu2d_root / "annotations"
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected '{self.image_dir}' and '{self.mask_dir}' to exist."
            )

        self.split_by_stem = self._load_splits()
        self.class_by_stem = self._load_class_metadata()
        self.stems = self._sample_stems()

    def _resolve_otu2d_root(self) -> Path:
        candidates = [
            self.root / "MMOTU" / "OTU_2d",
            self.root / "OTU_2d",
            self.root,
        ]
        for candidate in candidates:
            if (candidate / "images").exists() and (candidate / "annotations").exists():
                return candidate
        raise FileNotFoundError(
            "Could not find MMOTU OTU_2d layout. Expected either "
            "'MMOTU/OTU_2d/' or direct 'OTU_2d/' under the root."
        )

    def _load_splits(self) -> dict[str, str]:
        split_by_stem: dict[str, str] = {}
        for split in ("train", "val"):
            for stem in _read_split_file(self.otu2d_root / f"{split}.txt"):
                split_by_stem[Path(stem).stem] = split
        return split_by_stem

    def _load_class_metadata(self) -> dict[str, int]:
        class_by_stem: dict[str, int] = {}
        for split in ("train", "val"):
            class_by_stem.update(_read_class_file(self.otu2d_root / f"{split}_cls.txt"))
        return class_by_stem

    def _sample_stems(self) -> list[str]:
        image_stems = {path.stem for path in self.image_dir.glob("*.JPG")}
        mask_suffix = "_binary.PNG"
        mask_stems = {
            path.name[: -len(mask_suffix)]
            for path in self.mask_dir.glob(f"*{mask_suffix}")
            if not path.name.endswith("_binary_binary.PNG")
        }
        def stem_key(value: str) -> tuple[int, int | str]:
            return (0, int(value)) if value.isdigit() else (1, value)

        paired = sorted(image_stems & mask_stems, key=stem_key)
        split_order = {"train": 0, "val": 1}
        return sorted(
            paired,
            key=lambda stem: (
                split_order.get(self.split_by_stem.get(stem, ""), 2),
                stem_key(stem),
            ),
        )

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = len(self.stems)
        return min(total, max_samples) if max_samples is not None else total

    def load_sample(self, sample_id: str) -> DecodedSample:
        parts = sample_id.split("/", maxsplit=1)
        stem = parts[-1]
        if stem not in self.stems:
            raise KeyError(f"MMOTU sample '{sample_id}' not found.")
        return self._load_sample(stem)

    def _load_sample(self, stem: str) -> DecodedSample:
        image_path = self.image_dir / f"{stem}.JPG"
        mask_path = self.mask_dir / f"{stem}_binary.PNG"
        if not image_path.exists() or not mask_path.exists():
            raise KeyError(f"MMOTU sample '{stem}' not found.")

        image = io.imread(image_path)
        with Image.open(mask_path) as mask_image:
            raw_mask_shape = (int(mask_image.height), int(mask_image.width))
            raw_mask = np.asarray(mask_image.convert("L"))

        mask = to_binary_mask(raw_mask)
        if mask.ndim == 3:
            mask = mask.max(axis=-1)

        original_image_shape = tuple(int(v) for v in image.shape)
        original_mask_shape = raw_mask_shape
        mask = _resize_mask_nearest(mask, image.shape[:2])

        split = self.split_by_stem.get(stem, "")
        source_sample_id = f"{split}/{stem}" if split else stem
        metadata = {
            "source_sample_id": source_sample_id,
            "target_class_id": TARGET_CLASS_ID,
            "target_class_name": TARGET_CLASS_NAME,
            "target_color": TARGET_COLOR,
            "target_index": 0,
            "targets_per_source": 1,
            "split": split,
            "raw_image_path": str(image_path),
            "raw_mask_path": str(mask_path),
            "original_image_shape": original_image_shape,
            "original_mask_shape": original_mask_shape,
            "mask_resized": original_mask_shape[:2] != original_image_shape[:2],
        }
        if stem in self.class_by_stem:
            metadata["classification_label_id"] = self.class_by_stem[stem]

        return DecodedSample(
            dataset="MMOTU",
            sample_id=source_sample_id,
            image=normalize_to_uint8(image),
            mask=mask,
            metadata=metadata,
        )

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for stem in self.stems:
            yield self._load_sample(stem)
            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode MMOTU 2D ovarian tumor dataset")
    parser.add_argument("--root", type=str, default="datasets/MMOTU")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    decoder = MMOTU2DDecoder(root=args.root)
    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        split_counts: dict[str, int] = {}
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            split = str(sample.metadata.get("split", ""))
            split_counts[split] = split_counts.get(split, 0) + 1
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"mask_pixels={int(sample.mask.sum())}, split={split}, "
                f"classification_label_id={sample.metadata.get('classification_label_id', '')}"
            )
            count += 1
        print(f"Decoded {count} samples")
        print(f"Split counts: {split_counts}")
