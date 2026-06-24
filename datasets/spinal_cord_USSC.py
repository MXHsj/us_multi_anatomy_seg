from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from skimage import io

try:
    from datasets.common import (
        DecodedSample,
        export_samples,
        make_target_metadata,
        make_target_sample_id,
        normalize_to_uint8,
        slugify_target_name,
    )
except ModuleNotFoundError:
    from common import (
        DecodedSample,
        export_samples,
        make_target_metadata,
        make_target_sample_id,
        normalize_to_uint8,
        slugify_target_name,
    )


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class USSCLabel:
    id: int
    name: str
    color: tuple[int, int, int]
    description: str

    @property
    def slug(self) -> str:
        return slugify_target_name(self.name)

    @property
    def hex_color(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.color)


# The released masks use the common VOC/CVAT RGB palette. Class names are from
# the published semantic taxonomy for this dataset.
_USSC_LABELS = (
    USSCLabel(1, "dura", (128, 0, 0), "dura"),
    USSCLabel(2, "csf", (0, 128, 0), "cerebrospinal_fluid"),
    USSCLabel(3, "pia", (128, 128, 0), "pia"),
    USSCLabel(4, "spinal_cord", (0, 0, 128), "spinal_cord"),
    USSCLabel(5, "dorsal_space", (128, 0, 128), "dorsal_space"),
    USSCLabel(6, "hematoma", (0, 128, 128), "hematoma"),
    USSCLabel(7, "dura_pia_complex", (128, 128, 128), "dura_pia_complex"),
    USSCLabel(8, "dura_ventral_complex", (64, 0, 0), "dura_ventral_complex"),
    USSCLabel(9, "ventral_space", (192, 0, 0), "ventral_space"),
)

_LABELS_BY_ID = {label.id: label for label in _USSC_LABELS}
_LABELS_BY_TOKEN = {
    token: label
    for label in _USSC_LABELS
    for token in {
        str(label.id),
        label.name,
        label.slug,
        label.description,
        label.name.replace("_", " "),
        label.description.replace("_", " "),
    }
}


@dataclass(frozen=True)
class USSCSampleInfo:
    split: str
    stem: str
    image_path: Path
    mask_path: Path

    @property
    def source_sample_id(self) -> str:
        return f"{self.split}/{self.stem}"


def _parse_label_tokens(labels: Optional[list[str] | tuple[str, ...]]) -> tuple[USSCLabel, ...]:
    if not labels:
        return _USSC_LABELS

    parsed: list[USSCLabel] = []
    for value in labels:
        token = str(value).strip().lower().replace("-", "_")
        token = "_".join(token.split())
        if not token:
            continue
        if token not in _LABELS_BY_TOKEN:
            valid = ", ".join(label.name for label in _USSC_LABELS)
            raise ValueError(f"Unknown USSC label '{value}'. Expected one of: {valid}.")
        label = _LABELS_BY_TOKEN[token]
        if label not in parsed:
            parsed.append(label)

    return tuple(parsed) if parsed else _USSC_LABELS


def parse_csv_labels(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def parse_csv_splits(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _parse_splits(splits: Optional[list[str] | tuple[str, ...]]) -> tuple[str, ...]:
    if not splits:
        return SPLITS

    parsed: list[str] = []
    for value in splits:
        split = str(value).strip().lower()
        if not split:
            continue
        if split not in SPLITS:
            raise ValueError(f"Unknown USSC split '{value}'. Expected one of: {SPLITS}.")
        if split not in parsed:
            parsed.append(split)
    return tuple(parsed) if parsed else SPLITS


class SpinalCordUSDecoder:
    def __init__(
        self,
        root: str | Path = "datasets/USSC",
        labels: Optional[list[str] | tuple[str, ...]] = None,
        splits: Optional[list[str] | tuple[str, ...]] = None,
    ):
        root_path = Path(root)
        nested = root_path / "SegmentationDataset"
        if nested.is_dir():
            root_path = nested
        self.root = root_path
        self.labels = _parse_label_tokens(labels)
        self.splits = _parse_splits(splits)

        if not self.root.exists():
            raise FileNotFoundError(f"USSC root '{self.root}' not found.")

        missing = [
            folder
            for split in self.splits
            for folder in (self.root / f"{split}_images", self.root / f"{split}_masks")
            if not folder.is_dir()
        ]
        if missing:
            raise FileNotFoundError(f"USSC root '{self.root}' is missing folders: {missing}")

    def _iter_sample_infos(self, max_sources: Optional[int] = None) -> Iterator[USSCSampleInfo]:
        count = 0
        for split in self.splits:
            image_dir = self.root / f"{split}_images"
            mask_dir = self.root / f"{split}_masks"
            for image_path in sorted(image_dir.glob("*.png"), key=lambda path: path.name):
                mask_path = mask_dir / image_path.name
                if not mask_path.exists():
                    continue
                yield USSCSampleInfo(
                    split=split,
                    stem=image_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                )
                count += 1
                if max_sources is not None and count >= max_sources:
                    return

    def count_source_samples(self, max_samples: Optional[int] = None) -> int:
        total = sum(1 for _ in self._iter_sample_infos())
        if max_samples is None:
            return total
        targets_per_source = max(len(self.labels), 1)
        return min(total, int(np.ceil(max_samples / targets_per_source)))

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total_sources = sum(1 for _ in self._iter_sample_infos())
        total = total_sources * len(self.labels)
        return min(total, max_samples) if max_samples is not None else total

    def _read_mask_rgb(self, mask_path: Path) -> np.ndarray:
        mask = io.imread(mask_path)
        if mask.ndim == 2:
            raise ValueError(
                f"USSC mask '{mask_path}' is grayscale; expected an RGB semantic mask."
            )
        return np.asarray(mask[..., :3], dtype=np.uint8)

    def _sample_from_arrays(
        self,
        info: USSCSampleInfo,
        label: USSCLabel,
        target_index: int,
        image: np.ndarray,
        mask_rgb: np.ndarray,
    ) -> DecodedSample:
        target_color = np.array(label.color, dtype=np.uint8)
        target_mask = np.all(mask_rgb == target_color, axis=-1).astype(np.uint8)

        target_metadata = make_target_metadata(
            source_sample_id=info.source_sample_id,
            target_class_id=label.id,
            target_class_name=label.name,
            target_instance_id=0,
            target_color=label.hex_color,
            target_index=target_index,
            targets_per_source=len(self.labels),
        )
        target_metadata.update(
            {
                "split": info.split,
                "target_description": label.description,
                "raw_image_path": str(info.image_path),
                "raw_mask_path": str(info.mask_path),
            }
        )

        return DecodedSample(
            dataset="USSC",
            sample_id=make_target_sample_id(
                source_sample_id=info.source_sample_id,
                target_class_name=label.name,
                target_instance_id=0,
            ),
            image=image,
            mask=target_mask,
            metadata=target_metadata,
        )

    def _load_target(self, info: USSCSampleInfo, label: USSCLabel, target_index: int) -> DecodedSample:
        image = normalize_to_uint8(io.imread(info.image_path))
        mask_rgb = self._read_mask_rgb(info.mask_path)
        return self._sample_from_arrays(
            info=info,
            label=label,
            target_index=target_index,
            image=image,
            mask_rgb=mask_rgb,
        )

    def load_sample(self, sample_id: str) -> DecodedSample:
        source_sample_id, sep, target_slug = sample_id.rpartition("__")
        if not sep:
            raise KeyError(f"USSC sample '{sample_id}' not found.")

        split, split_sep, stem = source_sample_id.partition("/")
        if not split_sep or split not in SPLITS:
            raise KeyError(f"USSC sample '{sample_id}' not found.")

        selected_labels = tuple(self.labels)
        label_by_slug = {label.slug: (index, label) for index, label in enumerate(selected_labels)}
        if target_slug not in label_by_slug:
            raise KeyError(f"USSC sample '{sample_id}' not found.")

        image_path = self.root / f"{split}_images" / f"{stem}.png"
        mask_path = self.root / f"{split}_masks" / f"{stem}.png"
        if not image_path.exists() or not mask_path.exists():
            raise KeyError(f"USSC sample '{sample_id}' not found.")

        target_index, label = label_by_slug[target_slug]
        return self._load_target(
            USSCSampleInfo(split=split, stem=stem, image_path=image_path, mask_path=mask_path),
            label=label,
            target_index=target_index,
        )

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        if max_samples is not None and max_samples <= 0:
            return

        target_count = 0
        for source_index, info in enumerate(self._iter_sample_infos()):
            image = normalize_to_uint8(io.imread(info.image_path))
            mask_rgb = self._read_mask_rgb(info.mask_path)
            for target_index, label in enumerate(self.labels):
                sample = self._sample_from_arrays(
                    info=info,
                    label=label,
                    target_index=target_index,
                    image=image,
                    mask_rgb=mask_rgb,
                )
                sample.metadata["source_sample_index"] = source_index
                yield sample

                target_count += 1
                if max_samples is not None and target_count >= max_samples:
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode USSC semantic spinal cord ultrasound dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/USSC",
        help="Path to USSC root, containing SegmentationDataset or split image/mask folders.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated USSC labels. Defaults to all non-background classes.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated USSC splits to decode. Defaults to train,val,test.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    decoder = SpinalCordUSDecoder(
        root=args.root,
        labels=parse_csv_labels(args.labels),
        splits=parse_csv_splits(args.splits),
    )

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            meta = sample.metadata
            print(
                f"{sample.sample_id}: split={meta['split']}, "
                f"target={meta['target_class_name']}, "
                f"image={sample.image.shape}, mask={sample.mask.shape}, "
                f"mask_pixels={int(sample.mask.sum())}"
            )
            count += 1
        print(f"Decoded {count} targets")
