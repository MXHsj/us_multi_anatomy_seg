from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


_SUBJECTS = ("AP", "BM", "CP", "SG", "XM")
_LABELS = ("pleural_line", "rib_shadow")
_DEFAULT_LABELS = ("pleural_line",)


@dataclass(frozen=True)
class RobLUSSampleInfo:
    subject: str
    frame_id: str
    image_path: Path
    mask_paths: dict[str, Path]


def _natural_frame_key(path: Path) -> tuple[int, str]:
    stem = _remove_us_prefix(path.stem)
    try:
        return int(stem), stem
    except ValueError:
        return 0, stem


def _remove_us_prefix(stem: str) -> str:
    return stem[3:] if stem.startswith("US_") else stem


def _parse_csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


class RobLUSDecoder:
    def __init__(
        self,
        root: str | Path = "datasets/RobLUS",
        labels: Optional[list[str]] = None,
        subjects: Optional[list[str]] = None,
    ):
        self.root = Path(root)
        self.labels = labels or list(_DEFAULT_LABELS)
        self.subjects = subjects or list(_SUBJECTS)

        if not self.root.exists():
            raise FileNotFoundError(f"RobLUS root '{self.root}' not found.")

        unknown_labels = sorted(set(self.labels) - set(_LABELS))
        if unknown_labels:
            raise ValueError(f"Unsupported RobLUS labels: {unknown_labels}. Expected one of: {_LABELS}")

        unknown_subjects = sorted(set(self.subjects) - set(_SUBJECTS))
        if unknown_subjects:
            raise ValueError(f"Unsupported RobLUS subjects: {unknown_subjects}. Expected one of: {_SUBJECTS}")

    def _subject_dirs(self) -> list[Path]:
        return [self.root / subject for subject in self.subjects if (self.root / subject).is_dir()]

    def iter_sample_infos(self, max_samples: Optional[int] = None) -> Iterator[RobLUSSampleInfo]:
        count = 0
        for subject_dir in self._subject_dirs():
            subject = subject_dir.name
            for image_path in sorted(subject_dir.glob("US_*.jpg"), key=_natural_frame_key):
                frame_id = _remove_us_prefix(image_path.stem)
                mask_paths = {
                    label: subject_dir / "mask" / label / f"mask_{frame_id}.png"
                    for label in self.labels
                }
                if not all(mask_path.exists() for mask_path in mask_paths.values()):
                    continue
                yield RobLUSSampleInfo(
                    subject=subject,
                    frame_id=frame_id,
                    image_path=image_path,
                    mask_paths=mask_paths,
                )
                count += 1
                if max_samples is not None and count >= max_samples:
                    return

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = sum(1 for _ in self.iter_sample_infos(max_samples=max_samples))
        return total

    def load_sample_from_info(self, info: RobLUSSampleInfo) -> DecodedSample:
        image = io.imread(info.image_path)
        label_areas: dict[str, int] = {}
        merged_mask: np.ndarray | None = None
        for label, mask_path in info.mask_paths.items():
            mask = to_binary_mask(io.imread(mask_path))
            label_areas[label] = int(mask.sum())
            merged_mask = mask if merged_mask is None else (merged_mask | mask).astype(np.uint8)

        if merged_mask is None:
            raise ValueError("RobLUSDecoder requires at least one selected label.")

        return DecodedSample(
            dataset="RobLUS",
            sample_id=f"{info.subject}/US_{info.frame_id}",
            image=normalize_to_uint8(image),
            mask=merged_mask,
            metadata={
                "subject": info.subject,
                "frame_id": info.frame_id,
                "labels": list(self.labels),
                "label_areas": label_areas,
                "is_negative_control": int(merged_mask.sum()) == 0,
                "raw_image_path": str(info.image_path),
                "raw_mask_paths": {label: str(path) for label, path in info.mask_paths.items()},
            },
        )

    def load_sample(self, sample_id: str) -> DecodedSample:
        subject, sep, frame = sample_id.partition("/")
        if not sep:
            raise KeyError(f"RobLUS sample '{sample_id}' not found.")
        frame_id = _remove_us_prefix(frame)
        if subject not in self.subjects:
            raise KeyError(f"RobLUS sample '{sample_id}' not found.")

        subject_dir = self.root / subject
        image_path = subject_dir / f"US_{frame_id}.jpg"
        mask_paths = {
            label: subject_dir / "mask" / label / f"mask_{frame_id}.png"
            for label in self.labels
        }
        if not image_path.exists() or not all(path.exists() for path in mask_paths.values()):
            raise KeyError(f"RobLUS sample '{sample_id}' not found.")

        return self.load_sample_from_info(
            RobLUSSampleInfo(
                subject=subject,
                frame_id=frame_id,
                image_path=image_path,
                mask_paths=mask_paths,
            )
        )

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        for info in self.iter_sample_infos(max_samples=max_samples):
            yield self.load_sample_from_info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode cleaned RobLUS lung ultrasound dataset")
    parser.add_argument("--root", type=str, default="datasets/RobLUS")
    parser.add_argument(
        "--labels",
        type=str,
        default="pleural_line",
        help="Comma-separated labels to merge into the binary mask. Defaults to pleural_line.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Optional comma-separated subject filter, e.g. AP,BM.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    subjects = _parse_csv_values(args.subjects) or None
    decoder = RobLUSDecoder(
        root=args.root,
        labels=_parse_csv_values(args.labels),
        subjects=subjects,
    )

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        negative_controls = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            negative_controls += int(sample.metadata["is_negative_control"])
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"mask_pixels={int(sample.mask.sum())}, labels={sample.metadata['label_areas']}"
            )
            count += 1
        print(f"Decoded {count} samples ({negative_controls} negative controls)")
