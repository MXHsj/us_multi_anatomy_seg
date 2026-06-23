from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

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


_FILE_RE = re.compile(
    r"(?P<patient>patient\d+)_(?P<view>2CH|4CH)_(?P<phase>ED|ES|half_sequence)_gt\.nii\.gz$"
)
_SAMPLE_RE = re.compile(
    r"(?P<patient>patient\d+)_(?P<view>2CH|4CH)_(?P<phase>ED|ES|half_sequence)_f(?P<frame>\d+)__(?P<target>[A-Za-z0-9_]+)$"
)

_CAMUS_LABELS = {
    1: {"name": "LV", "color": "#1f77b4", "description": "left_ventricle"},
    2: {"name": "MYO", "color": "#ff7f0e", "description": "myocardium"},
    3: {"name": "LA", "color": "#2ca02c", "description": "left_atrium"},
}

_FALLBACK_COLORS = (
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


class HeartCAMUSDecoder:
    def __init__(
        self,
        root: str | Path = "datasets/CAMUS",
        include_half_sequence: bool = False,
        positive_labels: tuple[int, ...] = (1, 3),
    ):
        self.root = Path(root)
        self.include_half_sequence = include_half_sequence
        self.positive_labels = positive_labels

        if not self.root.exists():
            raise FileNotFoundError(
                f"CAMUS root '{self.root}' not found. Add CAMUS data and retry."
            )

        try:
            import nibabel as nib  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "CAMUS decoding requires nibabel. Install it with 'pip install nibabel'."
            ) from exc

        self.nib = nib

    def _gt_files(self) -> list[Path]:
        files = sorted(self.root.glob("patient*/**/*_gt.nii.gz"))
        if self.include_half_sequence:
            return files
        return [p for p in files if "half_sequence" not in p.name]

    def _frame_count_from_shape(self, shape: tuple[int, ...]) -> int:
        squeezed = tuple(dim for dim in shape if dim != 1)
        if len(squeezed) == 2:
            return 1
        if len(squeezed) != 3:
            raise ValueError(f"Unsupported CAMUS shape {shape}")
        return int(min(squeezed))

    def _label_info(self, label: int, target_index: int) -> dict[str, str]:
        if label in _CAMUS_LABELS:
            return _CAMUS_LABELS[label]
        return {
            "name": f"label_{label}",
            "color": _FALLBACK_COLORS[target_index % len(_FALLBACK_COLORS)],
            "description": f"label_{label}",
        }

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        if max_samples is not None and max_samples <= 0:
            return 0
        total = 0
        targets_per_source = len(self.positive_labels)
        for gt_path in self._gt_files():
            match = _FILE_RE.search(gt_path.name)
            if not match:
                continue
            frame_count = self._frame_count_from_shape(self.nib.load(str(gt_path)).shape)
            total += frame_count * targets_per_source
            if max_samples is not None and total >= max_samples:
                return max_samples
        return total

    def count_source_samples(self, max_samples: Optional[int] = None) -> int:
        if max_samples is not None and max_samples <= 0:
            return 0
        total_sources = 0
        for gt_path in self._gt_files():
            match = _FILE_RE.search(gt_path.name)
            if not match:
                continue
            total_sources += self._frame_count_from_shape(
                self.nib.load(str(gt_path)).shape
            )

        if max_samples is None:
            return total_sources

        targets_per_source = max(len(self.positive_labels), 1)
        return min(total_sources, int(np.ceil(max_samples / targets_per_source)))

    def _iter_frames(self, img: np.ndarray, msk: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        if img.shape != msk.shape:
            raise ValueError(f"Image/mask shape mismatch: {img.shape} vs {msk.shape}")

        arr_img = np.squeeze(img)
        arr_msk = np.squeeze(msk)

        if arr_img.ndim == 2:
            yield arr_img, arr_msk, 0
            return

        if arr_img.ndim != 3:
            raise ValueError(f"Unsupported CAMUS shape {arr_img.shape}")

        frame_axis = int(np.argmin(arr_img.shape))
        arr_img = np.moveaxis(arr_img, frame_axis, -1)
        arr_msk = np.moveaxis(arr_msk, frame_axis, -1)
        for frame_idx in range(arr_img.shape[-1]):
            yield arr_img[..., frame_idx], arr_msk[..., frame_idx], frame_idx

    def _orient_frame(self, frame: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.rot90(frame, k=-1))

    def load_sample(self, sample_id: str) -> DecodedSample:
        match = _SAMPLE_RE.fullmatch(sample_id)
        if not match:
            raise KeyError(f"CAMUS sample '{sample_id}' not found.")

        patient = match.group("patient")
        view = match.group("view")
        phase = match.group("phase")
        frame_idx = int(match.group("frame"))
        target_slug = match.group("target")

        labels = tuple(int(label) for label in self.positive_labels)
        target_label: int | None = None
        target_index = 0
        target_class_name = ""
        label_info: dict[str, str] = {}
        for index, label in enumerate(labels):
            info = self._label_info(label=label, target_index=index)
            if slugify_target_name(info["name"]) == target_slug:
                target_label = label
                target_index = index
                target_class_name = info["name"]
                label_info = info
                break
        if target_label is None:
            raise KeyError(f"CAMUS sample '{sample_id}' not found.")

        gt_path = self.root / patient / f"{patient}_{view}_{phase}_gt.nii.gz"
        img_path = gt_path.with_name(gt_path.name.replace("_gt", ""))
        if not gt_path.exists() or not img_path.exists():
            raise KeyError(f"CAMUS sample '{sample_id}' not found.")

        img = self.nib.load(str(img_path)).get_fdata()
        msk = self.nib.load(str(gt_path)).get_fdata()
        for img_frame, msk_frame, current_frame_idx in self._iter_frames(img, msk):
            if current_frame_idx != frame_idx:
                continue

            source_sample_id = f"{patient}_{view}_{phase}_f{frame_idx:03d}"
            img_frame = self._orient_frame(img_frame)
            msk_frame = self._orient_frame(msk_frame)
            image_uint8 = normalize_to_uint8(img_frame)
            rounded_mask = np.rint(msk_frame).astype(np.int32)
            targets_per_source = len(labels)
            target_metadata = make_target_metadata(
                source_sample_id=source_sample_id,
                target_class_id=target_label,
                target_class_name=target_class_name,
                target_instance_id=0,
                target_color=label_info["color"],
                target_index=target_index,
                targets_per_source=targets_per_source,
            )
            target_metadata.update(
                {
                    "patient": patient,
                    "view": view,
                    "phase": phase,
                    "frame_index": frame_idx,
                    "target_description": label_info["description"],
                    "raw_image_path": str(img_path),
                    "raw_mask_path": str(gt_path),
                }
            )
            return DecodedSample(
                dataset="CAMUS",
                sample_id=sample_id,
                image=image_uint8,
                mask=(rounded_mask == target_label).astype(np.uint8),
                metadata=target_metadata,
            )

        raise KeyError(f"CAMUS sample '{sample_id}' not found.")

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        if max_samples is not None and max_samples <= 0:
            return
        target_count = 0
        source_count = 0
        labels = tuple(int(label) for label in self.positive_labels)
        targets_per_source = len(labels)
        for gt_path in self._gt_files():
            match = _FILE_RE.search(gt_path.name)
            if not match:
                continue

            patient = match.group("patient")
            view = match.group("view")
            phase = match.group("phase")
            img_path = gt_path.with_name(gt_path.name.replace("_gt", ""))

            img_nii = self.nib.load(str(img_path))
            msk_nii = self.nib.load(str(gt_path))
            img = img_nii.get_fdata()
            msk = msk_nii.get_fdata()

            for img_frame, msk_frame, frame_idx in self._iter_frames(img, msk):
                source_sample_id = f"{patient}_{view}_{phase}_f{frame_idx:03d}"
                img_frame = self._orient_frame(img_frame)
                msk_frame = self._orient_frame(msk_frame)
                image_uint8 = normalize_to_uint8(img_frame)
                rounded_mask = np.rint(msk_frame).astype(np.int32)

                for target_index, label in enumerate(labels):
                    label_info = self._label_info(label=label, target_index=target_index)
                    target_class_name = label_info["name"]
                    target_metadata = make_target_metadata(
                        source_sample_id=source_sample_id,
                        target_class_id=label,
                        target_class_name=target_class_name,
                        target_instance_id=0,
                        target_color=label_info["color"],
                        source_sample_index=source_count,
                        target_index=target_index,
                        targets_per_source=targets_per_source,
                    )
                    target_metadata.update(
                        {
                            "patient": patient,
                            "view": view,
                            "phase": phase,
                            "frame_index": frame_idx,
                            "target_description": label_info["description"],
                            "raw_image_path": str(img_path),
                            "raw_mask_path": str(gt_path),
                        }
                    )
                    target_sample_id = make_target_sample_id(
                        source_sample_id=source_sample_id,
                        target_class_name=target_class_name,
                        target_instance_id=0,
                    )

                    yield DecodedSample(
                        dataset="CAMUS",
                        sample_id=target_sample_id,
                        image=image_uint8,
                        mask=(rounded_mask == label).astype(np.uint8),
                        metadata=target_metadata,
                    )
                    target_count += 1
                    if max_samples is not None and target_count >= max_samples:
                        return

                source_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode CAMUS ultrasound dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/CAMUS",
        help="Path to CAMUS raw dataset root",
    )
    parser.add_argument(
        "--include-half-sequence",
        action="store_true",
        help="Include half-sequence volumes in addition to ED/ES",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="1,3",
        help="Comma-separated CAMUS label ids to evaluate as separate binary targets",
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
    args = parser.parse_args()

    labels = tuple(int(x) for x in args.labels.split(",") if x.strip())
    decoder = HeartCAMUSDecoder(
        root=args.root,
        include_half_sequence=args.include_half_sequence,
        positive_labels=labels,
    )

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"class={sample.metadata.get('target_class_name')}"
            )
            count += 1
        print(f"Decoded {count} samples")
