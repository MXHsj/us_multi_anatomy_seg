from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


_FILE_RE = re.compile(
    r"(?P<patient>patient\d+)_(?P<view>2CH|4CH)_(?P<phase>ED|ES|half_sequence)_gt\.nii\.gz$"
)


class HeartCAMUSDecoder:
    def __init__(
        self,
        root: str | Path = "datasets/CAMUS",
        include_half_sequence: bool = False,
        positive_labels: tuple[int, ...] = (1, 2, 3),
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

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = 0
        for gt_path in self._gt_files():
            match = _FILE_RE.search(gt_path.name)
            if not match:
                continue
            total += self._frame_count_from_shape(self.nib.load(str(gt_path)).shape)
            if max_samples is not None and total >= max_samples:
                return max_samples
        return total

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

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
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
                sample_id = f"{patient}_{view}_{phase}_f{frame_idx:03d}"
                yield DecodedSample(
                    dataset="CAMUS",
                    sample_id=sample_id,
                    image=normalize_to_uint8(img_frame),
                    mask=to_binary_mask(msk_frame, self.positive_labels),
                    metadata={
                        "patient": patient,
                        "view": view,
                        "phase": phase,
                        "frame_index": frame_idx,
                        "raw_image_path": str(img_path),
                        "raw_mask_path": str(gt_path),
                    },
                )
                count += 1
                if max_samples is not None and count >= max_samples:
                    return


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
        default="1,2,3",
        help="Comma-separated positive label ids to include",
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
            print(f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}")
            count += 1
        print(f"Decoded {count} samples")
