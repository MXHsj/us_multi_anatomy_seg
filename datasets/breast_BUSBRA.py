from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


class BreastBUSBRADecoder:
    def __init__(
        self,
        root: str | Path = "datasets/BUSBRA",
        pathology_filter: Optional[Iterable[str]] = None,
        birads_filter: Optional[Iterable[int]] = None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "Images"
        self.mask_dir = self.root / "Masks"
        self.bus_data_csv = self.root / "bus_data.csv"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected '{self.image_dir}' and '{self.mask_dir}' to exist."
            )

        self.pathology_filter = (
            {p.strip().lower() for p in pathology_filter if str(p).strip()}
            if pathology_filter
            else None
        )
        self.birads_filter = (
            {int(b) for b in birads_filter} if birads_filter else None
        )

        self.metadata_by_id = self._read_metadata()

    def _read_metadata(self) -> Dict[str, Dict[str, str]]:
        if not self.bus_data_csv.exists():
            return {}
        mapping: Dict[str, Dict[str, str]] = {}
        with self.bus_data_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = (row.get("ID") or "").strip()
                if sample_id:
                    mapping[sample_id] = row
        return mapping

    def _matches_filters(self, sample_id: str) -> bool:
        if self.pathology_filter is None and self.birads_filter is None:
            return True
        row = self.metadata_by_id.get(sample_id)
        if row is None:
            return False
        if self.pathology_filter is not None:
            pathology = (row.get("Pathology") or "").strip().lower()
            if pathology not in self.pathology_filter:
                return False
        if self.birads_filter is not None:
            birads_raw = (row.get("BIRADS") or "").strip()
            try:
                birads = int(birads_raw)
            except ValueError:
                return False
            if birads not in self.birads_filter:
                return False
        return True

    def _sample_ids(self) -> List[str]:
        image_stems = {p.stem for p in self.image_dir.glob("bus_*.png")}
        mask_stems = {
            p.stem.replace("mask_", "bus_", 1)
            for p in self.mask_dir.glob("mask_*.png")
        }
        paired = sorted(image_stems & mask_stems)
        return [sid for sid in paired if self._matches_filters(sid)]

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = len(self._sample_ids())
        return min(total, max_samples) if max_samples is not None else total

    def load_sample(self, sample_id: str) -> DecodedSample:
        if not self._matches_filters(sample_id):
            raise KeyError(f"BUSBRA sample '{sample_id}' not found.")

        image_path = self.image_dir / f"{sample_id}.png"
        mask_name = sample_id.replace("bus_", "mask_", 1) + ".png"
        mask_path = self.mask_dir / mask_name
        if not image_path.exists() or not mask_path.exists():
            raise KeyError(f"BUSBRA sample '{sample_id}' not found.")

        image = io.imread(image_path)
        mask = io.imread(mask_path)
        row = self.metadata_by_id.get(sample_id, {})

        birads_raw = (row.get("BIRADS") or "").strip()
        try:
            birads_value: Optional[int] = int(birads_raw) if birads_raw else None
        except ValueError:
            birads_value = None

        return DecodedSample(
            dataset="BUSBRA",
            sample_id=sample_id,
            image=normalize_to_uint8(image),
            mask=to_binary_mask(mask),
            metadata={
                "image_file": image_path.name,
                "mask_file": mask_path.name,
                "pathology": (row.get("Pathology") or "").strip() or None,
                "birads": birads_value,
                "side": (row.get("Side") or "").strip() or None,
                "device": (row.get("Device") or "").strip() or None,
                "histology": (row.get("Histology") or "").strip() or None,
                "bbox_csv": (row.get("BBOX") or "").strip() or None,
            },
        )

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for sample_id in self._sample_ids():
            yield self.load_sample(sample_id)

            count += 1
            if max_samples is not None and count >= max_samples:
                break


def _parse_csv_arg(value: str) -> List[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode BUS-BRA breast ultrasound dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/BUSBRA",
        help="Path to BUS-BRA raw dataset root",
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
        "--pathology",
        type=str,
        default="",
        help="Comma-separated pathology filter (benign, malignant)",
    )
    parser.add_argument(
        "--birads",
        type=str,
        default="",
        help="Comma-separated BI-RADS filter (e.g. 4,5)",
    )
    args = parser.parse_args()

    pathology_tokens = _parse_csv_arg(args.pathology)
    birads_tokens = [int(b) for b in _parse_csv_arg(args.birads)]

    decoder = BreastBUSBRADecoder(
        args.root,
        pathology_filter=pathology_tokens or None,
        birads_filter=birads_tokens or None,
    )

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            meta = sample.metadata
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"pathology={meta['pathology']}, birads={meta['birads']}, side={meta['side']}"
            )
            count += 1
        print(f"Decoded {count} samples")
