from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterator, Optional

from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


class ThyroidTNSC2020Decoder:
    def __init__(self, root: str | Path = "datasets/TNSC2020"):
        self.root = Path(root)
        self.image_dir = self.root / "image"
        self.mask_dir = self.root / "mask"
        self.train_csv = self.root / "train.csv"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected '{self.image_dir}' and '{self.mask_dir}' to exist."
            )

        self.category_map = self._read_category_map()

    def _read_category_map(self) -> Dict[str, int]:
        if not self.train_csv.exists():
            return {}
        mapping: Dict[str, int] = {}
        with self.train_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("ID", "")
                cate = row.get("CATE", "")
                if image_id:
                    mapping[image_id] = int(cate) if cate != "" else -1
        return mapping

    def _sample_ids(self) -> list[str]:
        image_ids = {p.name for p in self.image_dir.glob("*.PNG")}
        mask_ids = {p.name for p in self.mask_dir.glob("*.PNG")}
        return sorted(image_ids & mask_ids)

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = len(self._sample_ids())
        return min(total, max_samples) if max_samples is not None else total

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for sample_name in self._sample_ids():
            image = io.imread(self.image_dir / sample_name)
            mask = io.imread(self.mask_dir / sample_name)

            sample = DecodedSample(
                dataset="TNSC2020",
                sample_id=sample_name.replace(".PNG", ""),
                image=normalize_to_uint8(image),
                mask=to_binary_mask(mask),
                metadata={
                    "raw_file": sample_name,
                    "category": self.category_map.get(sample_name, -1),
                },
            )
            yield sample

            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode TNSC2020 ultrasound dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/TNSC2020",
        help="Path to TNSC2020 raw dataset root",
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
        "--no-auto-download",
        action="store_true",
        help="Deprecated; this decoder reads only local files.",
    )
    args = parser.parse_args()

    decoder = ThyroidTNSC2020Decoder(args.root)

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"category={sample.metadata['category']}"
            )
            count += 1
        print(f"Decoded {count} samples")
