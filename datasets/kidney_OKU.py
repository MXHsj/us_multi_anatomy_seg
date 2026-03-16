from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from skimage import io
from skimage.draw import polygon

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8


def _parse_json_like(value: str) -> dict:
    if value is None:
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {}


def _extract_polygon(shape_attrs: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not shape_attrs:
        return None
    if shape_attrs.get("name") != "polygon":
        return None
    xs = shape_attrs.get("all_points_x", [])
    ys = shape_attrs.get("all_points_y", [])
    if not xs or not ys or len(xs) != len(ys):
        return None
    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


class KidneyOKUDecoder:
    def __init__(
        self,
        root: str | Path = "datasets/OKU",
        anatomy_filter: Optional[List[str]] = None,
        prefer_labels_2: bool = True,
    ):
        self.root = Path(root)
        self.labels_1 = self.root / "reviewed_labels_1.csv"
        self.labels_2 = self.root / "reviewed_labels_2.csv"
        self.prefer_labels_2 = prefer_labels_2
        self.anatomy_filter = [a.strip() for a in (anatomy_filter or []) if a.strip()]

        if not self.root.exists():
            raise FileNotFoundError(f"OKU root '{self.root}' not found.")
        if not self.labels_1.exists() and not self.labels_2.exists():
            raise FileNotFoundError("OKU labels CSV files not found.")

        self._annotations = self._load_annotations()

    def _read_csv(self, path: Path) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]]:
        annotations: Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "").strip()
                if not filename:
                    continue
                shape_attrs = _parse_json_like(row.get("region_shape_attributes", ""))
                region_attrs = _parse_json_like(row.get("region_attributes", ""))
                poly = _extract_polygon(shape_attrs)
                if poly is None:
                    continue
                anatomy = str(region_attrs.get("Anatomy", "")).strip()
                annotations.setdefault(filename, []).append((poly[0], poly[1], anatomy))
        return annotations

    def _load_annotations(self) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, str]]]:
        ann_1 = self._read_csv(self.labels_1) if self.labels_1.exists() else {}
        ann_2 = self._read_csv(self.labels_2) if self.labels_2.exists() else {}

        if self.prefer_labels_2:
            merged = dict(ann_2)
            for k, v in ann_1.items():
                if k not in merged:
                    merged[k] = v
            return merged
        merged = dict(ann_1)
        for k, v in ann_2.items():
            if k not in merged:
                merged[k] = v
        return merged

    def _image_paths(self) -> list[Path]:
        return sorted(self.root.glob("*.png"))

    def _build_mask(self, image_shape: Tuple[int, int], polys: List[Tuple[np.ndarray, np.ndarray, str]]) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        for xs, ys, anatomy in polys:
            if self.anatomy_filter and anatomy not in self.anatomy_filter:
                continue
            rr, cc = polygon(ys, xs, shape=image_shape)
            mask[rr, cc] = 1
        return mask

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for img_path in self._image_paths():
            filename = img_path.name
            polys = self._annotations.get(filename, [])
            if not polys:
                continue

            image = io.imread(img_path)
            if image.ndim == 3:
                h, w = image.shape[:2]
            else:
                h, w = image.shape
            mask = self._build_mask((h, w), polys)
            if mask.sum() == 0:
                continue

            sample = DecodedSample(
                dataset="OKU",
                sample_id=img_path.stem,
                image=normalize_to_uint8(image),
                mask=mask,
                metadata={
                    "filename": filename,
                    "num_polygons": len(polys),
                    "anatomy_filter": self.anatomy_filter,
                },
            )
            yield sample

            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode OKU kidney ultrasound dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/OKU",
        help="Path to OKU raw dataset root",
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
        "--anatomy",
        type=str,
        default="",
        help="Comma-separated anatomy filter (e.g., Capsule,Cortex)",
    )
    parser.add_argument(
        "--prefer-labels-1",
        action="store_true",
        help="Prefer reviewed_labels_1.csv over reviewed_labels_2.csv",
    )
    args = parser.parse_args()

    anatomy_filter = [a.strip() for a in args.anatomy.split(",") if a.strip()]
    decoder = KidneyOKUDecoder(
        root=args.root,
        anatomy_filter=anatomy_filter,
        prefer_labels_2=not args.prefer_labels_1,
    )

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"polygons={sample.metadata['num_polygons']}"
            )
            count += 1
        print(f"Decoded {count} samples")
