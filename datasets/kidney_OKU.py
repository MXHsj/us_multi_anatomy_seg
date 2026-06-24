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


# Known OKU anatomy classes, keyed by slug, with stable class ids/colors so each
# anatomy is emitted as its own binary target (analogous to CAMUS labels).
_OKU_LABELS = {
    "capsule": {"id": 1, "name": "Capsule", "color": "#1f77b4"},
    "central_echo_complex": {"id": 2, "name": "Central Echo Complex", "color": "#ff7f0e"},
    "cortex": {"id": 3, "name": "Cortex", "color": "#2ca02c"},
    "medulla": {"id": 4, "name": "Medulla", "color": "#d62728"},
}

_FALLBACK_COLORS = (
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


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

    def _label_info(self, anatomy: str, target_index: int) -> dict[str, object]:
        slug = slugify_target_name(anatomy) if str(anatomy).strip() else "unlabeled"
        if slug in _OKU_LABELS:
            info = _OKU_LABELS[slug]
            return {"id": info["id"], "name": info["name"], "color": info["color"], "slug": slug}
        display_name = str(anatomy).strip() or "Unlabeled"
        return {
            "id": 100 + target_index,
            "name": display_name,
            "color": _FALLBACK_COLORS[target_index % len(_FALLBACK_COLORS)],
            "slug": slug,
        }

    def _ordered_anatomies(
        self, polys: List[Tuple[np.ndarray, np.ndarray, str]]
    ) -> List[str]:
        """Stable, de-duplicated list of anatomy classes present in an image.

        Known classes come first (in canonical id order), followed by any unknown
        classes in first-seen order. Honors ``anatomy_filter`` when provided.
        """
        present: List[str] = []
        for _, _, anatomy in polys:
            if self.anatomy_filter and anatomy not in self.anatomy_filter:
                continue
            if anatomy not in present:
                present.append(anatomy)

        def sort_key(anatomy: str) -> Tuple[int, int, str]:
            slug = slugify_target_name(anatomy) if str(anatomy).strip() else "unlabeled"
            if slug in _OKU_LABELS:
                return (0, int(_OKU_LABELS[slug]["id"]), slug)
            return (1, present.index(anatomy), slug)

        return sorted(present, key=sort_key)

    def _build_mask(
        self,
        image_shape: Tuple[int, int],
        polys: List[Tuple[np.ndarray, np.ndarray, str]],
        anatomy: str,
    ) -> Tuple[np.ndarray, int]:
        mask = np.zeros(image_shape, dtype=np.uint8)
        num_polygons = 0
        for xs, ys, poly_anatomy in polys:
            if poly_anatomy != anatomy:
                continue
            rr, cc = polygon(ys, xs, shape=image_shape)
            mask[rr, cc] = 1
            num_polygons += 1
        return mask, num_polygons

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = 0
        for img_path in self._image_paths():
            polys = self._annotations.get(img_path.name, [])
            if not polys:
                continue
            total += len(self._ordered_anatomies(polys))
            if max_samples is not None and total >= max_samples:
                return max_samples
        return min(total, max_samples) if max_samples is not None else total

    def _iter_image_samples(self, img_path: Path) -> Iterator[DecodedSample]:
        filename = img_path.name
        polys = self._annotations.get(filename, [])
        if not polys:
            return

        anatomies = self._ordered_anatomies(polys)
        if not anatomies:
            return

        image = io.imread(img_path)
        if image.ndim == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
        image_uint8 = normalize_to_uint8(image)
        source_sample_id = img_path.stem
        targets_per_source = len(anatomies)

        for target_index, anatomy in enumerate(anatomies):
            mask, num_polygons = self._build_mask((h, w), polys, anatomy)
            if mask.sum() == 0:
                continue

            label_info = self._label_info(anatomy, target_index)
            target_class_name = str(label_info["name"])
            target_metadata = make_target_metadata(
                source_sample_id=source_sample_id,
                target_class_id=label_info["id"],
                target_class_name=target_class_name,
                target_instance_id=0,
                target_color=str(label_info["color"]),
                target_index=target_index,
                targets_per_source=targets_per_source,
            )
            target_metadata.update(
                {
                    "filename": filename,
                    "anatomy": anatomy,
                    "num_polygons": num_polygons,
                }
            )
            target_sample_id = make_target_sample_id(
                source_sample_id=source_sample_id,
                target_class_name=target_class_name,
                target_instance_id=0,
            )

            yield DecodedSample(
                dataset="OKU",
                sample_id=target_sample_id,
                image=image_uint8,
                mask=mask,
                metadata=target_metadata,
            )

    def load_sample(self, sample_id: str) -> DecodedSample:
        source_stem, sep, target_slug = sample_id.rpartition("__")
        if not sep:
            # Backward-compatible: bare image stem -> first available target.
            source_stem, target_slug = sample_id, ""

        img_path = self.root / f"{source_stem}.png"
        if img_path.exists():
            for sample in self._iter_image_samples(img_path):
                if not target_slug or sample.sample_id == sample_id:
                    return sample
        raise KeyError(f"OKU sample '{sample_id}' not found.")

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for img_path in self._image_paths():
            for sample in self._iter_image_samples(img_path):
                yield sample
                count += 1
                if max_samples is not None and count >= max_samples:
                    return


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
                f"class={sample.metadata.get('target_class_name')}, "
                f"polygons={sample.metadata['num_polygons']}"
            )
            count += 1
        print(f"Decoded {count} samples")
