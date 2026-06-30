from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from skimage import io
from skimage.draw import polygon

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8


TARGET_CLASS_ID = 1
TARGET_CLASS_NAME = "gastrointestinal_submucosal_tumor"
TARGET_COLOR = "#ffffff"
ANNOTATION_FILE = "all_anno_crop.json"


def _polygon_mask(points: list[float], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not points:
        return mask
    coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    rr, cc = polygon(coords[:, 1], coords[:, 0], shape=shape)
    mask[rr, cc] = 1
    return mask


class GIST514Decoder:
    def __init__(self, root: str | Path = "datasets/GIST514"):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"GIST514 root '{self.root}' not found.")

        self.data_root = self._resolve_data_root()
        self.image_dir = self.data_root / "images"
        self.annotation_path = self.data_root / "annotations" / ANNOTATION_FILE
        if not self.image_dir.exists() or not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Expected '{self.image_dir}' and '{self.annotation_path}' to exist."
            )

        self.data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.categories = {
            int(category["id"]): str(category["name"])
            for category in self.data.get("categories", [])
        }
        self.images_by_id = {
            int(image["id"]): image
            for image in self.data.get("images", [])
        }
        self.annotations_by_image_id = {
            int(annotation["image_id"]): annotation
            for annotation in self.data.get("annotations", [])
        }
        self.image_ids = sorted(
            image_id
            for image_id in self.images_by_id
            if image_id in self.annotations_by_image_id
        )
        self.sample_id_by_image_id = {
            image_id: self._sample_id_for_image_id(image_id)
            for image_id in self.image_ids
        }
        self.image_id_by_sample_id = {
            sample_id: image_id
            for image_id, sample_id in self.sample_id_by_image_id.items()
        }

    def _resolve_data_root(self) -> Path:
        candidates = [
            self.root / "usd514_jpeg_roi",
            self.root / "GIST514-DB",
            self.root,
        ]
        for candidate in candidates:
            if (candidate / "images").exists() and (candidate / "annotations").exists():
                return candidate
        raise FileNotFoundError(
            "Could not find GIST514 layout. Expected 'usd514_jpeg_roi/' or direct "
            "'images/' and 'annotations/' under the root."
        )

    def _sample_id_for_image_id(self, image_id: int) -> str:
        image_info = self.images_by_id[image_id]
        annotation = self.annotations_by_image_id[image_id]
        category = self.categories.get(int(annotation.get("category_id", 0)), "unknown")
        stem = Path(str(image_info["file_name"])).stem
        return f"{category}/{stem}"

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        total = len(self.image_ids)
        return min(total, max_samples) if max_samples is not None else total

    def load_sample(self, sample_id: str) -> DecodedSample:
        image_id = self.image_id_by_sample_id.get(sample_id)
        if image_id is None:
            stem = sample_id.split("/", maxsplit=1)[-1]
            matches = [
                image_id
                for image_id in self.image_ids
                if Path(str(self.images_by_id[image_id]["file_name"])).stem == stem
            ]
            if len(matches) == 1:
                image_id = matches[0]
        if image_id is None:
            raise KeyError(f"GIST514 sample '{sample_id}' not found.")
        return self._load_sample(image_id)

    def _load_sample(self, image_id: int) -> DecodedSample:
        image_info = self.images_by_id[image_id]
        annotation = self.annotations_by_image_id[image_id]
        image_path = self.image_dir / str(image_info["file_name"])
        if not image_path.exists():
            raise FileNotFoundError(f"GIST514 image '{image_path}' not found.")

        image = io.imread(image_path)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for segment in annotation.get("segmentation", []):
            mask |= _polygon_mask(segment, shape=(height, width))

        category_id = int(annotation.get("category_id", 0))
        category = self.categories.get(category_id, "unknown")
        sample_id = self.sample_id_by_image_id[image_id]
        bbox = annotation.get("bbox", [])

        return DecodedSample(
            dataset="GIST514",
            sample_id=sample_id,
            image=normalize_to_uint8(image),
            mask=mask,
            metadata={
                "source_sample_id": sample_id,
                "target_class_id": TARGET_CLASS_ID,
                "target_class_name": TARGET_CLASS_NAME,
                "target_color": TARGET_COLOR,
                "target_index": 0,
                "targets_per_source": 1,
                "category": category,
                "category_id": category_id,
                "anatomy_id": image_info.get("anatomy_id", ""),
                "annotation_id": annotation.get("id", ""),
                "raw_image_path": str(image_path),
                "raw_annotation_path": str(self.annotation_path),
                "source_bbox_xywh": bbox,
                "original_image_shape": tuple(int(v) for v in image.shape),
                "original_mask_shape": tuple(int(v) for v in mask.shape),
            },
        )

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for image_id in self.image_ids:
            yield self._load_sample(image_id)
            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode GIST514-DB EUS ROI dataset")
    parser.add_argument("--root", type=str, default="datasets/GIST514")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    args = parser.parse_args()

    decoder = GIST514Decoder(root=args.root)
    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    else:
        count = 0
        category_counts: dict[str, int] = {}
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            category = str(sample.metadata.get("category", ""))
            category_counts[category] = category_counts.get(category, 0) + 1
            print(
                f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}, "
                f"mask_pixels={int(sample.mask.sum())}, category={category}"
            )
            count += 1
        print(f"Decoded {count} samples")
        print(f"Category counts: {category_counts}")
