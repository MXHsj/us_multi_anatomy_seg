from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
)
from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8, to_binary_mask


class UltraBones100kDecoder:
    def __init__(
        self,
        root: str | Path = "datasets/UltraBones100k",
        label_folder: str = "Labels_full",
        thicken_radius: int = 0,
        fill_mask: bool = True,
        frame_fraction: float = 1.0,
    ):
        self.root = Path(root)
        self.label_folder = label_folder
        self.thicken_radius = thicken_radius
        self.fill_mask = fill_mask
        if not (0.0 < frame_fraction <= 1.0):
            raise ValueError(f"frame_fraction must be in (0, 1], got {frame_fraction}")
        self.frame_fraction = frame_fraction

        if not self.root.exists():
            raise FileNotFoundError(f"UltraBones100k root '{self.root}' not found.")

    def _select_record_frames(self, image_paths: list[Path]) -> list[Path]:
        """Stratify within a record (video): keep equally-spaced frames at
        ``frame_fraction`` (e.g. 0.1 -> ~10% of the clip), always at least one."""
        n = len(image_paths)
        if n == 0 or self.frame_fraction >= 1.0:
            return image_paths
        k = max(1, int(round(n * self.frame_fraction)))
        if k >= n:
            return image_paths
        indices = np.unique(np.linspace(0, n - 1, k).round().astype(int))
        return [image_paths[i] for i in indices]

    def _record_frame_paths(self, record_dir: Path) -> list[Path]:
        """Sorted image paths in a record that have a matching label, after
        applying the per-record frame sampling."""
        image_dir = record_dir / "UltrasoundImages"
        label_dir = record_dir / self.label_folder
        if not label_dir.exists():
            return []
        valid = [
            image_path
            for image_path in sorted(image_dir.glob("*.png"))
            if (label_dir / f"{image_path.stem}_label.png").exists()
        ]
        return self._select_record_frames(valid)

    def _record_dirs(self) -> list[Path]:
        # Each record folder contains sibling UltrasoundImages and label folders.
        image_dirs = sorted(self.root.glob("specimen*/*/record*/UltrasoundImages"))
        return [p.parent for p in image_dirs]

    def _tracking_map(self, record_dir: Path) -> Dict[str, Dict[str, str]]:
        # tracking.csv rows are keyed by timestamp, which is also the image stem.
        tracking_path = record_dir / "tracking.csv"
        if not tracking_path.exists():
            return {}

        rows: Dict[str, Dict[str, str]] = {}
        with tracking_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = row.get("timestamp", "").strip()
                if timestamp:
                    rows[timestamp] = row
        return rows

    def _record_metadata(self, record_dir: Path) -> Dict[str, str]:
        # Path layout: specimenXX/{anatomy}/recordXX.
        parts = record_dir.parts
        return {
            "specimen_id": parts[-3],
            "anatomy": parts[-2],
            "record": parts[-1],
        }

    def _completed_edge_contour(self, mask: np.ndarray) -> np.ndarray:
        mask_bin = np.asarray(mask).astype(bool)
        completed = mask_bin.copy()
        h, w = completed.shape
        edge_loop = (
            [(0, y) for y in range(h)]
            + [(x, h - 1) for x in range(1, w)]
            + [(w - 1, y) for y in range(h - 2, -1, -1)]
            + [(x, 0) for x in range(w - 2, 0, -1)]
        )
        hit_indices = [i for i, (x, y) in enumerate(edge_loop) if mask_bin[y, x]]
        if len(hit_indices) < 2:
            return completed

        start = hit_indices[0]
        stop = hit_indices[-1]
        for x, y in edge_loop[start : stop + 1]:
            completed[y, x] = True
        return completed

    def _fill_contour_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_bin = np.asarray(mask).astype(bool)
        if not mask_bin.any():
            return np.zeros_like(mask_bin, dtype=np.uint8)

        completed = self._completed_edge_contour(mask_bin)
        filled = binary_fill_holes(completed)
        return filled.astype(np.uint8)

    def _prepare_mask(self, mask) -> np.ndarray:
        mask_bin = to_binary_mask(mask)
        if mask_bin.ndim == 3:
            mask_bin = mask_bin.max(axis=-1)
        if self.fill_mask:
            # UltraBones100k labels can be contours; complete edge-touching
            # contours before filling the enclosed mask region.
            mask_bin = self._fill_contour_mask(mask_bin)

        if self.thicken_radius <= 0:
            return mask_bin

        # Optional distance-transform expansion for thin surface labels.
        distance_map = distance_transform_edt(~mask_bin.astype(bool))
        return (distance_map <= self.thicken_radius).astype(np.uint8)

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        count = 0
        for record_dir in self._record_dirs():
            for _ in self._record_frame_paths(record_dir):
                count += 1
                if max_samples is not None and count >= max_samples:
                    return count
        return count

    def _load_sample(
        self,
        sample_id: str,
        record_dir: Path,
        image_path: Path,
        label_path: Path,
        tracking: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> DecodedSample:
        image = io.imread(image_path)
        mask = io.imread(label_path)
        timestamp = image_path.stem
        record_metadata = self._record_metadata(record_dir)
        metadata = {
            **record_metadata,
            "timestamp": timestamp,
            "raw_image_path": str(image_path),
            "raw_mask_path": str(label_path),
            "label_folder": self.label_folder,
            "thicken_radius": self.thicken_radius,
            "fill_mask": self.fill_mask,
        }
        if tracking is None:
            tracking = self._tracking_map(record_dir)
        if timestamp in tracking:
            metadata["tracking"] = tracking[timestamp]

        return DecodedSample(
            dataset="UltraBones100k",
            sample_id=sample_id,
            image=normalize_to_uint8(image),
            mask=self._prepare_mask(mask),
            metadata=metadata,
        )

    def load_sample(self, sample_id: str) -> DecodedSample:
        parts = sample_id.split("_", maxsplit=3)
        if len(parts) != 4:
            raise KeyError(f"UltraBones100k sample '{sample_id}' not found.")
        specimen_id, anatomy, record, timestamp = parts
        record_dir = self.root / specimen_id / anatomy / record
        image_path = record_dir / "UltrasoundImages" / f"{timestamp}.png"
        label_path = record_dir / self.label_folder / f"{timestamp}_label.png"
        if not image_path.exists() or not label_path.exists():
            raise KeyError(f"UltraBones100k sample '{sample_id}' not found.")
        return self._load_sample(sample_id, record_dir, image_path, label_path)

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for record_dir in self._record_dirs():
            frame_paths = self._record_frame_paths(record_dir)
            if not frame_paths:
                continue

            label_dir = record_dir / self.label_folder
            record_metadata = self._record_metadata(record_dir)
            tracking = self._tracking_map(record_dir)

            for image_path in frame_paths:
                # UltraBones pairs 24363.png with 24363_label.png.
                label_path = label_dir / f"{image_path.stem}_label.png"

                sample_id = (
                    f"{record_metadata['specimen_id']}_{record_metadata['anatomy']}_"
                    f"{record_metadata['record']}_{image_path.stem}"
                )
                yield self._load_sample(
                    sample_id=sample_id,
                    record_dir=record_dir,
                    image_path=image_path,
                    label_path=label_path,
                    tracking=tracking,
                )

                count += 1
                if max_samples is not None and count >= max_samples:
                    return


def visualize_samples(samples: Iterator[DecodedSample]) -> int:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Visualization requires matplotlib. Install it with 'pip install matplotlib'."
        ) from exc

    count = 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    try:
        for sample in samples:
            count += 1
            for ax in axes:
                ax.clear()
                ax.axis("off")

            axes[0].imshow(sample.image, cmap="gray")
            axes[0].set_title("Image")

            axes[1].imshow(sample.image, cmap="gray")
            axes[1].imshow(sample.mask, cmap="Reds", alpha=0.35)
            axes[1].set_title("Mask overlay")

            fig.suptitle(sample.sample_id)
            fig.tight_layout()
            fig.canvas.draw_idle()
            plt.show(block=False)

            # Matplotlib key events arrive asynchronously; buffer them until handled.
            pressed: list[str] = []

            def on_key(event):
                pressed.append(event.key or "")

            cid = fig.canvas.mpl_connect("key_press_event", on_key)
            try:
                while plt.fignum_exists(fig.number):
                    plt.pause(0.05)
                    if not pressed:
                        continue
                    key = pressed.pop(0).lower()
                    # n advances to the next sample; q quits the viewer.
                    if key == "n":
                        break
                    if key == "q":
                        return count
            finally:
                fig.canvas.mpl_disconnect(cid)

            if not plt.fignum_exists(fig.number):
                break
    finally:
        plt.close(fig)

    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode UltraBones100k ultrasound dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/UltraBones100k",
        help="Path to UltraBones100k raw dataset root",
    )
    parser.add_argument(
        "--label-folder",
        type=str,
        default="Labels_full",
        help="Label folder inside each record directory",
    )
    parser.add_argument(
        "--thicken-radius",
        type=int,
        default=0,
        help="Distance-transform radius used to thicken labels; use 0 for original masks",
    )
    parser.add_argument(
        "--no-fill-mask",
        action="store_true",
        help="Use original thin surface labels instead of the default filled bone-shadow region",
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
        "--no-visualize",
        action="store_true",
        help="Print sample shapes instead of opening the interactive viewer",
    )
    args = parser.parse_args()

    decoder = UltraBones100kDecoder(
        args.root,
        label_folder=args.label_folder,
        thicken_radius=args.thicken_radius,
        fill_mask=not args.no_fill_mask,
    )

    if args.export_dir:
        n = export_samples(decoder.iter_samples(max_samples=args.max_samples), args.export_dir)
        print(f"Exported {n} samples to {args.export_dir}")
    elif not args.no_visualize:
        n = visualize_samples(decoder.iter_samples(max_samples=args.max_samples))
        print(f"Visualized {n} samples")
    else:
        count = 0
        for sample in decoder.iter_samples(max_samples=args.max_samples):
            print(f"{sample.sample_id}: image={sample.image.shape}, mask={sample.mask.shape}")
            count += 1
        print(f"Decoded {count} samples")
