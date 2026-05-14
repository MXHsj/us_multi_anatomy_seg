from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
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
    ):
        self.root = Path(root)
        self.label_folder = label_folder
        self.thicken_radius = thicken_radius
        self.fill_mask = fill_mask

        if not self.root.exists():
            raise FileNotFoundError(f"UltraBones100k root '{self.root}' not found.")

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

    def _prepare_mask(self, mask) -> np.ndarray:
        mask_bin = to_binary_mask(mask)
        if self.fill_mask:
            mask_bin = binary_fill_holes(mask_bin.astype(bool)).astype(np.uint8)

        if self.thicken_radius <= 0:
            return mask_bin

        # Optional distance-transform expansion for thin surface labels.
        distance_map = distance_transform_edt(~mask_bin.astype(bool))
        return (distance_map <= self.thicken_radius).astype(np.uint8)

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        count = 0
        for record_dir in self._record_dirs():
            image_dir = record_dir / "UltrasoundImages"
            label_dir = record_dir / self.label_folder
            if not label_dir.exists():
                continue

            for image_path in sorted(image_dir.glob("*.png")):
                label_path = label_dir / f"{image_path.stem}_label.png"
                if not label_path.exists():
                    continue
                count += 1
                if max_samples is not None and count >= max_samples:
                    return count
        return count

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for record_dir in self._record_dirs():
            image_dir = record_dir / "UltrasoundImages"
            label_dir = record_dir / self.label_folder
            if not label_dir.exists():
                continue

            record_metadata = self._record_metadata(record_dir)
            tracking = self._tracking_map(record_dir)

            for image_path in sorted(image_dir.glob("*.png")):
                # UltraBones pairs 24363.png with 24363_label.png.
                label_path = label_dir / f"{image_path.stem}_label.png"
                if not label_path.exists():
                    continue

                image = io.imread(image_path)
                mask = io.imread(label_path)
                sample_id = (
                    f"{record_metadata['specimen_id']}_{record_metadata['anatomy']}_"
                    f"{record_metadata['record']}_{image_path.stem}"
                )

                metadata = {
                    **record_metadata,
                    "timestamp": image_path.stem,
                    "raw_image_path": str(image_path),
                    "raw_mask_path": str(label_path),
                    "label_folder": self.label_folder,
                    "thicken_radius": self.thicken_radius,
                    "fill_mask": self.fill_mask,
                }
                if image_path.stem in tracking:
                    # Keep the synchronized probe pose/tracking row available downstream.
                    metadata["tracking"] = tracking[image_path.stem]

                yield DecodedSample(
                    dataset="UltraBones100k",
                    sample_id=sample_id,
                    image=normalize_to_uint8(image),
                    mask=self._prepare_mask(mask),
                    metadata=metadata,
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
        help="Do not fill internal regions in the binary label mask",
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
