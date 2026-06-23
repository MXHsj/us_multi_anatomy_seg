from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    convolve,
    distance_transform_edt,
)
from skimage import io
from skimage.draw import line
from skimage.morphology import skeletonize

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

    def _skeleton_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        skel = (skeleton > 0).astype(np.uint8)
        neighbor_count = convolve(
            skel,
            np.ones((3, 3), dtype=np.uint8),
            mode="constant",
            cval=0,
        ) - skel
        return np.argwhere((skel > 0) & (neighbor_count == 1))

    def _farthest_endpoint_pair(
        self,
        endpoints: np.ndarray,
    ) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
        if len(endpoints) < 2:
            return None
        points = endpoints.astype(np.float32)
        distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        return tuple(int(v) for v in endpoints[i]), tuple(int(v) for v in endpoints[j])

    def _snap_to_shadow_edge(
        self,
        point: tuple[int, int],
        shape: tuple[int, int],
        tolerance: int = 3,
    ) -> tuple[Optional[str], tuple[int, int]]:
        y, x = point
        h, w = shape
        candidates = [
            (abs(x), "left", (y, 0)),
            (abs(y - (h - 1)), "bottom", (h - 1, x)),
            (abs(x - (w - 1)), "right", (y, w - 1)),
        ]
        distance, edge, snapped = min(candidates, key=lambda item: item[0])
        if distance <= tolerance:
            return edge, snapped
        return None, point

    def _draw_same_shadow_edge_segment(
        self,
        closed: np.ndarray,
        edge: str,
        p0: tuple[int, int],
        p1: tuple[int, int],
    ) -> None:
        h, w = closed.shape
        y0, x0 = p0
        y1, x1 = p1
        if edge == "left":
            closed[min(y0, y1) : max(y0, y1) + 1, 0] = True
        elif edge == "right":
            closed[min(y0, y1) : max(y0, y1) + 1, w - 1] = True
        elif edge == "bottom":
            closed[h - 1, min(x0, x1) : max(x0, x1) + 1] = True

    def _draw_shadow_edge_path(
        self,
        closed: np.ndarray,
        edge0: str,
        q0: tuple[int, int],
        edge1: str,
        q1: tuple[int, int],
    ) -> None:
        h, w = closed.shape
        if edge0 == edge1:
            self._draw_same_shadow_edge_segment(closed, edge0, q0, q1)
            return

        anchors: list[int] = []
        for edge, (y, x) in [(edge0, q0), (edge1, q1)]:
            if edge == "left":
                closed[y:, 0] = True
                anchors.append(0)
            elif edge == "right":
                closed[y:, w - 1] = True
                anchors.append(w - 1)
            elif edge == "bottom":
                anchors.append(x)

        if len(anchors) >= 2:
            closed[h - 1, min(anchors) : max(anchors) + 1] = True

    def _connect_endpoints_or_shadow_edge(
        self,
        closed: np.ndarray,
        p0: tuple[int, int],
        p1: tuple[int, int],
        tolerance: int = 3,
    ) -> None:
        edge0, q0 = self._snap_to_shadow_edge(p0, closed.shape, tolerance=tolerance)
        edge1, q1 = self._snap_to_shadow_edge(p1, closed.shape, tolerance=tolerance)
        if edge0 is not None and edge1 is not None:
            self._draw_shadow_edge_path(closed, edge0, q0, edge1, q1)
            return

        y0, x0 = p0
        y1, x1 = p1
        rr, cc = line(y0, x0, y1, x1)
        closed[rr, cc] = True

    def _close_shadow_border_contacts(
        self,
        mask: np.ndarray,
        tolerance: int = 3,
    ) -> np.ndarray:
        closed = np.asarray(mask).astype(bool).copy()
        h, w = closed.shape
        ys, xs = np.where(closed)
        if ys.size == 0:
            return closed

        contacts: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        touches_left = xs <= tolerance
        touches_right = xs >= w - 1 - tolerance
        touches_bottom = ys >= h - 1 - tolerance

        if touches_left.any():
            left_ys = ys[touches_left]
            contacts.append(("left", (int(left_ys.min()), 0), (int(left_ys.max()), 0)))
        if touches_right.any():
            right_ys = ys[touches_right]
            contacts.append(("right", (int(right_ys.min()), w - 1), (int(right_ys.max()), w - 1)))
        if touches_bottom.any():
            bottom_xs = xs[touches_bottom]
            contacts.append(("bottom", (h - 1, int(bottom_xs.min())), (h - 1, int(bottom_xs.max()))))

        if len(contacts) == 1:
            edge, p0, p1 = contacts[0]
            self._draw_same_shadow_edge_segment(closed, edge, p0, p1)
            return closed

        anchors: list[int] = []
        for edge, p0, p1 in contacts:
            if edge == "left":
                closed[p0[0] :, 0] = True
                anchors.append(0)
            elif edge == "right":
                closed[p0[0] :, w - 1] = True
                anchors.append(w - 1)
            elif edge == "bottom":
                anchors.extend([p0[1], p1[1]])

        if len(anchors) >= 2:
            closed[h - 1, min(anchors) : max(anchors) + 1] = True
        return closed

    def _touches_shadow_edge(self, mask: np.ndarray, tolerance: int = 3) -> bool:
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return False
        h, w = mask.shape
        return bool(
            (xs <= tolerance).any()
            or (xs >= w - 1 - tolerance).any()
            or (ys >= h - 1 - tolerance).any()
        )

    def _fill_contour_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_bin = np.asarray(mask).astype(bool)
        if not mask_bin.any():
            return np.zeros_like(mask_bin, dtype=np.uint8)

        hole_filled = binary_fill_holes(mask_bin)
        eroded = binary_erosion(
            hole_filled,
            structure=np.ones((5, 5), dtype=bool),
            iterations=1,
        )

        raw_area = int(mask_bin.sum())
        filled_area = int(hole_filled.sum())
        if int(eroded.sum()) >= 10 and (
            not self._touches_shadow_edge(mask_bin)
            or filled_area >= max(raw_area * 2, raw_area + 10)
        ):
            return hole_filled.astype(np.uint8)

        skeleton = skeletonize(mask_bin)
        endpoints = self._skeleton_endpoints(skeleton)
        closed = mask_bin.copy()
        pair = self._farthest_endpoint_pair(endpoints)
        if pair is not None:
            self._connect_endpoints_or_shadow_edge(closed, pair[0], pair[1])

        filled = binary_fill_holes(closed)
        if int(filled.sum()) <= raw_area + 10:
            filled = binary_fill_holes(self._close_shadow_border_contacts(mask_bin))
        return filled.astype(np.uint8)

    def _prepare_mask(self, mask) -> np.ndarray:
        mask_bin = to_binary_mask(mask)
        if mask_bin.ndim == 3:
            mask_bin = mask_bin.max(axis=-1)
        if self.fill_mask:
            # UltraBones100k is the one decoder where filling is intentional:
            # the source labels trace the visible bone surface, while the filled
            # region approximates the clinically meaningful acoustic shadow.
            mask_bin = self._fill_contour_mask(mask_bin)

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
