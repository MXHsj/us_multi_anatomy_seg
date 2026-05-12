from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from skimage import io

try:
    from datasets.common import DecodedSample, export_samples, normalize_to_uint8
    from datasets.hf_materialize import DEFAULT_HF_REPO_ID, ensure_bcu_pd_dataset
except ModuleNotFoundError:
    from common import DecodedSample, export_samples, normalize_to_uint8
    from hf_materialize import DEFAULT_HF_REPO_ID, ensure_bcu_pd_dataset


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_MASK_HINTS = {"mask", "masks", "label", "labels", "gt", "ground_truth", "groundtruth"}
_MASK_SUFFIXES = ("_mask", "-mask", "_label", "-label", "_gt", "-gt")


def _has_mask_hint(path: Path) -> bool:
    return any(part.lower() in _MASK_HINTS for part in path.parts)


def _mask_to_binary(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        return np.any(arr[..., :3] > 0, axis=-1).astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def _normalized_stem(path: Path) -> str:
    stem = path.stem
    stem_lower = stem.lower()
    for suffix in _MASK_SUFFIXES:
        if stem_lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _image_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_0000"):
        return stem[:-5]
    return stem


class BreastBCUPDDecoder:
    def __init__(
        self,
        root: str | Path = "datasets/BCU_PD",
        auto_download: bool = True,
        hf_repo_id: str = DEFAULT_HF_REPO_ID,
        hf_repo_path: Optional[str] = None,
        hf_revision: str = "main",
    ):
        self.root = Path(root)
        if auto_download:
            self.root = ensure_bcu_pd_dataset(
                root=self.root,
                repo_id=hf_repo_id,
                repo_path=hf_repo_path,
                revision=hf_revision,
            )
        if not self.root.exists():
            raise FileNotFoundError(f"BCU_PD root '{self.root}' not found.")

        self._pairs = self._build_pairs()
        if not self._pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found under '{self.root}'. Expected image files and "
                "mask files in separate mask/label/gt folders or with mask-like filename suffixes."
            )

    def _image_paths(self) -> list[Path]:
        paths = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        return sorted(p for p in paths if not _has_mask_hint(p) and _normalized_stem(p) == p.stem)

    def _mask_paths(self) -> list[Path]:
        paths = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        return sorted(p for p in paths if _has_mask_hint(p) or _normalized_stem(p) != p.stem)

    def _build_pairs(self) -> list[tuple[Path, Path]]:
        masks_by_stem: dict[str, list[Path]] = {}
        for mask_path in self._mask_paths():
            masks_by_stem.setdefault(_normalized_stem(mask_path), []).append(mask_path)

        pairs: list[tuple[Path, Path]] = []
        for image_path in self._image_paths():
            candidates = masks_by_stem.get(_image_key(image_path), [])
            if candidates:
                pairs.append((image_path, sorted(candidates)[0]))
        return pairs

    def iter_samples(self, max_samples: Optional[int] = None) -> Iterator[DecodedSample]:
        count = 0
        for image_path, mask_path in self._pairs:
            image = io.imread(image_path)
            mask = _mask_to_binary(io.imread(mask_path))
            if mask.sum() == 0:
                continue

            sample_id = image_path.relative_to(self.root).with_suffix("").as_posix()
            yield DecodedSample(
                dataset="BCU_PD",
                sample_id=sample_id,
                image=normalize_to_uint8(image),
                mask=mask,
                metadata={
                    "raw_image_path": str(image_path),
                    "raw_mask_path": str(mask_path),
                },
            )

            count += 1
            if max_samples is not None and count >= max_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode BCU_PD breast ultrasound dataset")
    parser.add_argument("--root", type=str, default="datasets/BCU_PD")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--export-dir", type=str, default="")
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default=DEFAULT_HF_REPO_ID)
    parser.add_argument(
        "--hf-repo-path",
        type=str,
        default="",
        help="Explicit zip path inside the Hugging Face repo, e.g. zips/Breast/BCU_PD.zip.",
    )
    parser.add_argument("--hf-revision", type=str, default="main")
    args = parser.parse_args()

    decoder = BreastBCUPDDecoder(
        args.root,
        auto_download=not args.no_auto_download,
        hf_repo_id=args.hf_repo_id,
        hf_repo_path=args.hf_repo_path or None,
        hf_revision=args.hf_revision,
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
