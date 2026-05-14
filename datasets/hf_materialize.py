from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.registry import DATASET_REGISTRY


DEFAULT_HF_REPO_ID = "us-segmentator/us-segmentation-dataset"


def _require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required to download datasets on demand. "
            "Install it with 'python -m pip install -r requirements.txt'."
        ) from exc
    return hf_hub_download


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zip_file:
        for member in zip_file.infolist():
            target = (out_dir / member.filename).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise ValueError(f"Refusing unsafe zip member path: {member.filename}")
        zip_file.extractall(out_dir)


def _is_noise_path(path: Path) -> bool:
    return any(part == "__MACOSX" for part in path.parts) or path.name in {".DS_Store"}


def _archive_payload_root(extracted_dir: Path) -> Path:
    children = [path for path in extracted_dir.iterdir() if not _is_noise_path(path)]
    dirs = [path for path in children if path.is_dir()]
    files = [path for path in children if path.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return extracted_dir


def materialize_dataset(
    dataset_name: str,
    output_dir: str | Path | None = None,
    repo_id: str = DEFAULT_HF_REPO_ID,
    repo_path: str | None = None,
    revision: str = "main",
    force: bool = False,
) -> Path:
    key = dataset_name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{dataset_name}'.")

    spec = DATASET_REGISTRY[key]
    output_path = Path(output_dir or spec.default_root)
    if not force and output_path.exists() and any(output_path.iterdir()):
        return output_path

    zip_repo_path = repo_path or spec.hf_repo_path
    if not zip_repo_path:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' is not registered as Hugging Face-backed. "
            f"Expected local files at: {output_path}"
        )

    hf_hub_download = _require_huggingface_hub()
    local_zip = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=zip_repo_path,
            revision=revision,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"{key}_", dir=str(output_path.parent)) as tmp:
        extracted_dir = Path(tmp) / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(local_zip, extracted_dir)
        payload_root = _archive_payload_root(extracted_dir)

        staging_dir = Path(tmp) / "staging"
        shutil.copytree(payload_root, staging_dir, ignore=shutil.ignore_patterns("__MACOSX", ".DS_Store"))
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(staging_dir), output_path)

    return output_path


def ensure_dataset_available(
    dataset_name: str,
    root: str | Path | None = None,
    auto_download: bool = True,
    repo_id: str = DEFAULT_HF_REPO_ID,
    repo_path: str | None = None,
    revision: str = "main",
) -> Path:
    key = dataset_name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{dataset_name}'.")

    local_root = Path(root or DATASET_REGISTRY[key].default_root)
    if local_root.exists() and any(local_root.iterdir()):
        return local_root
    if not auto_download:
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at: {local_root}")
    return materialize_dataset(
        dataset_name=key,
        output_dir=local_root,
        repo_id=repo_id,
        repo_path=repo_path,
        revision=revision,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Materialize a registered Hugging Face dataset zip.")
    parser.add_argument("dataset", choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--repo-path", default="")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    path = materialize_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir or None,
        repo_id=args.repo_id,
        repo_path=args.repo_path or None,
        revision=args.revision,
        force=args.force,
    )
    print(f"{args.dataset} materialized at: {path}")


if __name__ == "__main__":
    main()
