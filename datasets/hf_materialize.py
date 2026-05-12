from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional


DEFAULT_HF_REPO_ID = "us-segmentator/us-segmentation-dataset"
DEFAULT_TNSC2020_REPO_DIR = "zips/Thyroid"
DEFAULT_BCU_PD_REPO_DIR = "zips/Breast"


def _require_huggingface_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required to download datasets on demand. "
            "Install it with 'pip install huggingface_hub'."
        ) from exc
    return HfApi, hf_hub_download


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise ValueError(f"Refusing unsafe zip member path: {member.filename}")
        zf.extractall(out_dir)


def _normalize_name_for_match(value: str) -> str:
    return value.lower().replace("_", "").replace("-", "").replace(" ", "")


def _find_tnsc2020_root(extracted_dir: Path) -> Path:
    candidates = [
        path
        for path in extracted_dir.rglob("*")
        if path.is_dir() and (path / "image").is_dir() and (path / "mask").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not find a TNSC2020-style directory with 'image' and 'mask' subdirectories "
            f"inside extracted archive: {extracted_dir}"
        )
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def _discover_zip_path(
    repo_id: str,
    repo_dir: str,
    revision: str,
    name_hint: str | None = None,
) -> str:
    HfApi, _ = _require_huggingface_hub()
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    prefix = repo_dir.rstrip("/") + "/"
    matches = [
        path
        for path in files
        if path.startswith(prefix) and path.lower().endswith(".zip")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No zip files found in hf://datasets/{repo_id}/{repo_dir} at revision '{revision}'."
        )
    if name_hint:
        normalized_hint = _normalize_name_for_match(name_hint)
        hinted = [
            path
            for path in matches
            if normalized_hint in _normalize_name_for_match(Path(path).name)
        ]
        if len(hinted) == 1:
            return hinted[0]
        if len(hinted) > 1:
            raise ValueError(
                f"Multiple zip files matching '{name_hint}' were found. Pass an explicit repo path "
                f"with --hf-repo-path. Candidates: {hinted}"
            )
    tnsc_matches = [path for path in matches if "tnsc" in Path(path).name.lower()]
    if len(tnsc_matches) == 1:
        return tnsc_matches[0]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        "Multiple Thyroid zip files were found. Pass an explicit repo path with "
        f"--hf-repo-path. Candidates: {matches}"
    )


def _materialize_zip_dataset(
    output_dir: str | Path,
    repo_id: str,
    repo_path: Optional[str],
    repo_dir: str,
    revision: str,
    force: bool,
    name_hint: str | None = None,
    root_finder=None,
) -> Path:
    output_dir = Path(output_dir)
    if not force and output_dir.exists() and any(output_dir.iterdir()):
        return output_dir

    _, hf_hub_download = _require_huggingface_hub()
    zip_repo_path = repo_path or _discover_zip_path(
        repo_id=repo_id,
        repo_dir=repo_dir,
        revision=revision,
        name_hint=name_hint,
    )
    local_zip = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=zip_repo_path,
            revision=revision,
        )
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"{name_hint or 'dataset'}_", dir=str(output_dir.parent)) as tmp:
        extracted = Path(tmp) / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(local_zip, extracted)
        source_root = root_finder(extracted) if root_finder else extracted

        staging = Path(tmp) / "staging"
        shutil.copytree(source_root, staging)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staging), output_dir)

    return output_dir


def materialize_tnsc2020_from_hf(
    output_dir: str | Path = "datasets/TNSC2020",
    repo_id: str = DEFAULT_HF_REPO_ID,
    repo_path: Optional[str] = None,
    repo_dir: str = DEFAULT_TNSC2020_REPO_DIR,
    revision: str = "main",
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    if (
        not force
        and (output_dir / "image").is_dir()
        and (output_dir / "mask").is_dir()
    ):
        return output_dir
    return _materialize_zip_dataset(
        output_dir=output_dir,
        repo_id=repo_id,
        repo_path=repo_path,
        repo_dir=repo_dir,
        revision=revision,
        force=force,
        name_hint="tnsc",
        root_finder=_find_tnsc2020_root,
    )


def materialize_bcu_pd_from_hf(
    output_dir: str | Path = "datasets/BCU_PD",
    repo_id: str = DEFAULT_HF_REPO_ID,
    repo_path: Optional[str] = None,
    repo_dir: str = DEFAULT_BCU_PD_REPO_DIR,
    revision: str = "main",
    force: bool = False,
) -> Path:
    return _materialize_zip_dataset(
        output_dir=output_dir,
        repo_id=repo_id,
        repo_path=repo_path,
        repo_dir=repo_dir,
        revision=revision,
        force=force,
        name_hint="bcu_pd",
    )


def ensure_tnsc2020_dataset(
    root: str | Path = "datasets/TNSC2020",
    repo_id: str = DEFAULT_HF_REPO_ID,
    repo_path: Optional[str] = None,
    revision: str = "main",
) -> Path:
    root = Path(root)
    if (root / "image").is_dir() and (root / "mask").is_dir():
        return root
    return materialize_tnsc2020_from_hf(
        output_dir=root,
        repo_id=repo_id,
        repo_path=repo_path,
        revision=revision,
    )


def ensure_bcu_pd_dataset(
    root: str | Path = "datasets/BCU_PD",
    repo_id: str = DEFAULT_HF_REPO_ID,
    repo_path: Optional[str] = None,
    revision: str = "main",
) -> Path:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        return root
    return materialize_bcu_pd_from_hf(
        output_dir=root,
        repo_id=repo_id,
        repo_path=repo_path,
        revision=revision,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and extract TNSC2020 from Hugging Face")
    parser.add_argument("--output-dir", default="datasets/TNSC2020")
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument(
        "--repo-path",
        default="",
        help="Explicit zip path in the dataset repo, e.g. zips/Thyroid/TNSC2020.zip",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    path = materialize_tnsc2020_from_hf(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        repo_path=args.repo_path or None,
        revision=args.revision,
        force=args.force,
    )
    print(f"TNSC2020 materialized at: {path}")
