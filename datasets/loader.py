from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

from datasets.hf_materialize import DEFAULT_HF_REPO_ID, ensure_dataset_available
from datasets.registry import DATASET_REGISTRY


def dataset_choices() -> list[str]:
    return sorted(DATASET_REGISTRY)


def default_dataset_root(dataset_name: str) -> str:
    return DATASET_REGISTRY[dataset_name.lower()].default_root


def _load_class(class_path: str):
    module_name, class_name = class_path.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, class_name)


def _decoder_kwargs(dataset_name: str, args: argparse.Namespace) -> dict[str, Any]:
    if dataset_name == "camus":
        return {
            "include_half_sequence": args.include_half_sequence,
            "positive_labels": tuple(int(x) for x in args.camus_labels.split(",") if x.strip()),
        }
    if dataset_name == "blusg":
        return {"include_other": not args.blusg_only_tumor}
    if dataset_name == "busbra":
        pathology = [p.strip() for p in args.busbra_pathology.split(",") if p.strip()]
        birads = [int(b) for b in args.busbra_birads.split(",") if b.strip()]
        return {
            "pathology_filter": pathology or None,
            "birads_filter": birads or None,
        }
    if dataset_name == "busi":
        categories = [c.strip() for c in args.busi_categories.split(",") if c.strip()]
        return {"categories": categories or None}
    if dataset_name == "oku":
        anatomy_filter = [a.strip() for a in args.oku_anatomy.split(",") if a.strip()]
        return {
            "anatomy_filter": anatomy_filter,
            "prefer_labels_2": not args.oku_prefer_labels_1,
        }
    if dataset_name == "aulid":
        return {"label": args.aulid_label}
    if dataset_name == "roblus":
        labels = [label.strip() for label in args.roblus_labels.split(",") if label.strip()]
        subjects = [subject.strip() for subject in args.roblus_subjects.split(",") if subject.strip()]
        return {
            "labels": labels,
            "subjects": subjects or None,
        }
    if dataset_name == "ussc":
        labels = [label.strip() for label in args.ussc_labels.split(",") if label.strip()]
        splits = [split.strip() for split in args.ussc_splits.split(",") if split.strip()]
        return {"labels": labels or None, "splits": splits or None}
    if dataset_name == "ultrabones100k":
        return {"frame_fraction": args.ultrabones_frame_fraction}
    return {}


def build_decoder_from_args(args: argparse.Namespace):
    dataset_name = args.dataset.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    root = args.dataset_root or default_dataset_root(dataset_name)
    local_root = ensure_dataset_available(
        dataset_name=dataset_name,
        root=root,
        auto_download=not args.no_auto_download,
        repo_id=args.hf_repo_id,
        repo_path=args.hf_repo_path or None,
        revision=args.hf_revision,
    )

    decoder_cls = _load_class(DATASET_REGISTRY[dataset_name].decoder_class)
    return decoder_cls(root=Path(local_root), **_decoder_kwargs(dataset_name, args))


def add_dataset_args(parser: argparse.ArgumentParser, include_camus: bool = True) -> None:
    choices = dataset_choices()
    if not include_camus:
        choices = [choice for choice in choices if choice != "camus"]
    parser.add_argument("--dataset", type=str, default="tnsc2020", choices=choices)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Local dataset cache root. Defaults to the registered root for --dataset.",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Require local dataset files instead of downloading missing Hugging Face-backed datasets.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help="Hugging Face dataset repo used by on-demand dataset loaders.",
    )
    parser.add_argument(
        "--hf-repo-path",
        type=str,
        default="",
        help="Explicit zip path inside the Hugging Face repo for the selected dataset.",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default="main",
        help="Hugging Face repo revision used by on-demand dataset loaders.",
    )
    if include_camus:
        parser.add_argument("--include-half-sequence", action="store_true")
        parser.add_argument(
            "--camus-labels",
            type=str,
            default="1,3",
            help="Comma-separated CAMUS label ids to evaluate as separate binary targets.",
        )
    parser.add_argument(
        "--blusg-only-tumor",
        action="store_true",
        help="Use only tumor masks for BLUSG (ignore other lesion masks).",
    )
    parser.add_argument(
        "--busbra-pathology",
        type=str,
        default="",
        help="Comma-separated BUS-BRA pathology filter (benign, malignant).",
    )
    parser.add_argument(
        "--busbra-birads",
        type=str,
        default="",
        help="Comma-separated BUS-BRA BI-RADS filter (e.g. 4,5).",
    )
    parser.add_argument(
        "--busi-categories",
        type=str,
        default="benign,malignant",
        help="Comma-separated BUSI categories to decode (benign, malignant, normal).",
    )
    parser.add_argument(
        "--oku-anatomy",
        type=str,
        default="Capsule",
        help="Comma-separated anatomy filter for OKU (e.g., Capsule,Cortex).",
    )
    parser.add_argument(
        "--oku-prefer-labels-1",
        action="store_true",
        help="Prefer reviewed_labels_1.csv over reviewed_labels_2.csv.",
    )
    parser.add_argument(
        "--aulid-label",
        type=str,
        default="mass",
        choices=["mass", "liver", "outline"],
        help="AULID segmentation label used as the benchmark mask.",
    )
    parser.add_argument(
        "--roblus-labels",
        type=str,
        default="pleural_line",
        help="Comma-separated RobLUS labels to merge into the benchmark mask.",
    )
    parser.add_argument(
        "--roblus-subjects",
        type=str,
        default="",
        help="Optional comma-separated RobLUS subject filter, e.g. AP,BM.",
    )
    parser.add_argument(
        "--ussc-labels",
        type=str,
        default="",
        help=(
            "Comma-separated USSC semantic labels. Defaults to all non-background "
            "classes: dura,csf,pia,spinal_cord,dorsal_space,hematoma,"
            "dura_pia_complex,dura_ventral_complex,ventral_space."
        ),
    )
    parser.add_argument(
        "--ussc-splits",
        type=str,
        default="",
        help="Comma-separated USSC splits to decode (train,val,test). Defaults to all splits.",
    )
    parser.add_argument(
        "--ultrabones-frame-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of equally-spaced frames to sample within each UltraBones "
            "record/video (e.g. 0.05 for ~5%%, at least one frame per record). "
            "Defaults to 1.0 (all frames). This stratifies across specimen, "
            "structure, and record."
        ),
    )
