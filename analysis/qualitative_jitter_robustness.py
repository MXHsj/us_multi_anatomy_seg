from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "analysis" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "analysis" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.qualitative_results_analysis import (
    _draw_bboxes,
    _draw_gt_outline,
    _plot_result_pair,
    load_dataset_labels,
    load_decoder,
    load_results,
    load_samples,
    split_csv,
    str2bool,
)


DEFAULT_DATASETS = (
    "aulid",
    "blusg",
    "busbra",
    "busi",
    "camus",
    "oku",
    "mmotu",
    "gist514",
    "roblus",
    "tnsc2020",
    "ultrabones100k",
    "umud",
    "uns",
)
DEFAULT_SCALE_VALUES = ("0.9", "1", "1.10")
DEFAULT_VIS_TRANSLATION_VALUES = ("0", "0.5", "1.0")
DEFAULT_SEED = 812528458
DEFAULT_QUAL_CONFIG = ROOT / "analysis" / "figures" / "qualitative" / "dice_0_5" / "run_config.json"


def optional_text(value: str) -> str | None:
    value = value.strip()
    return value or None


def jitter_metrics_path(
    jitter_results_dir: Path,
    dataset: str,
    experiment: str,
    value_text: str,
) -> Path:
    if experiment == "scale":
        return jitter_results_dir / "scale" / f"ultrasam_scale_{value_text}_bbox_{dataset}" / "per_sample_metrics.csv"
    if experiment == "translation":
        return jitter_results_dir / "trans" / f"ultrasam_trans_{value_text}_bbox_{dataset}" / "per_sample_metrics.csv"
    raise ValueError(f"Unsupported experiment: {experiment}")


def read_dice_by_sample(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype={"sample_id": str})
    if "sample_id" not in frame or "dice" not in frame:
        raise ValueError(f"Expected sample_id and dice columns in {path}")
    values = pd.to_numeric(frame["dice"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=frame["sample_id"].astype(str))
    return series.dropna()


def discover_translation_values(jitter_results_dir: Path, dataset: str) -> tuple[str, ...]:
    trans_dir = jitter_results_dir / "trans"
    prefix = "ultrasam_trans_"
    suffix = f"_bbox_{dataset}"
    values: list[tuple[float, str]] = []
    for path in trans_dir.glob(f"{prefix}*{suffix}"):
        if not path.is_dir():
            continue
        value_text = path.name[len(prefix) : -len(suffix)]
        try:
            numeric = float(value_text)
        except ValueError:
            continue
        values.append((numeric, value_text))
    return tuple(value_text for _numeric, value_text in sorted(values))


def select_robustness_sample(
    *,
    jitter_results_dir: Path,
    datasets: tuple[str, ...],
    scale_values: tuple[str, ...],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for dataset in datasets:
        scale_series: dict[str, pd.Series] = {}
        missing = False
        for value_text in scale_values:
            path = jitter_metrics_path(jitter_results_dir, dataset, "scale", value_text)
            if not path.exists():
                missing = True
                break
            scale_series[value_text] = read_dice_by_sample(path)
        if missing:
            continue

        translation_values = discover_translation_values(jitter_results_dir, dataset)
        if len(translation_values) < 2:
            continue
        translation_series: dict[str, pd.Series] = {}
        for value_text in translation_values:
            path = jitter_metrics_path(jitter_results_dir, dataset, "translation", value_text)
            if path.exists():
                translation_series[value_text] = read_dice_by_sample(path)
        if len(translation_series) < 2:
            continue

        shared_ids = set.intersection(
            *(set(series.index.astype(str)) for series in (*scale_series.values(), *translation_series.values()))
        )
        for sample_id in sorted(shared_ids):
            scale_dice = {value: float(series.loc[sample_id]) for value, series in scale_series.items()}
            translation_dice = {value: float(series.loc[sample_id]) for value, series in translation_series.items()}
            scale_range = max(scale_dice.values()) - min(scale_dice.values())
            translation_range = max(translation_dice.values()) - min(translation_dice.values())
            score = scale_range + translation_range
            if score <= 0:
                continue
            candidates.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "score": score,
                    "scale_range": scale_range,
                    "translation_range": translation_range,
                    "scale_dice": scale_dice,
                    "translation_dice": translation_dice,
                }
            )

    if not candidates:
        raise SystemExit("No sample had valid scale and translation jitter Dice results.")

    candidates.sort(
        key=lambda item: (
            float(item["score"]),
            float(item["translation_range"]),
            float(item["scale_range"]),
            str(item["dataset"]),
            str(item["sample_id"]),
        ),
        reverse=True,
    )
    selected = candidates[0]
    selected["num_candidates"] = len(candidates)
    return selected


def load_summary_for_dataset(dataset: str, results_dir: Path, fallback_dataset_root: str = "") -> dict[str, Any]:
    try:
        _df, summary = load_results("ultrasam", dataset, "gt_bbox", results_dir)
    except FileNotFoundError:
        summary = {}
    if fallback_dataset_root and not summary.get("dataset_root"):
        summary["dataset_root"] = fallback_dataset_root
    return summary


def run_ultrasam_condition(
    *,
    sample: Any,
    summary: dict[str, Any],
    scale: float,
    translation: float,
    seed: int,
    device: str,
) -> Any:
    from benchmarks import ultrasam_inference as ultrasam

    return ultrasam.run_ultrasam_on_samples(
        [sample],
        device=device,
        batch_size=int(summary.get("batch_size") or 4),
        num_workers=0,
        checkpoint=summary.get("checkpoint") or "UltraSam/weights/UltraSam.pth",
        checkpoint_url=summary.get("checkpoint_url", ultrasam.DEFAULT_ULTRASAM_CHECKPOINT_URL),
        no_auto_download_checkpoint=True,
        ultrasam_dir=summary.get("ultrasam_dir") or "UltraSam",
        config=summary.get("config") or None,
        prompt_type="bbox",
        bbox_scale_factor=scale,
        bbox_translation_fraction=translation,
        point_prompt_jitter_fraction=0.0,
        bbox_mode=summary.get("bbox_mode") or "individual",
        seed=seed,
    )[0]


def result_metrics(result: Any) -> dict[str, float]:
    metrics = dict(result.metrics)
    metrics["infer_ms"] = float(result.infer_ms)
    return metrics


def gist_sample_from_config(config_path: Path, example_index: int) -> str:
    config = json.loads(config_path.read_text())
    sample_ids = config.get("selected_sample_ids", {}).get("gist514", [])
    if len(sample_ids) < example_index:
        raise SystemExit(
            f"{config_path} does not contain GIST514 example {example_index}; "
            f"found {len(sample_ids)} sample ids."
        )
    return str(sample_ids[example_index - 1])


def plot_robustness_figure(
    columns: list[dict[str, Any]],
    save_path: Path,
    *,
    pred_color: str,
    pred_alpha: float,
    show_image_background: bool,
    reference_bbox_color: str,
) -> None:
    bbox_linewidth = 1.35
    reference_bbox_linewidth = 1.25
    gt_outline_width = 1.25
    width_ratios = [1.0, 1.0, 1.0, 0.06, 1.0, 1.0, 1.0, 0.06, 1.0]
    plot_columns = [0, 1, 2, 4, 5, 6, 8]
    fig, axes = plt.subplots(
        2,
        len(width_ratios),
        figsize=(3.05 * len(columns), 5.6),
        gridspec_kw={"height_ratios": [2.85, 3.15], "width_ratios": width_ratios},
        squeeze=False,
    )
    for spacer_col in (3, 7):
        axes[0, spacer_col].remove()
        axes[1, spacer_col].remove()

    for col, column in enumerate(columns):
        axis_col = plot_columns[col]
        _plot_result_pair(
            axes[0, axis_col],
            axes[1, axis_col],
            column["result"],
            column["metrics"],
            dataset=column["dataset_label"],
            pred_color=pred_color,
            pred_alpha=pred_alpha,
            show_image_background=show_image_background,
            gt_outline_color="white",
            bbox_color="yellow",
        )
        _draw_bboxes(axes[0, axis_col], column["result"].bbox, "yellow", bbox_linewidth)
        _draw_gt_outline(axes[1, axis_col], column["result"].gt_mask, "white", gt_outline_width)
        reference_bbox = column.get("reference_bbox")
        if reference_bbox is not None:
            _draw_bboxes(axes[0, axis_col], reference_bbox, reference_bbox_color, reference_bbox_linewidth)
        axes[0, axis_col].set_title(column["title"], fontsize=16, pad=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.91, bottom=0.01)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a qualitative UltraSAM jitter robustness figure.")
    parser.add_argument("--jitter-results-dir", type=Path, default=ROOT / "experiments" / "prompt_robustness")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "figures" / "prompt_robustness")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--scale-values", default=",".join(DEFAULT_SCALE_VALUES))
    parser.add_argument("--visual-translation-values", default=",".join(DEFAULT_VIS_TRANSLATION_VALUES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gist-config", type=Path, default=DEFAULT_QUAL_CONFIG)
    parser.add_argument("--gist-example-index", type=int, default=15)
    parser.add_argument(
        "--gist-sample-id",
        default="",
        help="Optional explicit GIST514 sample_id for the final shape-characteristic column.",
    )
    parser.add_argument("--pred-color", default="#ff4a43")
    parser.add_argument("--pred-alpha", type=float, default=0.45)
    parser.add_argument("--reference-bbox-color", default="#00d7ff")
    parser.add_argument("--show-image-background", type=str2bool, default=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-id", type=optional_text, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = tuple(split_csv(args.datasets))
    scale_value_texts = tuple(split_csv(args.scale_values))
    scale_values = tuple(float(value) for value in scale_value_texts)
    visual_translation_values = tuple(float(value) for value in split_csv(args.visual_translation_values))
    device = args.device or (f"cuda:{args.gpu_id}" if args.gpu_id else "cuda:0")
    dataset_labels = load_dataset_labels()

    selected = select_robustness_sample(
        jitter_results_dir=args.jitter_results_dir,
        datasets=datasets,
        scale_values=scale_value_texts,
    )
    print(
        "Selected robustness sample: "
        f"{selected['dataset']} {selected['sample_id']} "
        f"(score={selected['score']:.4f}, scale_range={selected['scale_range']:.4f}, "
        f"translation_range={selected['translation_range']:.4f})"
    )

    selected_summary = load_summary_for_dataset(selected["dataset"], args.results_dir)
    selected_decoder = load_decoder(selected["dataset"], selected_summary.get("dataset_root", ""))
    selected_sample = load_samples(selected_decoder, [selected["sample_id"]])[0]
    reference_result = run_ultrasam_condition(
        sample=selected_sample,
        summary=selected_summary,
        scale=1.0,
        translation=0.0,
        seed=args.seed,
        device=device,
    )
    reference_bbox = reference_result.bbox

    columns: list[dict[str, Any]] = []
    visual_metrics: dict[str, dict[str, float]] = {}
    selected_dataset_label = dataset_labels.get(selected["dataset"], str(selected["dataset"]).upper())

    for value in scale_values:
        result = run_ultrasam_condition(
            sample=selected_sample,
            summary=selected_summary,
            scale=value,
            translation=0.0,
            seed=args.seed,
            device=device,
        )
        key = f"scale_{value:g}"
        metrics = result_metrics(result)
        visual_metrics[key] = metrics
        columns.append(
            {
                "title": f"Scale {value:g}",
                "dataset_label": selected_dataset_label,
                "result": result,
                "metrics": metrics,
                "reference_bbox": reference_bbox if not math.isclose(value, 1.0) else None,
            }
        )

    for value in visual_translation_values:
        result = run_ultrasam_condition(
            sample=selected_sample,
            summary=selected_summary,
            scale=1.0,
            translation=value,
            seed=args.seed,
            device=device,
        )
        key = f"translation_{value:g}"
        metrics = result_metrics(result)
        visual_metrics[key] = metrics
        columns.append(
            {
                "title": f"Translation {value:g}",
                "dataset_label": selected_dataset_label,
                "result": result,
                "metrics": metrics,
                "reference_bbox": reference_bbox if not math.isclose(value, 0.0) else None,
            }
        )

    gist_sample_id = args.gist_sample_id.strip() or gist_sample_from_config(
        args.gist_config,
        args.gist_example_index,
    )
    gist_summary = load_summary_for_dataset("gist514", args.results_dir, "datasets/GIST514")
    gist_decoder = load_decoder("gist514", gist_summary.get("dataset_root", "datasets/GIST514"))
    gist_sample = load_samples(gist_decoder, [gist_sample_id])[0]
    gist_result = run_ultrasam_condition(
        sample=gist_sample,
        summary=gist_summary,
        scale=1.0,
        translation=0.0,
        seed=args.seed,
        device=device,
    )
    gist_metrics = result_metrics(gist_result)
    columns.append(
        {
            "title": "Crescent-like object",
            "dataset_label": dataset_labels.get("gist514", "GIST514"),
            "result": gist_result,
            "metrics": gist_metrics,
        }
    )

    output_stem = "qualitative_prompt_robustness"
    output_path = args.output_dir / f"{output_stem}.png"
    plot_robustness_figure(
        columns,
        output_path,
        pred_color=args.pred_color,
        pred_alpha=args.pred_alpha,
        show_image_background=args.show_image_background,
        reference_bbox_color=args.reference_bbox_color,
    )

    sidecar = {
        "output_path": str(output_path),
        "seed": args.seed,
        "selected_sample": selected,
        "visual_conditions": {
            "scale": list(scale_values),
            "translation": list(visual_translation_values),
        },
        "visual_metrics": visual_metrics,
        "reference_bbox_color": args.reference_bbox_color,
        "gist514": {
            "config": str(args.gist_config),
            "example_index": args.gist_example_index,
            "sample_id_source": "explicit" if args.gist_sample_id.strip() else "run_config",
            "sample_id": gist_sample_id,
            "metrics": gist_metrics,
        },
    }
    sidecar_path = args.output_dir / f"{output_stem}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote figure to {output_path}")
    print(f"Wrote metadata to {sidecar_path}")


if __name__ == "__main__":
    main()
