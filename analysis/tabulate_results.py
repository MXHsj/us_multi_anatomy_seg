from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.model_across_datasets import (  # noqa: E402
    DATASET_LABELS,
    DEFAULT_METRICS,
    format_float,
    iter_run_dirs,
    metric_summary_columns,
    parse_metrics_arg,
)


DEFAULT_DATASETS = (
    "aulid",
    "blusg",
    "busbra",
    "busi",
    "camus",
    "oku",
    "roblus",
    "tnsc2020",
    "ultrabones100k",
    "uns",
)
DEFAULT_FORMATS = ("markdown", "csv", "latex")
MODEL_LABELS = {
    "medicalsam3": "Medical SAM3",
    "medsam": "MedSAM",
    "medsam3": "MedSAM-3",
    "samus": "SAMUS",
    "ultrasam": "UltraSam",
}
MODEL_CITATIONS = {
    "Medical SAM3": "jiang2026medicalsam3",
    "MedSAM": "ma2024medsam",
    "MedSAM-3": "liu2025medsam3",
    "SAMUS": "lin2024samus",
    "UltraSam": "meyer2025ultrasam",
}
PROMPT_LABELS = {
    "gt_bbox": "Bbox",
    "gt_point": "Point",
    "label": "Text (label)",
    "object": "Text (object)",
    "text+bbox": "Text + Bbox",
}
LOWER_IS_BETTER = {"hd95", "assd", "relative_area_error"}
LATEX_CAPTION = (
    "Benchmark results by model and prompt. Dice values are reported as percentages. "
    "Best values in each metric column are bolded and second-best values are underlined."
)


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got: {value}")


def parse_formats_arg(value: str) -> tuple[str, ...]:
    formats = parse_csv_arg(value)
    allowed = set(DEFAULT_FORMATS)
    unknown = sorted(set(formats) - allowed)
    if unknown:
        raise SystemExit(
            f"Unsupported format(s): {', '.join(unknown)}. "
            f"Expected one of: {', '.join(DEFAULT_FORMATS)}"
        )
    if not formats:
        raise SystemExit("At least one output format must be selected.")
    return formats


def dataset_label(dataset: str) -> str:
    if dataset == "tnsc2020":
        return "TNSC"
    if dataset == "ultrabones100k":
        return "Bones"
    return DATASET_LABELS.get(dataset, dataset.upper())


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def prompt_label(prompt: str) -> str:
    return PROMPT_LABELS.get(prompt, prompt)


def metric_column(metric: str) -> str:
    return f"{metric}_mean"


def load_result_table(
    results_dir: Path,
    datasets: tuple[str, ...],
    metrics: tuple[str, ...],
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    dataset_filter = set(datasets)
    requested_columns = metric_summary_columns(metrics)
    table: dict[tuple[str, str], dict[str, dict[str, float]]] = {}

    for result_model, result_protocol, dataset, run_dir in iter_run_dirs(results_dir):
        if dataset not in dataset_filter:
            continue

        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model = str(summary.get("model") or result_model).strip()
        protocol = str(summary.get("protocol") or result_protocol).strip()
        if result_protocol in {"label", "object"}:
            # These prompt-family folders currently record the generic summary
            # protocol "text". Keep the prompt family so distinct result series
            # do not collapse into the same model/protocol row.
            protocol = result_protocol
        if not model or not protocol:
            continue

        values: dict[str, float] = {}
        for column in requested_columns:
            value = summary.get(column)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values[column] = number

        if not values:
            continue

        table.setdefault((model, protocol), {})[dataset] = values

    return table


def row_keys(table: dict[tuple[str, str], dict[str, dict[str, float]]]) -> list[tuple[str, str]]:
    order = {
        "medicalsam3": 0,
        "medsam": 1,
        "medsam3": 2,
        "samus": 3,
        "ultrasam": 4,
    }
    return sorted(table, key=lambda key: (order.get(key[0], 99), key[0], key[1]))


def display_metric_label(metric: str) -> str:
    if metric == "dice":
        return "Dice"
    if metric == "assd":
        return "ASSD"
    return metric


def caption_text(metrics: tuple[str, ...]) -> str:
    metric_note = ""
    if len(metrics) == 1:
        metric_note = f" Reported metric: {display_metric_label(metrics[0])}."
    return LATEX_CAPTION + metric_note


def format_cell(value: float | None, metric: str | None = None) -> str:
    if value is None:
        return "-"
    if metric == "dice":
        return f"{value * 100:.2f}"
    if metric == "assd":
        return f"{value:.2f}"
    return format_float(value)


def per_dataset_csv_columns(datasets: tuple[str, ...], metrics: tuple[str, ...]) -> list[str]:
    columns = ["model", "prompt"]
    for dataset in datasets:
        for metric in metrics:
            columns.append(f"{dataset}_{metric}_mean")
    for metric in metrics:
        columns.append(f"avg_{metric}_mean")
    return columns


def aggregate_csv_columns(metrics: tuple[str, ...]) -> list[str]:
    columns = ["model", "prompt"]
    for metric in metrics:
        columns.extend([f"{metric}_mean", f"{metric}_std"])
    return columns


def mean_for_metric(
    table: dict[tuple[str, str], dict[str, dict[str, float]]],
    key: tuple[str, str],
    datasets: tuple[str, ...],
    metric: str,
) -> float | None:
    values = [
        table[key][dataset][metric_column(metric)]
        for dataset in datasets
        if metric_column(metric) in table[key].get(dataset, {})
    ]
    if not values:
        return None
    return statistics.fmean(values)


def per_dataset_numeric_rows(
    table: dict[tuple[str, str], dict[str, dict[str, float]]],
    datasets: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = per_dataset_csv_columns(datasets, metrics)
    for model, protocol in row_keys(table):
        row: dict[str, Any] = {column: None for column in columns}
        row["model"] = model_label(model)
        row["prompt"] = prompt_label(protocol)
        for dataset in datasets:
            values = table[(model, protocol)].get(dataset, {})
            for metric in metrics:
                row[f"{dataset}_{metric}_mean"] = values.get(metric_column(metric))
        for metric in metrics:
            row[f"avg_{metric}_mean"] = mean_for_metric(
                table, (model, protocol), datasets, metric
            )
        rows.append(row)
    return rows


def aggregate_numeric_rows(
    table: dict[tuple[str, str], dict[str, dict[str, float]]],
    datasets: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = aggregate_csv_columns(metrics)
    for model, protocol in row_keys(table):
        row: dict[str, Any] = {column: None for column in columns}
        row["model"] = model_label(model)
        row["prompt"] = prompt_label(protocol)
        for metric in metrics:
            values = [
                table[(model, protocol)][dataset][metric_column(metric)]
                for dataset in datasets
                if metric_column(metric) in table[(model, protocol)].get(dataset, {})
            ]
            if not values:
                continue
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def sort_rows_by_primary_avg(
    rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    per_dataset: bool,
) -> list[dict[str, Any]]:
    if not metrics:
        return rows
    column = f"avg_{metrics[0]}_mean" if per_dataset else f"{metrics[0]}_mean"
    higher_is_better = metrics[0] not in LOWER_IS_BETTER

    def sort_key(row: dict[str, Any]) -> tuple[int, float, str, str]:
        value = row.get(column)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return (1, 0.0, str(row.get("model", "")), str(row.get("prompt", "")))
        sortable = value if higher_is_better else -value
        return (0, sortable, str(row.get("model", "")), str(row.get("prompt", "")))

    return sorted(rows, key=sort_key)


def display_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for row in rows:
        formatted_row: dict[str, str] = {}
        for column in columns:
            value = row.get(column)
            if column in {"model", "prompt"}:
                formatted_row[column] = str(value)
            else:
                formatted_row[column] = format_cell(value, metric_from_column(column))
        formatted.append(formatted_row)
    return formatted


def csv_display_columns(columns: list[str]) -> list[str]:
    return ["Model" if column == "model" else "Prompt" if column == "prompt" else column for column in columns]


def csv_display_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    display: list[dict[str, str]] = []
    for row in rows:
        display_row: dict[str, str] = {}
        for column, value in row.items():
            output_column = "Model" if column == "model" else "Prompt" if column == "prompt" else column
            display_row[output_column] = value
        display.append(display_row)
    return display


def ranking_maps(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    best: dict[str, set[int]] = {}
    second: dict[str, set[int]] = {}
    for column in columns:
        if column in {"model", "prompt"}:
            continue
        values = [
            (idx, row[column])
            for idx, row in enumerate(rows)
            if isinstance(row.get(column), (int, float)) and math.isfinite(row[column])
        ]
        if not values:
            continue
        metric = metric_from_column(column)
        reverse = not column.endswith("_std") and metric not in LOWER_IS_BETTER
        ordered_unique = sorted({value for _idx, value in values}, reverse=reverse)
        if not ordered_unique:
            continue
        best_value = ordered_unique[0]
        best[column] = {idx for idx, value in values if value == best_value}
        if len(ordered_unique) > 1:
            second_value = ordered_unique[1]
            second[column] = {idx for idx, value in values if value == second_value}
    return best, second


def best_row_indices(rows: list[dict[str, Any]], column: str) -> set[int]:
    values = [
        (idx, row[column])
        for idx, row in enumerate(rows)
        if isinstance(row.get(column), (int, float)) and math.isfinite(row[column])
    ]
    if not values:
        return set()
    metric = metric_from_column(column)
    reverse = not column.endswith("_std") and metric not in LOWER_IS_BETTER
    best_value = sorted({value for _idx, value in values}, reverse=reverse)[0]
    return {idx for idx, value in values if value == best_value}


def metric_from_column(column: str) -> str:
    if column.startswith("avg_") and column.endswith("_mean"):
        return column[len("avg_") : -len("_mean")]
    if column.endswith("_mean"):
        middle = column[: -len("_mean")]
        parts = middle.split("_", 1)
        return parts[1] if len(parts) == 2 else middle
    if column.endswith("_std"):
        return column[: -len("_std")]
    return column


def markdown_emphasis(
    value: str,
    row_idx: int,
    column: str,
    best: dict[str, set[int]],
    second: dict[str, set[int]],
) -> str:
    if row_idx in best.get(column, set()):
        return f"**{value}**"
    if row_idx in second.get(column, set()):
        return f"<u>{value}</u>"
    return value


def latex_emphasis(
    value: str,
    row_idx: int,
    column: str,
    best: dict[str, set[int]],
    second: dict[str, set[int]],
) -> str:
    escaped = latex_escape(value)
    if row_idx in best.get(column, set()):
        return rf"\textbf{{{escaped}}}"
    if row_idx in second.get(column, set()):
        return rf"\underline{{{escaped}}}"
    return escaped


def latex_bold(value: str) -> str:
    return rf"\textbf{{{latex_escape(value)}}}"


def latex_model_cell(model: str, bold: bool = False) -> str:
    citation = MODEL_CITATIONS.get(model)
    value = latex_escape(model)
    if citation:
        value = rf"{value}~\cite{{{citation}}}"
    if bold:
        return rf"\textbf{{{value}}}"
    return value


def write_csv(rows: list[dict[str, str]], columns: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(lines: list[list[str]], alignments: list[str] | None = None) -> str:
    widths = [max(len(row[idx]) for row in lines) for idx in range(len(lines[0]))]
    rendered: list[str] = []
    for row_idx, row in enumerate(lines):
        rendered.append(
            "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        )
        if row_idx == 0:
            if alignments is None:
                alignments = ["---"] * len(row)
            rendered.append(
                "| "
                + " | ".join(alignments[idx].ljust(widths[idx]) for idx in range(len(row)))
                + " |"
            )
    return "\n".join(rendered)


def render_markdown_per_dataset(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metrics: tuple[str, ...],
) -> str:
    columns = per_dataset_csv_columns(datasets, metrics)
    best, second = ranking_maps(rows, columns)
    best_model_rows = best_row_indices(rows, f"avg_{metrics[0]}_mean")
    formatted = display_rows(rows, columns)

    if len(metrics) == 1:
        metric = metrics[0]
        header = ["Model", "Prompt", *[dataset_label(dataset) for dataset in datasets], "Avg"]
        body = [
            [
                f"**{row['model']}**" if row_idx in best_model_rows else row["model"],
                row["prompt"],
                *[
                    markdown_emphasis(
                        row[f"{dataset}_{metric}_mean"],
                        row_idx,
                        f"{dataset}_{metric}_mean",
                        best,
                        second,
                    )
                    for dataset in datasets
                ],
                markdown_emphasis(
                    row[f"avg_{metric}_mean"],
                    row_idx,
                    f"avg_{metric}_mean",
                    best,
                    second,
                ),
            ]
            for row_idx, row in enumerate(formatted)
        ]
        return markdown_table([header, *body])

    first_header = ["Model", "Prompt"]
    for dataset in datasets:
        first_header.extend([dataset_label(dataset), *[""] * (len(metrics) - 1)])
    first_header.extend(["Avg", *[""] * (len(metrics) - 1)])
    second_header = ["", ""]
    for _dataset in datasets:
        second_header.extend(display_metric_label(metric) for metric in metrics)
    second_header.extend(display_metric_label(metric) for metric in metrics)
    body = [
        [
            f"**{row['model']}**" if row_idx in best_model_rows else row["model"],
            row["prompt"],
            *[
                markdown_emphasis(
                    row[f"{dataset}_{metric}_mean"],
                    row_idx,
                    f"{dataset}_{metric}_mean",
                    best,
                    second,
                )
                for dataset in datasets
                for metric in metrics
            ],
            *[
                markdown_emphasis(
                    row[f"avg_{metric}_mean"],
                    row_idx,
                    f"avg_{metric}_mean",
                    best,
                    second,
                )
                for metric in metrics
            ],
        ]
        for row_idx, row in enumerate(formatted)
    ]
    return markdown_table([first_header, second_header, *body])


def render_markdown_aggregate(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> str:
    columns = aggregate_csv_columns(metrics)
    best, second = ranking_maps(rows, columns)
    best_model_rows = best_row_indices(rows, f"{metrics[0]}_mean")
    formatted = display_rows(rows, columns)
    columns = aggregate_csv_columns(metrics)
    header = ["Model", "Prompt"]
    for metric in metrics:
        label = display_metric_label(metric)
        header.extend([f"{label} mean", f"{label} std"])
    body = [
        [
            f"**{row['model']}**" if row_idx in best_model_rows else row["model"]
            if source_column == "model"
            else row["prompt"]
            if source_column == "prompt"
            else markdown_emphasis(row[source_column], row_idx, source_column, best, second)
            for source_column in columns
        ]
        for row_idx, row in enumerate(formatted)
    ]
    return markdown_table([header, *body])


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def latex_row(values: list[str], escape: bool = True) -> str:
    if escape:
        return " & ".join(latex_escape(value) for value in values) + r" \\"
    return " & ".join(values) + r" \\"


def render_latex_per_dataset(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metrics: tuple[str, ...],
) -> str:
    columns = per_dataset_csv_columns(datasets, metrics)
    best, second = ranking_maps(rows, columns)
    best_model_rows = best_row_indices(rows, f"avg_{metrics[0]}_mean")
    formatted = display_rows(rows, columns)
    dataset_metric_columns = len(datasets) * len(metrics)
    avg_columns = len(metrics)
    column_spec = "l|l|" + "c" * dataset_metric_columns + "|" + "c" * avg_columns
    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{latex_escape(caption_text(metrics))}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
    ]
    if len(metrics) == 1:
        metric = metrics[0]
        lines.append(
            latex_row(
                [
                    latex_bold("Model"),
                    latex_bold("Prompt"),
                    *[latex_bold(dataset_label(dataset)) for dataset in datasets],
                    latex_bold("Avg"),
                ],
                escape=False,
            )
        )
        lines.append("\\midrule")
        for row_idx, row in enumerate(formatted):
            lines.append(
                latex_row(
                    [
                        latex_model_cell(row["model"], bold=row_idx in best_model_rows),
                        latex_escape(row["prompt"]),
                        *[
                            latex_emphasis(
                                row[f"{dataset}_{metric}_mean"],
                                row_idx,
                                f"{dataset}_{metric}_mean",
                                best,
                                second,
                            )
                            for dataset in datasets
                        ],
                        latex_emphasis(
                            row[f"avg_{metric}_mean"],
                            row_idx,
                            f"avg_{metric}_mean",
                            best,
                            second,
                        ),
                    ],
                    escape=False,
                )
            )
        lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
        return "\n".join(lines) + "\n"

    header = [
        rf"\multirow{{2}}{{*}}{{{latex_bold('Model')}}}",
        rf"\multirow{{2}}{{*}}{{{latex_bold('Prompt')}}}",
    ]
    for dataset in datasets:
        header.append(
            f"\\multicolumn{{{len(metrics)}}}{{c}}{{{latex_bold(dataset_label(dataset))}}}"
        )
    header.append(f"\\multicolumn{{{len(metrics)}}}{{|c}}{{{latex_bold('Avg')}}}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(
        r"\cmidrule(lr){3-"
        + str(2 + len(metrics))
        + "}"
        if len(datasets) + 1 == 1
        else " ".join(
            rf"\cmidrule(lr){{{3 + idx * len(metrics)}-{2 + (idx + 1) * len(metrics)}}}"
            for idx in range(len(datasets) + 1)
        )
    )
    lines.append(
        latex_row(
            [
                "",
                "",
                *[
                    latex_bold(display_metric_label(metric))
                    for _dataset in datasets
                    for metric in metrics
                ],
                *[latex_bold(display_metric_label(metric)) for metric in metrics],
            ],
            escape=False,
        )
    )
    lines.append("\\midrule")
    for row_idx, row in enumerate(formatted):
        lines.append(
            latex_row(
                [
                    latex_model_cell(row["model"], bold=row_idx in best_model_rows),
                    latex_escape(row["prompt"]),
                    *[
                        latex_emphasis(
                            row[f"{dataset}_{metric}_mean"],
                            row_idx,
                            f"{dataset}_{metric}_mean",
                            best,
                            second,
                        )
                        for dataset in datasets
                        for metric in metrics
                    ],
                    *[
                        latex_emphasis(
                            row[f"avg_{metric}_mean"],
                            row_idx,
                            f"avg_{metric}_mean",
                            best,
                            second,
                        )
                        for metric in metrics
                    ],
                ],
                escape=False,
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def render_latex_aggregate(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> str:
    columns = aggregate_csv_columns(metrics)
    best, second = ranking_maps(rows, columns)
    best_model_rows = best_row_indices(rows, f"{metrics[0]}_mean")
    formatted = display_rows(rows, columns)
    column_spec = "l|l|" + "c" * (len(columns) - 2)
    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{latex_escape(caption_text(metrics))}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        latex_row(
            [
                latex_bold("Model")
                if column == "model"
                else latex_bold("Prompt")
                if column == "prompt"
                else latex_bold(f"{display_metric_label(metric_from_column(column))} std")
                if column.endswith("_std")
                else latex_bold(f"{display_metric_label(metric_from_column(column))} mean")
                for column in columns
            ],
            escape=False,
        ),
        "\\midrule",
    ]
    for row_idx, row in enumerate(formatted):
        lines.append(
            latex_row(
                [
                    (
                        latex_model_cell(row[column], bold=row_idx in best_model_rows)
                        if column == "model"
                        else latex_escape(row[column])
                    )
                    if column in {"model", "prompt"}
                    else latex_emphasis(row[column], row_idx, column, best, second)
                    for column in columns
                ],
                escape=False,
            )
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def write_text(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tabulate benchmark results with model/protocol rows."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets to include, in output order.",
    )
    parser.add_argument(
        "--metrics",
        default="dice,assd",
        help=(
            "Comma-separated result metrics to include, or 'all'. "
            f"Available: {', '.join(DEFAULT_METRICS)}."
        ),
    )
    parser.add_argument("--per-dataset", type=parse_bool, default=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("analysis") / "figures" / "results_table",
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated output formats: markdown,csv,latex.",
    )
    args = parser.parse_args()

    datasets = parse_csv_arg(args.datasets)
    if not datasets:
        raise SystemExit("At least one dataset must be selected.")
    metrics = parse_metrics_arg(args.metrics)
    formats = parse_formats_arg(args.formats)

    table = load_result_table(args.results_dir, datasets, metrics)
    if not table:
        raise SystemExit("No matching result summaries found.")

    if args.per_dataset:
        columns = per_dataset_csv_columns(datasets, metrics)
        rows = per_dataset_numeric_rows(table, datasets, metrics)
        rows = sort_rows_by_primary_avg(rows, metrics, per_dataset=True)
        markdown = render_markdown_per_dataset(rows, datasets, metrics)
        latex = render_latex_per_dataset(rows, datasets, metrics)
    else:
        columns = aggregate_csv_columns(metrics)
        rows = aggregate_numeric_rows(table, datasets, metrics)
        rows = sort_rows_by_primary_avg(rows, metrics, per_dataset=False)
        markdown = render_markdown_aggregate(rows, metrics)
        latex = render_latex_aggregate(rows, metrics)
    csv_rows = display_rows(rows, columns)

    print(markdown)

    output_paths = {
        "markdown": args.output_prefix.with_suffix(".md"),
        "csv": args.output_prefix.with_suffix(".csv"),
        "latex": args.output_prefix.with_suffix(".tex"),
    }
    if "markdown" in formats:
        write_text(output_paths["markdown"], markdown + "\n")
        print(f"\nSaved Markdown to: {output_paths['markdown']}")
    if "csv" in formats:
        write_csv(csv_display_rows(csv_rows), csv_display_columns(columns), output_paths["csv"])
        print(f"Saved CSV to: {output_paths['csv']}")
    if "latex" in formats:
        write_text(output_paths["latex"], latex)
        print(f"Saved LaTeX to: {output_paths['latex']}")


if __name__ == "__main__":
    main()
