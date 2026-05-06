#!/usr/bin/env python3
"""Combine metrics summary CSVs into a paper-ready summary CSV.

Usage:
    python create_paper_metric_summary.py with_xy_summary.csv
    python create_paper_metric_summary.py with_xy_summary.csv without_xy_summary.csv
    python create_paper_metric_summary.py --with-xy with_xy_summary.csv --without-xy without_xy_summary.csv --out paper_metric_summary.csv

From the repository root:
    python NeurIPS_2026/analysis/create_paper_metric_summary.py NeurIPS_2026/analysis/results_metric/metrics_summary_small_25_together.csv NeurIPS_2026/analysis/results_metric/metrics_summary_no_xy_edge.csv

The input summaries are expected to have the schema produced by
``summarise_lsas_metrics.py``. The output aggregates over noise types using the
pooled mean and sample standard deviation requested for the paper tables.
The without-X->Y summary is optional; when omitted, the all-SCM component is
computed from the with-X->Y summary only.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_WITH_XY = ANALYSIS_DIR / "results_metric" / "metrics_summary_small_25_together.csv"
DEFAULT_OUT = ANALYSIS_DIR / "results_metric" / "paper_metric_summary.csv"

GROUP_COLUMNS = ["n_nodes", "sample_size", "cyclicity", "pretreatment", "mb_method"]
ID_COLUMNS = GROUP_COLUMNS + ["component", "scm_type"]
OUTPUT_COLUMNS = ID_COLUMNS + [
    "effective_n",
    "source_rows",
    "source_summaries",
    "mean_re",
    "sd_re",
    "mean_precision",
    "sd_precision",
    "mean_correct_edge_identification",
    "sd_correct_edge_identification",
    "mean_empty_set_fraction",
    "sd_empty_set_fraction",
    "mean_ntest",
    "sd_ntest",
    "mean_cpu_time",
    "sd_cpu_time",
    "mean_wall_time",
    "sd_wall_time",
]

XY_ONLY_METRICS = {
    "re": ("mean_re", "sd_re"),
    "precision": ("mean_precision", "sd_precision"),
}
ALL_INPUT_METRICS = {
    "correct_edge_identification": (
        "correct_edge_identification",
        "sd_correct_edge_identification",
    ),
    "empty_set_fraction": ("empty_set_fraction", None),
    "ntest": ("mean_ntest", "sd_ntest"),
    "cpu_time": ("mean_cpu_time", "sd_cpu_time"),
    "wall_time": ("mean_wall_time", "sd_wall_time"),
}
NA_VALUES = {"", "NA", "N/A", "NaN", "nan", "None", "none", "null", "NULL"}
NORMALIZED_LABEL_COLUMNS = {"cyclicity", "pretreatment", "mb_method"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "with_xy_summary",
        nargs="?",
        type=Path,
        help="Metrics summary CSV generated from data with the X->Y edge.",
    )
    parser.add_argument(
        "without_xy_summary",
        nargs="?",
        type=Path,
        help="Metrics summary CSV generated from data without the X->Y edge.",
    )
    parser.add_argument(
        "--with-xy",
        type=Path,
        default=DEFAULT_WITH_XY,
        help=f"Metrics summary CSV generated from data with the X->Y edge. Default: {DEFAULT_WITH_XY}.",
    )
    parser.add_argument(
        "--without-xy",
        type=Path,
        default=None,
        help="Optional metrics summary CSV generated from data without the X->Y edge.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output paper metric summary CSV. Default: {DEFAULT_OUT}.",
    )
    args = parser.parse_args()
    if args.with_xy_summary is not None:
        args.with_xy = args.with_xy_summary
    if args.without_xy_summary is not None:
        args.without_xy = args.without_xy_summary
    return args


def parse_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text in NA_VALUES:
        return math.nan
    return float(text)


def parse_weight(row: dict[str, str]) -> float:
    n_rows = parse_float(row.get("n_rows"))
    empty_fraction = parse_float(row.get("empty_set_fraction"))
    if math.isnan(n_rows):
        return 0.0
    if math.isnan(empty_fraction):
        empty_fraction = 0.0
    return max(0.0, n_rows * (1.0 - empty_fraction))


def group_value(row: dict[str, str], column: str) -> str:
    value = str(row.get(column, "")).strip()
    if column in NORMALIZED_LABEL_COLUMNS:
        return value.lower()
    return value


def group_key(row: dict[str, str], component: str, scm_type: str) -> tuple[str, ...]:
    return tuple(group_value(row, column) for column in GROUP_COLUMNS) + (
        component,
        scm_type,
    )


def load_rows(path: Path, source_label: str) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Input summary does not exist: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_source_label"] = source_label
    return rows


def load_optional_rows(path: Path | None, source_label: str) -> list[dict[str, str]]:
    if path is None:
        return []
    return load_rows(path, source_label)


def finite_metric_parts(
    rows: Iterable[dict[str, str]], mean_column: str, sd_column: str | None
) -> list[tuple[float, float, float]]:
    parts = []
    for row in rows:
        weight = parse_weight(row)
        mean_value = parse_float(row.get(mean_column))
        if weight <= 0 or math.isnan(mean_value):
            continue
        sd_value = 0.0 if sd_column is None else parse_float(row.get(sd_column))
        if math.isnan(sd_value):
            sd_value = 0.0
        parts.append((weight, mean_value, sd_value))
    return parts


def pooled_mean_sd(parts: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    total_n = sum(weight for weight, _, _ in parts)
    if total_n <= 0:
        return math.nan, math.nan, 0.0

    pooled_mean = sum(weight * mean_value for weight, mean_value, _ in parts) / total_n
    if total_n <= 1:
        return pooled_mean, math.nan, total_n

    numerator = 0.0
    for weight, mean_value, sd_value in parts:
        numerator += max(0.0, weight - 1.0) * sd_value**2
        numerator += weight * (mean_value - pooled_mean) ** 2
    pooled_variance = numerator / (total_n - 1.0)
    return pooled_mean, math.sqrt(max(0.0, pooled_variance)), total_n


def format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def source_row_count(rows: list[dict[str, str]]) -> float:
    total = 0.0
    for row in rows:
        n_rows = parse_float(row.get("n_rows"))
        if not math.isnan(n_rows):
            total += n_rows
    return total


def empty_output_row(key: tuple[str, ...], rows: list[dict[str, str]]) -> dict[str, str]:
    output = dict(zip(ID_COLUMNS, key))
    output["effective_n"] = "0"
    output["source_rows"] = format_number(source_row_count(rows))
    output["source_summaries"] = str(len(rows))
    for column in OUTPUT_COLUMNS:
        output.setdefault(column, "nan")
    return output


def add_metric(
    output: dict[str, str],
    rows: list[dict[str, str]],
    output_name: str,
    mean_column: str,
    sd_column: str | None,
) -> None:
    mean_value, sd_value, effective_n = pooled_mean_sd(
        finite_metric_parts(rows, mean_column, sd_column)
    )
    output[f"mean_{output_name}"] = format_number(mean_value)
    output[f"sd_{output_name}"] = format_number(sd_value)
    output["effective_n"] = format_number(max(parse_float(output["effective_n"]), effective_n))


def build_summary(with_xy_rows: list[dict[str, str]], without_xy_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)

    for row in with_xy_rows:
        scm_type = str(row.get("scm_type", "")).strip().lower()
        if scm_type in {"linear", "nonlinear"}:
            grouped[group_key(row, f"xy_{scm_type}", scm_type)].append(row)

    for row in with_xy_rows + without_xy_rows:
        grouped[group_key(row, "all_scm", "all")].append(row)

    output_rows = []
    for key in sorted(grouped):
        rows = grouped[key]
        output = empty_output_row(key, rows)
        component = key[ID_COLUMNS.index("component")]

        if component in {"xy_linear", "xy_nonlinear"}:
            for output_name, (mean_column, sd_column) in XY_ONLY_METRICS.items():
                add_metric(output, rows, output_name, mean_column, sd_column)
        elif component == "all_scm":
            for output_name, (mean_column, sd_column) in ALL_INPUT_METRICS.items():
                add_metric(output, rows, output_name, mean_column, sd_column)
        else:
            raise ValueError(f"Unknown component: {component}")

        output_rows.append(output)

    return output_rows


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with_xy_rows = load_rows(args.with_xy, "with_xy")
    without_xy_rows = load_optional_rows(args.without_xy, "without_xy")
    output_rows = build_summary(with_xy_rows, without_xy_rows)
    write_summary(output_rows, args.out)
    print(f"Wrote {len(output_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
