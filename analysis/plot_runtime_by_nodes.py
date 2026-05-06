#!/usr/bin/env python3
"""Plot LSAS runtime by node count from raw result CSV files.

Usage:
    python plot_runtime_by_nodes.py
    python plot_runtime_by_nodes.py --input real_lsas_results/gaussian_run_with_all_use_for_time_complexity --out figures/runtime_by_nodes.png
    python plot_runtime_by_nodes.py --runtime-column cpu_runtime_seconds --summary-out results_metric/runtime_by_nodes.csv

From the repository root:
    python NeurIPS_2026/analysis/plot_runtime_by_nodes.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ANALYSIS_DIR / "real_lsas_results" / "gaussian_run_with_all_use_for_time_complexity"
)
DEFAULT_OUT = ANALYSIS_DIR / "figures" / "runtime_by_nodes.png"
DEFAULT_SUMMARY_OUT = ANALYSIS_DIR / "results_metric" / "runtime_by_nodes.csv"
RUNTIME_COLUMNS = {"wall_runtime_seconds", "cpu_runtime_seconds"}
NA_VALUES = {"", "NA", "N/A", "NaN", "nan", "None", "none", "null", "NULL"}
PAPER_TICK_LABEL_SIZE = 30
PAPER_AXIS_LABEL_SIZE = 34
PAPER_LEGEND_LABEL_SIZE = 28
PAPER_MARKER_SIZE = 18
PAPER_LINE_WIDTH = 4.2
PAPER_ERROR_LINE_WIDTH = 3.6
PAPER_CAP_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV file or directory scanned recursively. Default: {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output figure path. Default: {DEFAULT_OUT}.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_OUT,
        help=f"Output CSV with plotted means and SDs. Default: {DEFAULT_SUMMARY_OUT}.",
    )
    parser.add_argument(
        "--runtime-column",
        choices=sorted(RUNTIME_COLUMNS),
        default="wall_runtime_seconds",
        help="Runtime column to plot. Default: wall_runtime_seconds.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Optional extra formats to write using --out as the filename stem, for example: png pdf svg.",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Connect method points with lines.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not draw the method legend.",
    )
    return parser.parse_args()


def raise_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise SystemExit(
            "Missing dependency: matplotlib is required to create the runtime figure."
        ) from exc
    return plt


def parse_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text in NA_VALUES:
        return math.nan
    return float(text)


def csv_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.csv"))
    raise SystemExit(f"Input path does not exist: {input_path}")


def read_runtime_values(
    input_path: Path, runtime_column: str
) -> dict[tuple[str, float], list[float]]:
    raise_csv_field_size_limit()
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    paths = csv_paths(input_path)
    if not paths:
        raise SystemExit(f"No CSV files found under: {input_path}")

    required = {"n_nodes", "mb_method", runtime_column}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(
                    f"{path} is missing required columns: {', '.join(sorted(missing))}"
                )
            for row in reader:
                n_nodes = parse_float(row.get("n_nodes"))
                runtime = parse_float(row.get(runtime_column))
                method = str(row.get("mb_method", "")).strip()
                if not method or math.isnan(n_nodes) or math.isnan(runtime):
                    continue
                grouped[(method, n_nodes)].append(runtime)

    if not grouped:
        raise SystemExit(f"No finite {runtime_column} values found in: {input_path}")
    return grouped


def summarize_runtime(
    grouped: dict[tuple[str, float], list[float]],
) -> list[dict[str, str]]:
    rows = []
    for (method, n_nodes), values in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        rows.append(
            {
                "n_nodes": format_number(n_nodes),
                "mb_method": method,
                "n_samples": str(len(values)),
                "mean_runtime_seconds": format_number(mean(values)),
                "sd_runtime_seconds": format_number(
                    stdev(values) if len(values) > 1 else math.nan
                ),
            }
        )
    return rows


def format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def parse_int_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def visible_node_ticks(node_counts: list[float]) -> list[float]:
    if not node_counts:
        return node_counts
    hide_15 = max(node_counts) >= 100.0
    hide_between_8_and_50 = max(node_counts) > 200.0
    return [
        value
        for value in node_counts
        if not (hide_15 and abs(value - 15.0) < 1e-6)
        and not (hide_between_8_and_50 and 8.0 < value < 50.0)
    ]


def method_label(method: str) -> str:
    labels = {
        "fast_iamb": "Fast-IAMB",
        "hiton_mb": "HITON-MB",
        "iamb": "IAMB",
        "tc": "TC",
    }
    return labels.get(method.lower(), method.replace("_", " "))


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "n_nodes",
                "mb_method",
                "n_samples",
                "mean_runtime_seconds",
                "sd_runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def x_offset(index: int, n_methods: int, node_counts: list[float]) -> float:
    if n_methods <= 1:
        return 0.0
    unique_nodes = sorted(set(node_counts))
    gaps = [
        right - left
        for left, right in zip(unique_nodes, unique_nodes[1:])
        if right > left
    ]
    step = min(gaps) * 0.045 if gaps else max(unique_nodes[0] * 0.025, 0.025)
    return (index - (n_methods - 1) / 2) * step


def apply_paper_style(ax: Any) -> None:
    ax.tick_params(axis="both", which="major", labelsize=PAPER_TICK_LABEL_SIZE)
    ax.xaxis.label.set_size(PAPER_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(PAPER_AXIS_LABEL_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(PAPER_LEGEND_LABEL_SIZE)


def save_plot(
    summary_rows: list[dict[str, str]],
    out_path: Path,
    runtime_column: str,
    lines: bool,
    show_legend: bool,
    formats: list[str] | None,
) -> list[Path]:
    plt = load_matplotlib()
    by_method: dict[str, dict[float, tuple[float, float]]] = defaultdict(dict)
    for row in summary_rows:
        method = row["mb_method"]
        n_nodes = parse_float(row["n_nodes"])
        runtime_mean = parse_float(row["mean_runtime_seconds"])
        runtime_sd = parse_float(row["sd_runtime_seconds"])
        if math.isnan(n_nodes) or math.isnan(runtime_mean):
            continue
        by_method[method][n_nodes] = (runtime_mean, runtime_sd)

    methods = sorted(by_method)
    node_counts = sorted(
        {n_nodes for series in by_method.values() for n_nodes in series}
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    for index, method in enumerate(methods):
        series = by_method[method]
        method_nodes = sorted(series)
        offset = x_offset(index, len(methods), node_counts)
        x_values = [n_nodes + offset for n_nodes in method_nodes]
        y_values = [series[n_nodes][0] for n_nodes in method_nodes]
        sds = [series[n_nodes][1] for n_nodes in method_nodes]
        yerr = (
            None
            if all(math.isnan(sd) for sd in sds)
            else [0.0 if math.isnan(sd) else sd for sd in sds]
        )
        ax.errorbar(
            x_values,
            y_values,
            yerr=yerr,
            marker="o",
            linestyle="-" if lines else "none",
            markersize=PAPER_MARKER_SIZE,
            linewidth=PAPER_LINE_WIDTH,
            elinewidth=PAPER_ERROR_LINE_WIDTH,
            capsize=PAPER_CAP_SIZE,
            label=method_label(method),
        )

    runtime_label = (
        "Wall runtime" if runtime_column == "wall_runtime_seconds" else "CPU runtime"
    )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(f"{runtime_label} per sample (s)")
    shown_node_counts = visible_node_ticks(node_counts)
    ax.set_xticks(shown_node_counts)
    ax.set_xticklabels([parse_int_label(n_nodes) for n_nodes in shown_node_counts])
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25)
    if show_legend:
        ax.legend()
    apply_paper_style(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []
    requested_formats = formats or [out_path.suffix.lstrip(".") or "png"]
    for image_format in requested_formats:
        suffix = image_format.lstrip(".")
        path = (
            out_path
            if len(requested_formats) == 1
            else out_path.with_suffix(f".{suffix}")
        )
        if path.suffix.lower() != f".{suffix.lower()}":
            path = path.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300)
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def main() -> None:
    args = parse_args()
    raise_csv_field_size_limit()
    grouped = read_runtime_values(args.input, args.runtime_column)
    summary_rows = summarize_runtime(grouped)
    write_summary(summary_rows, args.summary_out)
    written = save_plot(
        summary_rows,
        args.out,
        args.runtime_column,
        lines=args.lines,
        show_legend=not args.no_legend,
        formats=args.formats,
    )
    print(f"Wrote {len(summary_rows)} summary row(s) to {args.summary_out}")
    print(
        f"Wrote {len(written)} figure file(s): {', '.join(str(path) for path in written)}"
    )


if __name__ == "__main__":
    main()
