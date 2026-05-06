#!/usr/bin/env python3
"""Create LSAS metric figures from a metrics summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY = ANALYSIS_DIR / "results_metric" / "metrics_summary.csv"
DEFAULT_FIGURE_DIR_NAME = "figures"
PAPER_TICK_LABEL_SIZE = 30
PAPER_AXIS_LABEL_SIZE = 34
PAPER_LEGEND_LABEL_SIZE = 28
PAPER_MARKER_SIZE = 18
PAPER_LINE_WIDTH = 4.2
PAPER_ERROR_LINE_WIDTH = 3.6
PAPER_CAP_SIZE = 10
NORMAL_MARKER_SIZE = 6
NORMAL_LINE_WIDTH = 1.5
NORMAL_ERROR_LINE_WIDTH = 1.2
NORMAL_CAP_SIZE = 3

METRICS = {
    "re": {
        "mean": "mean_re",
        "sd": "sd_re",
        "title": "Relative Error",
        "ylabel": "RE",
    },
    "ntest": {
        "mean": "mean_ntest",
        "sd": "sd_ntest",
        "title": "Independence Tests",
        "ylabel": "Number of tests",
    },
    "cpu_time": {
        "mean": "mean_cpu_time",
        "sd": "sd_cpu_time",
        "title": "CPU Runtime",
        "ylabel": "CPU runtime (seconds)",
    },
    "wall_time": {
        "mean": "mean_wall_time",
        "sd": "sd_wall_time",
        "title": "Wall Runtime",
        "ylabel": "Wall runtime (s)",
    },
    "subset_validity": {
        "mean": "mean_subset_validity",
        "sd": "sd_subset_validity",
        "title": "Subset Validity",
        "ylabel": "Subset validity",
    },
    "precision": {
        "mean": "mean_precision",
        "sd": "sd_precision",
        "title": "Precision",
        "ylabel": "Pre",
    },
    "error_rate": {
        "mean": "mean_error_rate",
        "sd": "sd_error_rate",
        "title": "Error Rate",
        "ylabel": "Error rate",
    },
    "correct_edge_identification": {
        "mean": "correct_edge_identification",
        "sd": "sd_correct_edge_identification",
        "title": "Correct Edge Identification",
        "ylabel": "EF",
    },
    "empty_set_fraction": {
        "mean": "empty_set_fraction",
        "sd": "sd_empty_set_fraction",
        "title": "Empty Set Prediction",
        "ylabel": "Empty fraction",
    },
}
NODE_SCALING_METRICS = ["ntest", "cpu_time", "wall_time"]
SAMPLE_15000_GRAPH_TYPE_NODE_SCALING_METRICS = [
    "re",
    "precision",
    "correct_edge_identification",
]
NODE_SCALING_SAMPLE_SIZE = 15000.0
COMPARISON_METRICS = [
    "re",
    "precision",
    "subset_validity",
    "error_rate",
    "correct_edge_identification",
    "empty_set_fraction",
]
OPTIONAL_SD_METRICS = {"correct_edge_identification"}
GRAPH_TYPE_COLOR_METRICS = {
    "re",
    "precision",
    "correct_edge_identification",
    "empty_set_fraction",
}
DEFAULT_COMPARISON_MB_METHOD = "Fast_IAMB"
PAPER_METRIC_SLUGS = {
    "re": "re",
    "ntest": "ntest",
    "cpu_time": "cpu",
    "wall_time": "wall",
    "subset_validity": "subset",
    "precision": "prec",
    "error_rate": "err",
    "correct_edge_identification": "edge_frac",
    "empty_set_fraction": "empty_frac",
}
PAPER_COMPARISON_SLUGS = {
    "graph_type": "graph_type",
    "noise_type": "noise",
    "scm_type": "scm",
}


def default_figure_dir() -> Path:
    cwd = Path.cwd().resolve()
    if cwd == ANALYSIS_DIR:
        return Path(DEFAULT_FIGURE_DIR_NAME)
    return ANALYSIS_DIR / DEFAULT_FIGURE_DIR_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Metrics summary CSV. Default: {DEFAULT_SUMMARY}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Figure output directory. Default: {default_figure_dir()}.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Image formats to write, for example: png pdf svg.",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Connect points with lines. By default, plots show points only.",
    )
    parser.add_argument(
        "--comparison-mb-method",
        default=DEFAULT_COMPARISON_MB_METHOD,
        help=(
            "Markov blanket method used for graph-type and noise-type comparison "
            f"plots. Default: {DEFAULT_COMPARISON_MB_METHOD}."
        ),
    )
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help=(
            "Create paper-ready figures: no in-figure titles, larger axis ticks "
            "and labels, and filenames that include the plot context."
        ),
    )
    parser.add_argument(
        "--no-legends",
        action="store_true",
        help="Do not draw legends in generated figures.",
    )
    parser.add_argument(
        "--one-mb-only",
        action="store_true",
        help=(
            "Use only --comparison-mb-method for all figures, instead of "
            "comparing Markov blanket methods in the standard plots."
        ),
    )
    return parser.parse_args()


def load_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise SystemExit(
            "Missing dependency: matplotlib is required to create figures."
        ) from exc
    return plt


def parse_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text in {"", "NA", "NaN", "nan", "None", "null"}:
        return math.nan
    return float(text)


def parse_metric_sd(row: dict[str, str], metric_key: str) -> float:
    """Return nan when an optional SD column is absent."""
    sd_column = METRICS[metric_key]["sd"]
    if not sd_column:
        return math.nan
    if sd_column not in row and metric_key in OPTIONAL_SD_METRICS:
        return math.nan
    return parse_float(row.get(sd_column))


def errorbar_yerr(sds: list[float]) -> list[float] | None:
    """Use plain point/line plots when no SDs are available."""
    if all(math.isnan(sd) for sd in sds):
        return None
    return [0 if math.isnan(sd) else sd for sd in sds]


def parse_int_label(value: Any) -> str:
    number = parse_float(value)
    if math.isnan(number):
        return str(value).strip()
    if number.is_integer():
        return str(int(number))
    return str(number)


def node_count_tick_label(value: Any, *, hide_15: bool = False) -> str:
    number = parse_float(value)
    if hide_15 and not math.isnan(number) and abs(number - 15.0) < 1e-6:
        return ""
    return parse_int_label(value)


def apply_node_axis_ticks(ax: Any, node_counts: list[float]) -> None:
    hide_15 = bool(node_counts) and max(node_counts) >= 100.0
    hide_between_8_and_50 = bool(node_counts) and max(node_counts) > 200.0
    visible_node_counts = [
        value
        for value in node_counts
        if not (hide_15 and abs(value - 15.0) < 1e-6)
        and not (hide_between_8_and_50 and 8.0 < value < 50.0)
    ]
    ax.set_xticks(visible_node_counts)
    ax.set_xticklabels(
        [node_count_tick_label(value, hide_15=hide_15) for value in visible_node_counts]
    )


def graph_type(row: dict[str, str]) -> str:
    value = (row.get("cyclicity") or row.get("graph_family") or "").strip().lower()
    if value not in {"acyclic", "cyclic"}:
        raise ValueError(f"Missing or invalid graph type: {value!r}")
    return value


def scm_type(row: dict[str, str]) -> str:
    value = value_label(row, "scm_type")
    if value not in {"linear", "nonlinear", "all"}:
        raise ValueError(f"Missing or invalid SCM type: {value!r}")
    return value


def value_label(row: dict[str, str], column: str, default: str = "unknown") -> str:
    value = str(row.get(column) or "").strip().lower()
    return value if value else default


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def paper_metric_slug(metric_key: str) -> str:
    return PAPER_METRIC_SLUGS.get(metric_key, slug(metric_key))


def paper_pretreatment_slug(pretreatment: str) -> str:
    return "pre" if pretreatment == "true" else "nopre"


def paper_graph_slug(graph_kind: str) -> str:
    return "acyc" if graph_kind == "acyclic" else "cyc"


def paper_noise_slug(noise_type: str) -> str:
    return {
        "gaussian": "gau",
        "non_gaussian": "ngau",
        "random": "ran",
        "with_xy": "xy",
        "all_inputs": "all",
    }.get(noise_type, slug(noise_type))


def paper_method_slug(method: str) -> str:
    return slug(method).replace("fast_iamb", "fiamb")


def method_label(method: str) -> str:
    labels = {
        "fast_iamb": "Fast-IAMB",
        "hiton_mb": "HITON-MB",
        "iamb": "IAMB",
        "tc": "TC",
    }
    return labels.get(method.lower(), method.replace("_", " "))


def is_paper_metric_summary(fieldnames: list[str]) -> bool:
    return "component" in fieldnames and "effective_n" in fieldnames


def normalize_paper_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for row in rows:
        component = value_label(row, "component")
        adapted = dict(row)
        if component in {"xy_linear", "xy_nonlinear"}:
            adapted["noise_type"] = "with_xy"
        elif component == "all_scm":
            adapted["noise_type"] = "all_inputs"
        else:
            adapted["noise_type"] = component

        adapted["correct_edge_identification"] = row.get(
            "mean_correct_edge_identification", ""
        )
        adapted["empty_set_fraction"] = row.get("mean_empty_set_fraction", "")
        normalized.append(adapted)
    return normalized


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Summary CSV does not exist: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if is_paper_metric_summary(fieldnames):
            required = {
                "n_nodes",
                "sample_size",
                "mb_method",
                "cyclicity",
                "scm_type",
                "pretreatment",
                "component",
            }
            available_metric_columns = {
                "mean_re",
                "mean_precision",
                "mean_correct_edge_identification",
                "mean_empty_set_fraction",
                "mean_ntest",
                "mean_cpu_time",
                "mean_wall_time",
            }
            if not available_metric_columns.intersection(fieldnames):
                required.update(available_metric_columns)
            missing = required - set(fieldnames)
            if missing:
                raise SystemExit(
                    f"{path} is missing required columns: {', '.join(sorted(missing))}"
                )
            return normalize_paper_summary_rows(list(reader))

        required = {
            "n_nodes",
            "sample_size",
            "mb_method",
            "cyclicity",
            "scm_type",
            "noise_type",
            "pretreatment",
        }
        for metric in METRICS.values():
            required.add(metric["mean"])
        missing = required - set(fieldnames)
        if missing:
            raise SystemExit(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        return list(reader)


def aggregate_metric(
    rows: list[dict[str, str]], metric_key: str
) -> dict[str, dict[float, tuple[float, float]]]:
    metric = METRICS[metric_key]
    grouped: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)

    for row in rows:
        sample_size = parse_float(row["sample_size"])
        value = parse_float(row.get(metric["mean"]))
        sd = parse_metric_sd(row, metric_key)
        if math.isnan(sample_size) or math.isnan(value):
            continue
        grouped[(row["mb_method"], sample_size)].append((value, sd))

    by_method: dict[str, dict[float, tuple[float, float]]] = defaultdict(dict)
    for (method, sample_size), values in grouped.items():
        means = [value for value, _ in values if not math.isnan(value)]
        sds = [sd for _, sd in values if not math.isnan(sd)]
        by_method[method][sample_size] = (
            mean(means) if means else math.nan,
            mean(sds) if sds else math.nan,
        )
    return by_method


def aggregate_metric_by_label(
    rows: list[dict[str, str]], metric_key: str, label_fn: Any
) -> dict[str, dict[float, tuple[float, float]]]:
    metric = METRICS[metric_key]
    grouped: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)

    for row in rows:
        sample_size = parse_float(row["sample_size"])
        value = parse_float(row.get(metric["mean"]))
        sd = parse_metric_sd(row, metric_key)
        if math.isnan(sample_size) or math.isnan(value):
            continue
        grouped[(label_fn(row), sample_size)].append((value, sd))

    by_label: dict[str, dict[float, tuple[float, float]]] = defaultdict(dict)
    for (label, sample_size), values in grouped.items():
        means = [value for value, _ in values if not math.isnan(value)]
        sds = [sd for _, sd in values if not math.isnan(sd)]
        by_label[label][sample_size] = (
            mean(means) if means else math.nan,
            mean(sds) if sds else math.nan,
        )
    return by_label


def sorted_sample_sizes(series: dict[float, tuple[float, float]]) -> list[float]:
    return sorted(series)


def x_offset(
    index: int,
    n_methods: int,
    sample_sizes: list[float],
    fraction: float = 0.045,
) -> float:
    if n_methods <= 1:
        return 0.0

    unique_sizes = sorted(set(sample_sizes))
    gaps = [
        right - left
        for left, right in zip(unique_sizes, unique_sizes[1:])
        if right > left
    ]
    step = min(gaps) * fraction if gaps else max(unique_sizes[0] * fraction, 0.025)
    return (index - (n_methods - 1) / 2) * step


def sample_sd(values: list[float]) -> float:
    cleaned = [value for value in values if not math.isnan(value)]
    if len(cleaned) <= 1:
        return math.nan
    return stdev(cleaned)


def graph_type_title(value: str) -> str:
    return value.capitalize()


def noise_type_title(value: str) -> str:
    return value.replace("_", " ").title()


def pretreatment_title(value: str) -> str:
    return "Pretreatment" if value == "true" else "No pretreatment"


def apply_paper_style(ax: Any) -> None:
    """Make axes readable when figures are embedded without titles."""
    ax.tick_params(axis="both", which="major", labelsize=PAPER_TICK_LABEL_SIZE)
    ax.xaxis.label.set_size(PAPER_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(PAPER_AXIS_LABEL_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(PAPER_LEGEND_LABEL_SIZE)


def errorbar_style(paper_mode: bool) -> dict[str, float]:
    if paper_mode:
        return {
            "markersize": PAPER_MARKER_SIZE,
            "linewidth": PAPER_LINE_WIDTH,
            "elinewidth": PAPER_ERROR_LINE_WIDTH,
            "capsize": PAPER_CAP_SIZE,
        }
    return {
        "markersize": NORMAL_MARKER_SIZE,
        "linewidth": NORMAL_LINE_WIDTH,
        "elinewidth": NORMAL_ERROR_LINE_WIDTH,
        "capsize": NORMAL_CAP_SIZE,
    }


def maybe_add_legend(ax: Any, show_legend: bool) -> None:
    if show_legend:
        ax.legend()


def apply_axis_limits(ax: Any, metric_key: str) -> None:
    ax.set_xlim(left=0)
    if metric_key in {
        "re",
        "precision",
        "correct_edge_identification",
        "empty_set_fraction",
    }:
        ax.set_ylim(0, 1)
        return

    if metric_key in {
        "subset_validity",
        "error_rate",
    }:
        ax.set_ylim(0, 1.05)
    else:
        ax.set_ylim(bottom=0)


def save_metric_plot(
    plt: Any,
    rows: list[dict[str, str]],
    metric_key: str,
    graph_kind: str,
    noise_type: str,
    pretreatment: str,
    n_nodes: str,
    out_dir: Path,
    formats: list[str],
    lines: bool = False,
    paper_mode: bool = False,
    show_legend: bool = True,
) -> list[Path]:
    metric = METRICS[metric_key]
    by_method = aggregate_metric(rows, metric_key)
    if not by_method:
        return []

    fig, ax = plt.subplots(
        figsize=(9.0, 5.4) if paper_mode else (8.5, 5.2),
        constrained_layout=True,
    )
    methods = sorted(by_method)
    all_sample_sizes = sorted(
        {sample_size for series in by_method.values() for sample_size in series}
    )
    for index, method in enumerate(methods):
        series = by_method[method]
        sample_sizes = sorted_sample_sizes(series)
        offset = x_offset(index, len(methods), all_sample_sizes)
        x_values = [sample_size + offset for sample_size in sample_sizes]
        means = [series[sample_size][0] for sample_size in sample_sizes]
        sds = [series[sample_size][1] for sample_size in sample_sizes]
        ax.errorbar(
            x_values,
            means,
            yerr=errorbar_yerr(sds),
            marker=graph_kind_marker(graph_kind),
            linestyle="-" if lines else "none",
            label=method_label(method),
            color=graph_kind_color(plt, graph_kind, index, len(methods)),
            **errorbar_style(paper_mode),
        )

    if not paper_mode:
        title = (
            f"{metric['title']} — {graph_type_title(graph_kind)} graphs — "
            f"{n_nodes} nodes — {noise_type_title(noise_type)} noise — "
            f"{pretreatment_title(pretreatment)}"
        )
        ax.set_title(title)
    ax.set_xlabel("Sample size")
    ax.set_ylabel(metric["ylabel"])
    ax.set_xticks(all_sample_sizes)
    ax.set_xticklabels([parse_int_label(value) for value in all_sample_sizes])
    ax.grid(True, axis="y", alpha=0.25)
    maybe_add_legend(ax, show_legend)
    apply_axis_limits(ax, metric_key)
    if paper_mode:
        apply_paper_style(ax)

    metric_slug = paper_metric_slug(metric_key)
    metric_dir = out_dir / metric_slug
    metric_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if paper_mode:
        filename_prefix = (
            f"p_{paper_metric_slug(metric_key)}_samples_"
            f"{paper_graph_slug(graph_kind)}_{paper_noise_slug(noise_type)}_"
            f"{paper_pretreatment_slug(pretreatment)}_{n_nodes}n"
        )
    else:
        filename_prefix = (
            f"{metric_slug}_vs_samples_"
            f"{graph_kind}_{slug(noise_type)}_pre{pretreatment}_nodes{n_nodes}"
        )
    for image_format in formats:
        suffix = image_format.lstrip(".")
        path = metric_dir / f"{filename_prefix}.{suffix}"
        fig.savefig(path, dpi=300)
        written.append(path)
    plt.close(fig)
    return written


def aggregate_metric_by_nodes(
    rows: list[dict[str, str]], metric_key: str, label_fn: Any | None = None
) -> dict[str, dict[float, tuple[float, float]]]:
    metric = METRICS[metric_key]
    grouped: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)
    if label_fn is None:
        label_fn = lambda row: row["mb_method"]

    for row in rows:
        n_nodes = parse_float(row["n_nodes"])
        value = parse_float(row.get(metric["mean"]))
        sd = parse_metric_sd(row, metric_key)
        if math.isnan(n_nodes) or math.isnan(value):
            continue
        grouped[(label_fn(row), n_nodes)].append((value, sd))

    by_method: dict[str, dict[float, tuple[float, float]]] = defaultdict(dict)
    for (method, n_nodes), values in grouped.items():
        means = [value for value, _ in values if not math.isnan(value)]
        sds = [sd for _, sd in values if not math.isnan(sd)]
        by_method[method][n_nodes] = (
            mean(means) if means else math.nan,
            mean(sds) if sds else math.nan,
        )
    return by_method


def save_node_scaling_plot(
    plt: Any,
    rows: list[dict[str, str]],
    metric_key: str,
    graph_kind: str,
    noise_type: str,
    pretreatment: str,
    out_dir: Path,
    formats: list[str],
    lines: bool = False,
    paper_mode: bool = False,
    show_legend: bool = True,
    sample_size: float | None = None,
) -> list[Path]:
    metric = METRICS[metric_key]
    by_method = aggregate_metric_by_nodes(rows, metric_key)
    if not by_method:
        return []

    fig, ax = plt.subplots(
        figsize=(9.0, 5.4) if paper_mode else (8.5, 5.2),
        constrained_layout=True,
    )
    methods = sorted(by_method)
    all_n_nodes = sorted(
        {n_nodes for series in by_method.values() for n_nodes in series}
    )
    for index, method in enumerate(methods):
        series = by_method[method]
        node_counts = sorted_sample_sizes(series)
        offset = x_offset(index, len(methods), all_n_nodes, fraction=0.12)
        x_values = [n_nodes + offset for n_nodes in node_counts]
        means = [series[n_nodes][0] for n_nodes in node_counts]
        sds = [series[n_nodes][1] for n_nodes in node_counts]
        ax.errorbar(
            x_values,
            means,
            yerr=errorbar_yerr(sds),
            marker=graph_kind_marker(graph_kind),
            linestyle="-" if lines else "none",
            label=method_label(method),
            color=graph_kind_color(plt, graph_kind, index, len(methods)),
            **errorbar_style(paper_mode),
        )

    if not paper_mode:
        sample_text = (
            f" - {parse_int_label(sample_size)} samples"
            if sample_size is not None
            else ""
        )
        title = (
            f"{metric['title']} by Nodes — {graph_type_title(graph_kind)} graphs — "
            f"{noise_type_title(noise_type)} noise — {pretreatment_title(pretreatment)}"
            f"{sample_text}"
        )
        ax.set_title(title)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(metric["ylabel"])
    apply_node_axis_ticks(ax, all_n_nodes)
    ax.grid(True, axis="y", alpha=0.25)
    maybe_add_legend(ax, show_legend and len(methods) > 1)
    apply_axis_limits(ax, metric_key)
    if paper_mode:
        apply_paper_style(ax)

    metric_dir = out_dir / (
        f"{paper_metric_slug(metric_key)}_nodes"
        if paper_mode
        else f"{paper_metric_slug(metric_key)}_by_nodes"
    )
    metric_dir.mkdir(parents=True, exist_ok=True)
    written = []
    sample_suffix = (
        f"_samples{parse_int_label(sample_size)}"
        if sample_size is not None
        else ""
    )
    if paper_mode:
        filename_prefix = (
            f"p_{paper_metric_slug(metric_key)}_nodes_"
            f"{paper_graph_slug(graph_kind)}_{paper_noise_slug(noise_type)}_"
            f"{paper_pretreatment_slug(pretreatment)}{sample_suffix}"
        )
    else:
        filename_prefix = (
            f"{paper_metric_slug(metric_key)}_by_nodes_"
            f"{graph_kind}_{slug(noise_type)}_pre{pretreatment}{sample_suffix}"
        )
    for image_format in formats:
        suffix = image_format.lstrip(".")
        path = metric_dir / f"{filename_prefix}.{suffix}"
        fig.savefig(path, dpi=300)
        written.append(path)
    plt.close(fig)
    return written


def graph_type_method_label(row: dict[str, str]) -> str:
    return f"{graph_type_title(graph_type(row))} - {method_label(row['mb_method'])}"


def graph_type_label(row: dict[str, str]) -> str:
    return graph_type_title(graph_type(row))


def label_graph_type(label: str) -> str:
    lowered = label.lower()
    if lowered.startswith("acyclic"):
        return "acyclic"
    if lowered.startswith("cyclic"):
        return "cyclic"
    return ""


def mix_rgb(color: tuple[float, float, float], target: float, amount: float) -> tuple[float, float, float]:
    return tuple(channel * (1.0 - amount) + target * amount for channel in color)


def graph_type_color(
    plt: Any, label: str, index_by_graph_type: dict[str, int], total_by_graph_type: dict[str, int]
) -> Any:
    graph_kind = label_graph_type(label)
    base_name = {
        "acyclic": "tab:red",
        "cyclic": "tab:blue",
    }.get(graph_kind)
    if base_name is None:
        return None

    base_rgb = plt.matplotlib.colors.to_rgb(base_name)
    total = total_by_graph_type.get(graph_kind, 1)
    index = index_by_graph_type.get(graph_kind, 0)
    index_by_graph_type[graph_kind] = index + 1
    if total <= 1:
        return base_rgb

    midpoint = (total - 1) / 2
    if index < midpoint:
        amount = 0.35 * (midpoint - index) / max(midpoint, 1)
        return mix_rgb(base_rgb, 1.0, amount)

    amount = 0.25 * (index - midpoint) / max(midpoint, 1)
    return mix_rgb(base_rgb, 0.0, amount)


def graph_type_marker(label: str) -> str:
    return "s" if label_graph_type(label) == "acyclic" else "o"


def graph_kind_marker(graph_kind: str) -> str:
    return "s" if graph_kind == "acyclic" else "o"


def graph_kind_base_color(graph_kind: str) -> str:
    return "tab:red" if graph_kind == "acyclic" else "tab:blue"


def graph_kind_color(
    plt: Any, graph_kind: str, index: int, total: int
) -> Any:
    base_rgb = plt.matplotlib.colors.to_rgb(graph_kind_base_color(graph_kind))
    if total <= 1:
        return base_rgb

    midpoint = (total - 1) / 2
    if index < midpoint:
        amount = 0.35 * (midpoint - index) / max(midpoint, 1)
        return mix_rgb(base_rgb, 1.0, amount)

    amount = 0.25 * (index - midpoint) / max(midpoint, 1)
    return mix_rgb(base_rgb, 0.0, amount)


def save_graph_type_node_scaling_plot(
    plt: Any,
    rows: list[dict[str, str]],
    metric_key: str,
    noise_type: str,
    pretreatment: str,
    out_dir: Path,
    formats: list[str],
    lines: bool = False,
    paper_mode: bool = False,
    show_legend: bool = True,
    sample_size: float | None = None,
) -> list[Path]:
    metric = METRICS[metric_key]
    mb_methods = {value_label(row, "mb_method") for row in rows}
    label_fn = graph_type_label if len(mb_methods) == 1 else graph_type_method_label
    by_label = aggregate_metric_by_nodes(
        rows, metric_key, label_fn=label_fn
    )
    if not by_label:
        return []

    fig, ax = plt.subplots(
        figsize=(9.0, 5.4) if paper_mode else (8.5, 5.2),
        constrained_layout=True,
    )
    labels = sorted(by_label)
    all_n_nodes = sorted(
        {n_nodes for series in by_label.values() for n_nodes in series}
    )
    total_by_graph_type: dict[str, int] = defaultdict(int)
    for label in labels:
        total_by_graph_type[label_graph_type(label)] += 1
    index_by_graph_type: dict[str, int] = defaultdict(int)
    for index, label in enumerate(labels):
        series = by_label[label]
        node_counts = sorted_sample_sizes(series)
        offset = x_offset(index, len(labels), all_n_nodes, fraction=0.12)
        x_values = [n_nodes + offset for n_nodes in node_counts]
        means = [series[n_nodes][0] for n_nodes in node_counts]
        sds = [series[n_nodes][1] for n_nodes in node_counts]
        color = graph_type_color(
            plt, label, index_by_graph_type, total_by_graph_type
        )
        ax.errorbar(
            x_values,
            means,
            yerr=errorbar_yerr(sds),
            marker=graph_type_marker(label),
            linestyle="-" if lines else "none",
            label=label,
            color=color,
            **errorbar_style(paper_mode),
        )

    if not paper_mode:
        sample_text = (
            f" - {parse_int_label(sample_size)} samples"
            if sample_size is not None
            else ""
        )
        title = (
            f"{metric['title']} by Nodes - Acyclic vs Cyclic - "
            f"{noise_type_title(noise_type)} noise - {pretreatment_title(pretreatment)}"
            f"{sample_text}"
        )
        ax.set_title(title)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(metric["ylabel"])
    apply_node_axis_ticks(ax, all_n_nodes)
    ax.grid(True, axis="y", alpha=0.25)
    maybe_add_legend(ax, show_legend and len(labels) > 1)
    apply_axis_limits(ax, metric_key)
    if paper_mode:
        apply_paper_style(ax)

    metric_dir = out_dir / (
        f"{paper_metric_slug(metric_key)}_nodes"
        if paper_mode
        else f"{paper_metric_slug(metric_key)}_by_nodes"
    )
    metric_dir.mkdir(parents=True, exist_ok=True)
    sample_suffix = (
        f"_samples{parse_int_label(sample_size)}"
        if sample_size is not None
        else ""
    )
    if paper_mode:
        filename_prefix = (
            f"p_{paper_metric_slug(metric_key)}_nodes_graph_type_"
            f"{paper_noise_slug(noise_type)}_{paper_pretreatment_slug(pretreatment)}"
            f"{sample_suffix}"
        )
    else:
        filename_prefix = (
            f"{paper_metric_slug(metric_key)}_by_nodes_graph_type_"
            f"{slug(noise_type)}_pre{pretreatment}{sample_suffix}"
        )

    written = []
    for image_format in formats:
        suffix = image_format.lstrip(".")
        path = metric_dir / f"{filename_prefix}.{suffix}"
        fig.savefig(path, dpi=300)
        written.append(path)
    plt.close(fig)
    return written


def save_sample_size_comparison_plot(
    plt: Any,
    rows: list[dict[str, str]],
    metric_key: str,
    comparison_kind: str,
    n_nodes: str,
    out_dir: Path,
    formats: list[str],
    mb_method: str,
    lines: bool = False,
    paper_mode: bool = False,
    show_legend: bool = True,
) -> list[Path]:
    if not rows:
        return []

    metric = METRICS[metric_key]
    if comparison_kind == "graph_type":
        by_label = aggregate_metric_by_label(rows, metric_key, graph_type)
        label_title = "Graph type"
        title_part = "Acyclic vs Cyclic"
        if paper_mode:
            filename_context = (
                f"{paper_noise_slug(value_label(rows[0], 'noise_type'))}_"
                f"{paper_pretreatment_slug(value_label(rows[0], 'pretreatment'))}"
            )
        else:
            filename_context = (
                f"{slug(value_label(rows[0], 'noise_type'))}_pre"
                f"{value_label(rows[0], 'pretreatment')}"
            )
    elif comparison_kind == "noise_type":
        by_label = aggregate_metric_by_label(
            rows, metric_key, lambda row: value_label(row, "noise_type")
        )
        label_title = "Noise type"
        title_part = "Noise types"
        if paper_mode:
            filename_context = (
                f"{paper_graph_slug(graph_type(rows[0]))}_"
                f"{paper_pretreatment_slug(value_label(rows[0], 'pretreatment'))}"
            )
        else:
            filename_context = (
                f"{graph_type(rows[0])}_pre{value_label(rows[0], 'pretreatment')}"
            )
    elif comparison_kind == "scm_type":
        by_label = aggregate_metric_by_label(rows, metric_key, scm_type)
        label_title = "SCM type"
        title_part = "Linear vs Nonlinear"
        if paper_mode:
            filename_context = (
                f"{paper_graph_slug(graph_type(rows[0]))}_"
                f"{paper_noise_slug(value_label(rows[0], 'noise_type'))}_"
                f"{paper_pretreatment_slug(value_label(rows[0], 'pretreatment'))}"
            )
        else:
            filename_context = (
                f"{graph_type(rows[0])}_{slug(value_label(rows[0], 'noise_type'))}_"
                f"pre{value_label(rows[0], 'pretreatment')}"
            )
    else:
        raise ValueError(f"Unknown comparison kind: {comparison_kind}")

    if len(by_label) <= 1:
        return []

    fig, ax = plt.subplots(
        figsize=(9.0, 5.4) if paper_mode else (8.5, 5.2),
        constrained_layout=True,
    )
    labels = sorted(by_label)
    all_sample_sizes = sorted(
        {sample_size for series in by_label.values() for sample_size in series}
    )
    fixed_graph_kind = None if comparison_kind == "graph_type" else graph_type(rows[0])
    fixed_marker = None if fixed_graph_kind is None else graph_kind_marker(fixed_graph_kind)
    total_by_graph_type: dict[str, int] = defaultdict(int)
    if comparison_kind == "graph_type" and metric_key in GRAPH_TYPE_COLOR_METRICS:
        for label in labels:
            total_by_graph_type[label_graph_type(label)] += 1
    index_by_graph_type: dict[str, int] = defaultdict(int)
    for index, label in enumerate(labels):
        series = by_label[label]
        sample_sizes = sorted_sample_sizes(series)
        offset = x_offset(index, len(labels), all_sample_sizes)
        x_values = [sample_size + offset for sample_size in sample_sizes]
        means = [series[sample_size][0] for sample_size in sample_sizes]
        sds = [series[sample_size][1] for sample_size in sample_sizes]
        color = None
        if comparison_kind == "graph_type" and metric_key in GRAPH_TYPE_COLOR_METRICS:
            color = graph_type_color(
                plt, label, index_by_graph_type, total_by_graph_type
            )
        elif fixed_graph_kind is not None:
            color = graph_kind_color(plt, fixed_graph_kind, index, len(labels))
        ax.errorbar(
            x_values,
            means,
            yerr=errorbar_yerr(sds),
            marker=(
                graph_type_marker(label)
                if comparison_kind == "graph_type"
                else fixed_marker
            ),
            linestyle="-" if lines else "none",
            label=label.replace("_", " ").title(),
            color=color,
            **errorbar_style(paper_mode),
        )

    if not paper_mode:
        title = (
            f"{metric['title']} — {title_part} — {n_nodes} nodes — "
            f"{mb_method}"
        )
        ax.set_title(title)
    ax.set_xlabel("Sample size")
    ax.set_ylabel(metric["ylabel"])
    ax.set_xticks(all_sample_sizes)
    ax.set_xticklabels([parse_int_label(value) for value in all_sample_sizes])
    ax.grid(True, axis="y", alpha=0.25)
    maybe_add_legend(ax, show_legend)
    apply_axis_limits(ax, metric_key)
    if paper_mode:
        apply_paper_style(ax)

    comparison_slug = PAPER_COMPARISON_SLUGS.get(
        comparison_kind, slug(comparison_kind)
    )
    metric_dir = out_dir / f"{paper_metric_slug(metric_key)}_vs_{comparison_slug}"
    metric_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if paper_mode:
        filename_prefix = (
            f"p_{paper_metric_slug(metric_key)}_samples_vs_{comparison_slug}_"
            f"{n_nodes}n_{paper_method_slug(mb_method)}_{filename_context}"
        )
    else:
        filename_prefix = (
            f"{paper_metric_slug(metric_key)}_samples_vs_{comparison_slug}_"
            f"{n_nodes}n_{slug(mb_method)}_"
            f"{filename_context}"
        )
    for image_format in formats:
        suffix = image_format.lstrip(".")
        path = metric_dir / f"{filename_prefix}.{suffix}"
        fig.savefig(path, dpi=300)
        written.append(path)
    plt.close(fig)
    return written


def create_plots(
    rows: list[dict[str, str]],
    out_dir: Path,
    formats: list[str],
    lines: bool = False,
    comparison_mb_method: str = DEFAULT_COMPARISON_MB_METHOD,
    paper_mode: bool = False,
    show_legend: bool = True,
    one_mb_only: bool = False,
) -> list[Path]:
    plt = load_matplotlib()
    comparison_mb_method = comparison_mb_method.strip().lower()
    if one_mb_only:
        rows = [
            row
            for row in rows
            if value_label(row, "mb_method") == comparison_mb_method
        ]
        if not rows:
            return []

    written = []
    n_nodes_values = sorted(
        {parse_int_label(row["n_nodes"]) for row in rows},
        key=lambda value: parse_float(value),
    )
    graph_types = sorted({graph_type(row) for row in rows})
    noise_types = sorted({value_label(row, "noise_type") for row in rows})
    pretreatment_values = sorted({value_label(row, "pretreatment") for row in rows})

    for n_nodes in n_nodes_values:
        for graph_kind in graph_types:
            for noise_type in noise_types:
                for pretreatment in pretreatment_values:
                    subset = [
                        row
                        for row in rows
                        if parse_int_label(row["n_nodes"]) == n_nodes
                        and graph_type(row) == graph_kind
                        and value_label(row, "noise_type") == noise_type
                        and value_label(row, "pretreatment") == pretreatment
                    ]
                    if not subset:
                        continue
                    for metric_key in METRICS:
                        written.extend(
                            save_metric_plot(
                                plt,
                                subset,
                                metric_key,
                                graph_kind,
                                noise_type,
                                pretreatment,
                                n_nodes,
                                out_dir,
                                formats,
                                lines=lines,
                                paper_mode=paper_mode,
                                show_legend=show_legend,
                            )
                    )
    for graph_kind in graph_types:
        for noise_type in noise_types:
            for pretreatment in pretreatment_values:
                subset = [
                    row
                    for row in rows
                    if graph_type(row) == graph_kind
                    and value_label(row, "noise_type") == noise_type
                    and value_label(row, "pretreatment") == pretreatment
                ]
                if not subset:
                    continue
                for metric_key in NODE_SCALING_METRICS:
                    written.extend(
                        save_node_scaling_plot(
                            plt,
                            subset,
                            metric_key,
                            graph_kind,
                            noise_type,
                            pretreatment,
                            out_dir,
                            formats,
                            lines=lines,
                            paper_mode=paper_mode,
                            show_legend=show_legend,
                        )
                    )
    for noise_type in noise_types:
        for pretreatment in pretreatment_values:
            sample_15000_subset = [
                row
                for row in rows
                if value_label(row, "noise_type") == noise_type
                and value_label(row, "pretreatment") == pretreatment
                and parse_float(row.get("sample_size")) == NODE_SCALING_SAMPLE_SIZE
            ]
            if not sample_15000_subset:
                continue
            for metric_key in SAMPLE_15000_GRAPH_TYPE_NODE_SCALING_METRICS:
                written.extend(
                    save_graph_type_node_scaling_plot(
                        plt,
                        sample_15000_subset,
                        metric_key,
                        noise_type,
                        pretreatment,
                        out_dir,
                        formats,
                        lines=lines,
                        paper_mode=paper_mode,
                        show_legend=show_legend,
                        sample_size=NODE_SCALING_SAMPLE_SIZE,
                    )
                )
    comparison_rows = [
        row
        for row in rows
        if value_label(row, "mb_method") == comparison_mb_method
    ]
    if comparison_rows:
        for n_nodes in n_nodes_values:
            for noise_type in noise_types:
                for pretreatment in pretreatment_values:
                    graph_subset = [
                        row
                        for row in comparison_rows
                        if parse_int_label(row["n_nodes"]) == n_nodes
                        and value_label(row, "noise_type") == noise_type
                        and value_label(row, "pretreatment") == pretreatment
                    ]
                    for metric_key in COMPARISON_METRICS:
                        written.extend(
                            save_sample_size_comparison_plot(
                                plt,
                                graph_subset,
                                metric_key,
                                "graph_type",
                                n_nodes,
                                out_dir,
                                formats,
                                comparison_mb_method,
                                lines=lines,
                                paper_mode=paper_mode,
                                show_legend=show_legend,
                            )
                        )

            for graph_kind in graph_types:
                for pretreatment in pretreatment_values:
                    noise_subset = [
                        row
                        for row in comparison_rows
                        if parse_int_label(row["n_nodes"]) == n_nodes
                        and graph_type(row) == graph_kind
                        and value_label(row, "pretreatment") == pretreatment
                    ]
                    for metric_key in COMPARISON_METRICS:
                        written.extend(
                            save_sample_size_comparison_plot(
                                plt,
                                noise_subset,
                                metric_key,
                                "noise_type",
                                n_nodes,
                                out_dir,
                                formats,
                                comparison_mb_method,
                                lines=lines,
                                paper_mode=paper_mode,
                                show_legend=show_legend,
                            )
                        )
            for graph_kind in graph_types:
                for noise_type in noise_types:
                    for pretreatment in pretreatment_values:
                        scm_subset = [
                            row
                            for row in comparison_rows
                            if parse_int_label(row["n_nodes"]) == n_nodes
                            and graph_type(row) == graph_kind
                            and value_label(row, "noise_type") == noise_type
                            and value_label(row, "pretreatment") == pretreatment
                        ]
                        for metric_key in COMPARISON_METRICS:
                            written.extend(
                                save_sample_size_comparison_plot(
                                    plt,
                                    scm_subset,
                                    metric_key,
                                    "scm_type",
                                    n_nodes,
                                    out_dir,
                                    formats,
                                    comparison_mb_method,
                                    lines=lines,
                                    paper_mode=paper_mode,
                                    show_legend=show_legend,
                                )
                            )
    return written


def main() -> None:
    args = parse_args()
    out_dir = args.out or default_figure_dir()
    rows = read_summary(args.summary)
    written = create_plots(
        rows,
        out_dir,
        args.formats,
        lines=args.lines,
        comparison_mb_method=args.comparison_mb_method,
        paper_mode=args.paper_mode,
        show_legend=not args.no_legends,
        one_mb_only=args.one_mb_only,
    )
    print(f"Wrote {len(written)} figure file(s) to {out_dir}")


if __name__ == "__main__":
    main()
