#!/usr/bin/env python3
"""Summarise LSAS result CSVs into evaluation metrics.

Usage:
    python summarise_lsas_metrics.py input1.csv results_folder --out results_metric
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

ANALYSIS_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = ANALYSIS_DIR.parent
DATA_DIR = NEURIPS_DIR / "data"
DEFAULT_OUTPUT_DIR_NAME = "results_metric"
SUMMARY_FILENAME = "metrics_summary.csv"

nx: Any = None
is_valid_sigma_backdoor_set: Any = None
SigmaBackdoorChecker: Any = None


NA_VALUES = {"", "NA", "N/A", "NaN", "nan", "None", "none", "null", "NULL"}
COMPACT_FILENAME_RE = re.compile(
    r"^(?P<scm_type>linear|nonlinear)"
    r"_(?P<n_nodes>\d+)n"
    r"_(?P<noise_code>gau|non_gau|ran)"
    r"_(?P<cyclicity>acyclic|cyclic)"
    r"(?P<no_pre>_no_pre)?"
    r"_(?P<sample_k>\d+(?:\.\d+)?)Ks"
    r"_(?P<n_graphs>\d+)g$",
    flags=re.IGNORECASE,
)
NOISE_CODE_TO_TYPE = {
    "gau": "gaussian",
    "non_gau": "non_gaussian",
    "ran": "random",
}
SUMMARY_COLUMNS = [
    "dataset_name",
    "noise_type",
    "pretreatment",
    "sample_size",
    "n_graphs",
    "cyclicity",
    "graph_family",
    "scm_type",
    "n_nodes",
    "n_edges",
    "treatment",
    "outcome",
    "mb_method",
    "n_rows",
    "correct_edge_identification",
    "sd_correct_edge_identification",
    "empty_set_fraction",
    "mean_re",
    "sd_re",
    "mean_subset_validity",
    "sd_subset_validity",
    "mean_precision",
    "sd_precision",
    "mean_error_rate",
    "sd_error_rate",
    "mean_ntest",
    "sd_ntest",
    "mean_cpu_time",
    "sd_cpu_time",
    "mean_wall_time",
    "sd_wall_time",
]
PER_SAMPLE_COLUMNS = [
    "source_file",
    "row_number",
    "dataset_name",
    "noise_type",
    "pretreatment",
    "sample_size",
    "n_graphs",
    "cyclicity",
    "graph_family",
    "scm_type",
    "n_nodes",
    "n_edges",
    "treatment",
    "outcome",
    "mb_method",
    "estimated_ce",
    "true_ce",
    "relative_error",
    "selected_adjustment_set",
    "validity_indicators",
    "valid_adjustment_set",
    "has_treatment_outcome_edge",
    "correct_edge_delta",
    "empty_set_indicator",
    "subset_validity",
    "precision",
    "error_rate",
    "n_independence_tests",
    "cpu_runtime_seconds",
    "wall_runtime_seconds",
]
METADATA_COLUMNS = [
    "dataset_name",
    "noise_type",
    "pretreatment",
    "sample_size",
    "n_graphs",
    "cyclicity",
    "graph_family",
    "scm_type",
    "n_nodes",
    "n_edges",
    "treatment",
    "outcome",
]
METRIC_COLUMNS = {
    "re": "relative_error",
    "subset_validity": "subset_validity",
    "precision": "precision",
    "error_rate": "error_rate",
    "ntest": "n_independence_tests",
    "cpu_time": "cpu_runtime_seconds",
    "wall_time": "wall_runtime_seconds",
}


def raise_csv_field_size_limit() -> None:
    """Allow large JSON/list cells in LSAS result CSV files."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def likely_large_field(input_path: Path, line_number: int | None) -> str:
    """Return a best-effort description of the largest field on a CSV line."""
    if line_number is None or line_number < 2:
        return "field unknown"

    try:
        with input_path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
            for current_line, row in enumerate(csv.reader(handle), start=2):
                if current_line != line_number:
                    continue
                if not row:
                    return "field unknown"
                largest_index, largest_value = max(
                    enumerate(row), key=lambda item: len(item[1])
                )
                if largest_index < len(header):
                    field_name = header[largest_index]
                else:
                    field_name = f"field_index_{largest_index}"
                return f"{field_name} ({len(largest_value)} characters)"
    except Exception as exc:
        return f"field unknown; could not inspect line: {exc}"

    return "field unknown"


def is_missing(value: Any) -> bool:
    return value is None or str(value).strip() in NA_VALUES


def parse_list_cell(value: Any) -> list[Any]:
    """Parse JSON/Python list-like CSV cells, returning [] for empty cells."""
    if is_missing(value):
        return []
    if isinstance(value, list):
        return value

    text = str(value).strip()
    if not text:
        return []

    parsers = (json.loads, ast.literal_eval)
    for parser in parsers:
        try:
            parsed = parser(text)
            if parsed is None:
                return []
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
            return [parsed]
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue

    raise ValueError(f"Could not parse list-like value: {text!r}")


def parse_node_label(value: Any) -> Any:
    """Convert common node labels such as 'node_3' or '3' to integer 3."""
    if is_missing(value):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)

    text = str(value).strip().strip("\"'")
    match = re.fullmatch(r"node_(\d+)", text)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text


def parse_graph_structure(graph_structure: Any, n_nodes: Any = None) -> nx.DiGraph:
    """Parse graph strings like '{3->0,4->0,0->1}' into a DiGraph."""
    if is_missing(graph_structure):
        raise ValueError("graph_structure is empty")

    text = str(graph_structure).strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError(f"graph_structure must be wrapped in braces: {text!r}")

    graph = nx.DiGraph()
    if not is_missing(n_nodes):
        try:
            graph.add_nodes_from(range(int(float(str(n_nodes)))))
        except ValueError as exc:
            raise ValueError(f"Invalid n_nodes value {n_nodes!r}") from exc

    edge_text = text[1:-1].strip()
    if not edge_text:
        return graph

    for raw_edge in edge_text.split(","):
        edge = raw_edge.strip()
        match = re.fullmatch(r"(.+?)\s*->\s*(.+)", edge)
        if not match:
            raise ValueError(f"Invalid edge in graph_structure: {edge!r}")
        graph.add_edge(
            parse_node_label(match.group(1)), parse_node_label(match.group(2))
        )

    return graph


def load_runtime_dependencies() -> None:
    """Load dependencies needed for graph validation after argparse handles --help."""
    global nx, is_valid_sigma_backdoor_set, SigmaBackdoorChecker

    try:
        import networkx as networkx_module
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise SystemExit(
            "Missing dependency: networkx is required to parse graphs and validate "
            "sigma-backdoor sets."
        ) from exc

    if not DATA_DIR.exists():
        raise SystemExit(
            f"Cannot find data directory containing sigma_backdoor.py: {DATA_DIR}"
        )
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))

    try:
        from sigma_backdoor import (
            SigmaBackdoorChecker as checker_class,
            is_valid_sigma_backdoor_set as validator,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import SigmaBackdoorChecker and "
            f"is_valid_sigma_backdoor_set from {DATA_DIR / 'sigma_backdoor.py'}"
        ) from exc

    nx = networkx_module
    is_valid_sigma_backdoor_set = validator
    SigmaBackdoorChecker = checker_class


def to_float(value: Any, *, allow_nan: bool = True) -> float:
    if is_missing(value):
        return math.nan if allow_nan else 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value, got {value!r}") from exc


def estimate_ce(ce_estimates_for_adjustment_sets: Any) -> float:
    estimates = parse_list_cell(ce_estimates_for_adjustment_sets)
    if not estimates:
        return 0.0
    numeric_estimates = [to_float(value) for value in estimates]
    numeric_estimates = [value for value in numeric_estimates if not math.isnan(value)]
    if not numeric_estimates:
        return math.nan
    return mean(numeric_estimates)


def is_linear_scm(row: dict[str, Any]) -> bool:
    return str(row.get("scm_type", "")).strip().lower() == "linear"


def selected_adjustment_set_from_row(row: dict[str, Any]) -> list[Any]:
    """Choose Z_m from suggested_valid_adjustment_sets."""
    suggestions = parse_list_cell(row.get("suggested_valid_adjustment_sets"))
    if not suggestions:
        return []
    normalized = []
    for suggestion in suggestions:
        if isinstance(suggestion, dict):
            suggestion = suggestion.get("VAS", [])
        if isinstance(suggestion, (list, tuple, set)):
            normalized.append(list(suggestion))
        else:
            normalized.append([suggestion])
    return min(
        normalized,
        key=lambda values: (len(values), [str(value) for value in values]),
    )


def compute_relative_error(estimated_ce: float, true_ce: Any) -> float:
    true_value = to_float(true_ce)
    if math.isnan(estimated_ce) or math.isnan(true_value):
        return math.nan
    if true_value == 0:
        return math.nan
    return abs(estimated_ce - true_value) / abs(true_value)


def has_direct_edge(graph: Any, treatment: Any, outcome: Any) -> bool:
    return graph.has_edge(treatment, outcome)


def has_direct_edge_in_graph_structure(
    graph_structure: Any, treatment: Any, outcome: Any
) -> bool:
    """Check treatment->outcome directly in a compact graph string."""
    if is_missing(graph_structure):
        raise ValueError("graph_structure is empty")

    text = str(graph_structure).strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError(f"graph_structure must be wrapped in braces: {text!r}")

    edge_text = text[1:-1].strip()
    if not edge_text:
        return False

    for raw_edge in edge_text.split(","):
        edge = raw_edge.strip()
        match = re.fullmatch(r"(.+?)\s*->\s*(.+)", edge)
        if not match:
            raise ValueError(f"Invalid edge in graph_structure: {edge!r}")
        source = parse_node_label(match.group(1))
        target = parse_node_label(match.group(2))
        if source == treatment and target == outcome:
            return True
    return False


def update_correct_edge_count(has_edge: bool, shoudl_have_edge: bool) -> int:
    if has_edge and shoudl_have_edge:
        return 1
    if not has_edge and not shoudl_have_edge:
        return 1
    return 0


def get_sigma_backdoor_checker(
    row: dict[str, Any],
    treatment: Any,
    outcome: Any,
    checker_cache: dict[tuple[Any, ...], Any] | None,
) -> Any:
    """Return a cached checker for one graph/treatment/outcome tuple."""
    graph_key = (
        row.get("graph_structure"),
        row.get("n_nodes"),
        treatment,
        outcome,
    )
    if checker_cache is not None and graph_key in checker_cache:
        return checker_cache[graph_key]

    graph = parse_graph_structure(row.get("graph_structure"), row.get("n_nodes"))
    checker = SigmaBackdoorChecker(graph, treatment, outcome)
    if checker_cache is not None:
        checker_cache[graph_key] = checker
    return checker


def compute_adjustment_metrics(
    row: dict[str, Any],
    include_per_sample_details: bool = True,
    checker_cache: dict[tuple[Any, ...], Any] | None = None,
    use_fast_sigma_backdoor: bool = True,
) -> dict[str, Any]:
    adjustment_sets = json.loads(row["suggested_valid_adjustment_sets"])
    treatment = parse_node_label(row.get("treatment"))
    outcome = parse_node_label(row.get("outcome"))
    edge_exists = has_direct_edge_in_graph_structure(
        row.get("graph_structure"), treatment, outcome
    )

    true_ce = (
        to_float(row.get("true_ce")) if not is_missing(row.get("true_ce")) else 0.0
    )

    if adjustment_sets == 0:
        estimated_ce = estimate_ce(row.get("ce_estimates_for_adjustment_sets"))
        result = {
            "estimated_ce": estimated_ce,
            "relative_error": compute_relative_error(estimated_ce, true_ce),
            "has_treatment_outcome_edge": edge_exists,
            "correct_edge_delta": update_correct_edge_count(
                has_edge=edge_exists, shoudl_have_edge=False
            ),
            "empty_set_indicator": 0,
            "subset_validity": math.nan,
            "precision": math.nan,
            "error_rate": math.nan,
        }
        if include_per_sample_details:
            result.update(
                {
                    "selected_adjustment_set": json.dumps([]),
                    "validity_indicators": json.dumps([]),
                    "valid_adjustment_set": json.dumps([]),
                }
            )
        return result

    adjustment_sets = [
        [parse_node_label(value) for value in set] for set in adjustment_sets
    ]

    # Handle empty graph case
    if adjustment_sets == []:
        result = {
            "estimated_ce": math.nan,
            "relative_error": math.nan,
            "has_treatment_outcome_edge": True,
            "correct_edge_delta": 0,
            "empty_set_indicator": 1,
            "subset_validity": math.nan,
            "precision": math.nan,
            "error_rate": math.nan,
        }
        if include_per_sample_details:
            result.update(
                {
                    "selected_adjustment_set": json.dumps([]),
                    "validity_indicators": json.dumps([]),
                    "valid_adjustment_set": json.dumps([]),
                }
            )
        return result

    # If not empty nor nul, must be a list of sets.
    checker = get_sigma_backdoor_checker(
        row=row,
        treatment=treatment,
        outcome=outcome,
        checker_cache=checker_cache,
    )

    validity_indicators = []
    for va_set in adjustment_sets:
        is_valid = checker.is_valid(
            va_set,
            use_fast=use_fast_sigma_backdoor,
        )
        validity_indicators.append(1 if is_valid else 0)

    n_selected = len(validity_indicators)
    precision = sum(validity_indicators) / n_selected
    error_rate = sum(1 - indicator for indicator in validity_indicators) / n_selected
    estimated_ce = estimate_ce(row.get("ce_estimates_for_adjustment_sets"))

    result = {
        "estimated_ce": estimated_ce,
        "relative_error": compute_relative_error(estimated_ce, true_ce),
        "has_treatment_outcome_edge": edge_exists,
        "correct_edge_delta": update_correct_edge_count(
            has_edge=edge_exists, shoudl_have_edge=True
        ),
        "empty_set_indicator": 0,
        "subset_validity": 1.0 if all(validity_indicators) else 0.0,
        "precision": precision,
        "error_rate": error_rate,
    }
    if include_per_sample_details:
        result.update(
            {
                "selected_adjustment_set": json.dumps(adjustment_sets),
                "validity_indicators": json.dumps(validity_indicators),
                "valid_adjustment_set": True,
            }
        )
    return result


def clean_values(values: Iterable[Any]) -> list[float]:
    cleaned = []
    for value in values:
        number = to_float(value)
        if not math.isnan(number):
            cleaned.append(number)
    return cleaned


def mean_sd(values: Iterable[Any]) -> tuple[float, float]:
    cleaned = clean_values(values)
    if not cleaned:
        return math.nan, math.nan
    if len(cleaned) == 1:
        return cleaned[0], math.nan
    return mean(cleaned), stdev(cleaned)


def first_available(rows: list[dict[str, Any]], column: str) -> str:
    for row in rows:
        value = row.get(column)
        if not is_missing(value):
            return str(value)
    return ""


def noise_type_from_text(text: str) -> str:
    for noise_type in ["non_gaussian", "gaussian", "random"]:
        if noise_type in text:
            return noise_type
    return ""


def first_nonmissing(row: dict[str, Any], columns: list[str]) -> str:
    for column in columns:
        value = row.get(column)
        if not is_missing(value):
            return str(value)
    return ""


def sample_size_from_k_tag(value: str) -> str:
    sample_size = float(value) * 1000
    if sample_size.is_integer():
        return str(int(sample_size))
    return f"{sample_size:g}"


def metadata_from_filename(path: Path) -> dict[str, str]:
    stem = path.stem
    if stem.endswith("_lsas"):
        stem = stem[: -len("_lsas")]

    metadata = {
        "dataset_name": path.name,
        "noise_type": "",
        "pretreatment": (
            "false" if re.search(r"(?:^|_)d?no_pre(?:_|$)", stem) else "true"
        ),
        "sample_size": "",
        "n_graphs": "",
        "cyclicity": "",
        "graph_family": "",
        "scm_type": "",
        "n_nodes": "",
        "n_edges": "",
        "treatment": "",
        "outcome": "",
    }

    compact_match = COMPACT_FILENAME_RE.fullmatch(stem)
    if compact_match:
        noise_code = compact_match.group("noise_code").lower()
        cyclicity = compact_match.group("cyclicity").lower()
        metadata.update(
            {
                "noise_type": NOISE_CODE_TO_TYPE[noise_code],
                "pretreatment": ("false" if compact_match.group("no_pre") else "true"),
                "sample_size": sample_size_from_k_tag(compact_match.group("sample_k")),
                "n_graphs": compact_match.group("n_graphs"),
                "cyclicity": cyclicity,
                "graph_family": cyclicity,
                "scm_type": compact_match.group("scm_type").lower(),
                "n_nodes": compact_match.group("n_nodes"),
            }
        )
        return metadata

    metadata["noise_type"] = noise_type_from_text(stem.lower())

    match = re.search(r"_n(\d+)(?:_|$)", stem)
    if match:
        metadata["sample_size"] = match.group(1)
    match = re.search(r"_graphs(\d+)(?:_|$)", stem)
    if match:
        metadata["n_graphs"] = match.group(1)
    cyclicity_matches = re.findall(r"_(acyclic|cyclic)(?:_|$)", f"_{stem}_")
    if cyclicity_matches:
        metadata["graph_family"] = cyclicity_matches[-1]
        metadata["cyclicity"] = cyclicity_matches[-1]
    scm_type_matches = re.findall(r"_(linear|nonlinear)(?:_|$)", f"_{stem}_")
    if scm_type_matches:
        metadata["scm_type"] = scm_type_matches[-1]
    match = re.search(r"_nodes(\d+)(?:_|$)", stem)
    if match:
        metadata["n_nodes"] = match.group(1)
    match = re.search(r"_edges(\d+)(?:_|$)", stem)
    if match:
        metadata["n_edges"] = match.group(1)
    match = re.search(r"_t(\d+)(?:_|$)", stem)
    if match:
        metadata["treatment"] = match.group(1)
    match = re.search(r"_y(\d+)(?:_|$)", stem)
    if match:
        metadata["outcome"] = match.group(1)

    return metadata


def summarize_group(
    rows: list[dict[str, Any]], input_path: Path, mb_method: str
) -> dict[str, Any]:
    filename_metadata = metadata_from_filename(input_path)
    summary = {
        key: filename_metadata[key] or first_available(rows, key)
        for key in [
            "dataset_name",
            "noise_type",
            "pretreatment",
            "sample_size",
            "n_graphs",
            "cyclicity",
            "graph_family",
            "scm_type",
            "n_nodes",
            "n_edges",
            "treatment",
            "outcome",
        ]
    }
    if not summary["cyclicity"]:
        summary["cyclicity"] = summary["graph_family"]
    summary["mb_method"] = mb_method
    summary["n_rows"] = len(rows)
    n_rows = len(rows)
    if n_rows:
        empty_set_count = sum(
            to_float(row.get("empty_set_indicator"), allow_nan=False) for row in rows
        )
        correct_edge_values = [
            to_float(row.get("correct_edge_delta"), allow_nan=False)
            for row in rows
            if to_float(row.get("empty_set_indicator"), allow_nan=False) != 1
        ]

        normalised_count = n_rows - empty_set_count
        if normalised_count > 0:
            (
                summary["correct_edge_identification"],
                summary["sd_correct_edge_identification"],
            ) = mean_sd(correct_edge_values)
        else:
            summary["correct_edge_identification"] = math.nan
            summary["sd_correct_edge_identification"] = math.nan
        summary["empty_set_fraction"] = empty_set_count / n_rows
    else:
        summary["correct_edge_identification"] = math.nan
        summary["sd_correct_edge_identification"] = math.nan
        summary["empty_set_fraction"] = math.nan

    metric_columns = {
        "re": "relative_error",
        "subset_validity": "subset_validity",
        "precision": "precision",
        "error_rate": "error_rate",
        "ntest": "n_independence_tests",
        "cpu_time": "cpu_runtime_seconds",
        "wall_time": "wall_runtime_seconds",
    }
    for output_name, source_column in metric_columns.items():
        metric_values = [row.get(source_column) for row in rows]
        metric_mean, metric_sd = mean_sd(metric_values)
        summary[f"mean_{output_name}"] = metric_mean
        summary[f"sd_{output_name}"] = metric_sd

    return summary


def default_output_dir() -> Path:
    """Return the default output directory for the current launch location."""
    cwd = Path.cwd().resolve()
    if cwd == ANALYSIS_DIR:
        return Path(DEFAULT_OUTPUT_DIR_NAME)
    return ANALYSIS_DIR / DEFAULT_OUTPUT_DIR_NAME


def make_output_name(input_files: list[Path], output_dir: Path | None = None) -> Path:
    """Build the default summary filename inside the output directory."""
    output_dir = output_dir or default_output_dir()
    return output_dir / SUMMARY_FILENAME


def resolve_output_path(out_arg: Path | None, input_files: list[Path]) -> Path:
    """Treat --out as a CSV file when it ends in .csv, otherwise as a directory."""
    if out_arg is None:
        return make_output_name(input_files)
    if out_arg.suffix.lower() == ".csv":
        return out_arg
    return make_output_name(input_files, output_dir=out_arg)


def discover_input_files(paths: list[Path]) -> list[Path]:
    """Expand file and directory inputs into a sorted, de-duplicated CSV list."""
    discovered = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Input path does not exist: {path}")
        if path.is_dir():
            discovered.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*.csv")
                    if candidate.is_file()
                )
            )
        elif path.is_file():
            if path.suffix.lower() != ".csv":
                raise SystemExit(f"Input file is not a CSV: {path}")
            discovered.append(path)
        else:
            raise SystemExit(f"Input path is neither a file nor a directory: {path}")

    unique_files = []
    seen = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(path)

    if not unique_files:
        raise SystemExit("No CSV files found in the provided input path(s).")
    return sorted(unique_files)


def compact_processed_row(
    row: dict[str, Any],
    adjustment_metrics: dict[str, Any],
    filename_metadata: dict[str, str],
    filename_metadata_columns: list[str],
    input_path: Path,
    row_number: int,
    include_per_sample_details: bool,
) -> dict[str, Any]:
    """Keep only fields needed for summaries and optional per-sample output."""
    processed_row: dict[str, Any] = {}
    for column in filename_metadata_columns:
        filename_value = filename_metadata[column]
        processed_row[column] = (
            filename_value if filename_value else first_nonmissing(row, [column])
        )

    if not processed_row.get("cyclicity"):
        processed_row["cyclicity"] = processed_row.get("graph_family", "")

    for column in [
        "mb_method",
        "n_independence_tests",
        "cpu_runtime_seconds",
        "wall_runtime_seconds",
    ]:
        processed_row[column] = row.get(column, "")

    for column in [
        "relative_error",
        "correct_edge_delta",
        "empty_set_indicator",
        "subset_validity",
        "precision",
        "error_rate",
    ]:
        processed_row[column] = adjustment_metrics.get(column)

    if include_per_sample_details:
        for column in [
            "estimated_ce",
            "selected_adjustment_set",
            "validity_indicators",
            "valid_adjustment_set",
            "has_treatment_outcome_edge",
        ]:
            processed_row[column] = adjustment_metrics.get(column)
        processed_row["true_ce"] = row.get("true_ce", "")
        processed_row["source_file"] = str(input_path)
        processed_row["row_number"] = row_number

    return processed_row


class RunningStats:
    """Online sample mean and sample standard deviation."""

    def __init__(self) -> None:
        self.count = 0
        self.mean_value = 0.0
        self.m2 = 0.0

    def add(self, value: Any) -> None:
        number = to_float(value)
        if math.isnan(number):
            return
        self.count += 1
        delta = number - self.mean_value
        self.mean_value += delta / self.count
        delta2 = number - self.mean_value
        self.m2 += delta * delta2

    def mean_sd(self) -> tuple[float, float]:
        if self.count == 0:
            return math.nan, math.nan
        if self.count == 1:
            return self.mean_value, math.nan
        return self.mean_value, math.sqrt(self.m2 / (self.count - 1))


class SummaryAccumulator:
    """Aggregate one input file and MB method without storing per-row data."""

    def __init__(
        self, filename_metadata: dict[str, str], input_path: Path, mb_method: str
    ) -> None:
        self.filename_metadata = filename_metadata
        self.input_path = input_path
        self.mb_method = mb_method
        self.first_metadata: dict[str, str] = {}
        self.n_rows = 0
        self.correct_edge_count = 0.0
        self.empty_set_count = 0.0
        self.correct_edge_stats = RunningStats()
        self.stats = {output_name: RunningStats() for output_name in METRIC_COLUMNS}

    def add(self, row: dict[str, Any]) -> None:
        self.n_rows += 1
        for column in METADATA_COLUMNS:
            if self.filename_metadata[column] or column in self.first_metadata:
                continue
            value = row.get(column)
            if not is_missing(value):
                self.first_metadata[column] = str(value)

        empty_set_indicator = to_float(row.get("empty_set_indicator"), allow_nan=False)
        self.empty_set_count += empty_set_indicator
        if empty_set_indicator != 1:
            correct_edge_delta = to_float(
                row.get("correct_edge_delta"), allow_nan=False
            )
            self.correct_edge_count += correct_edge_delta
            self.correct_edge_stats.add(correct_edge_delta)

        for output_name, source_column in METRIC_COLUMNS.items():
            self.stats[output_name].add(row.get(source_column))

    def to_summary(self) -> dict[str, Any]:
        summary = {
            key: self.filename_metadata[key] or self.first_metadata.get(key, "")
            for key in METADATA_COLUMNS
        }
        if not summary["cyclicity"]:
            summary["cyclicity"] = summary["graph_family"]
        summary["mb_method"] = self.mb_method
        summary["n_rows"] = self.n_rows

        normalised_count = self.n_rows - self.empty_set_count
        if normalised_count > 0:
            summary["correct_edge_identification"] = (
                self.correct_edge_count / normalised_count
            )
            _, summary["sd_correct_edge_identification"] = (
                self.correct_edge_stats.mean_sd()
            )
        else:
            summary["correct_edge_identification"] = math.nan
            summary["sd_correct_edge_identification"] = math.nan
        summary["empty_set_fraction"] = (
            self.empty_set_count / self.n_rows if self.n_rows else math.nan
        )

        for output_name, stats in self.stats.items():
            metric_mean, metric_sd = stats.mean_sd()
            summary[f"mean_{output_name}"] = metric_mean
            summary[f"sd_{output_name}"] = metric_sd

        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more LSAS CSV files or directories searched recursively for CSVs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Summary CSV file or output directory. When a directory is given, "
            f"a generated CSV name is used. Default directory: {default_output_dir()}."
        ),
    )
    parser.add_argument(
        "--per-sample-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-sample metrics.",
    )
    parser.add_argument(
        "--no-fast-sigma-backdoor",
        action="store_true",
        help=(
            "Use the original all-simple-path sigma-backdoor validation instead "
            "of the cached fast reachability check."
        ),
    )
    return parser.parse_args()


def processed_rows_for_file(
    input_path: Path,
    include_per_sample_details: bool = False,
    use_fast_sigma_backdoor: bool = True,
) -> list[dict[str, Any]]:
    filename_metadata = metadata_from_filename(input_path)
    checker_cache: dict[tuple[Any, ...], Any] = {}
    try:
        with input_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "graph_structure",
                "treatment",
                "outcome",
                "mb_method",
                "suggested_valid_adjustment_sets",
                "ce_estimates_for_adjustment_sets",
                "true_ce",
                "n_independence_tests",
                "cpu_runtime_seconds",
                "wall_runtime_seconds",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{input_path} is missing required columns: "
                    f"{', '.join(sorted(missing))}"
                )

            processed_rows = []
            row_number = 1
            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error as exc:
                    line_number = getattr(reader, "line_num", None)
                    large_field = likely_large_field(input_path, line_number)
                    raise ValueError(
                        f"Failed to read {input_path} near CSV line "
                        f"{line_number}: {exc}. Largest visible field: "
                        f"{large_field}."
                    ) from exc

                row_number = reader.line_num
                try:
                    adjustment_metrics = compute_adjustment_metrics(
                        row,
                        include_per_sample_details=include_per_sample_details,
                        checker_cache=checker_cache,
                        use_fast_sigma_backdoor=use_fast_sigma_backdoor,
                    )
                    processed_row = compact_processed_row(
                        row=row,
                        adjustment_metrics=adjustment_metrics,
                        filename_metadata=filename_metadata,
                        filename_metadata_columns=METADATA_COLUMNS,
                        input_path=input_path,
                        row_number=row_number,
                        include_per_sample_details=include_per_sample_details,
                    )
                    processed_rows.append(processed_row)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to process {input_path} row {row_number}: {exc}"
                    ) from exc
    except csv.Error as exc:
        raise ValueError(f"Failed to read {input_path}: {exc}") from exc

    return processed_rows


def summarize_file(
    input_path: Path,
    include_per_sample_details: bool = False,
    per_sample_writer: csv.DictWriter | None = None,
    use_fast_sigma_backdoor: bool = True,
) -> list[dict[str, Any]]:
    filename_metadata = metadata_from_filename(input_path)
    accumulators: dict[str, SummaryAccumulator] = {}
    checker_cache: dict[tuple[Any, ...], Any] = {}
    try:
        with input_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "graph_structure",
                "treatment",
                "outcome",
                "mb_method",
                "suggested_valid_adjustment_sets",
                "ce_estimates_for_adjustment_sets",
                "true_ce",
                "n_independence_tests",
                "cpu_runtime_seconds",
                "wall_runtime_seconds",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{input_path} is missing required columns: "
                    f"{', '.join(sorted(missing))}"
                )

            row_number = 1
            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error as exc:
                    line_number = getattr(reader, "line_num", None)
                    large_field = likely_large_field(input_path, line_number)
                    raise ValueError(
                        f"Failed to read {input_path} near CSV line "
                        f"{line_number}: {exc}. Largest visible field: "
                        f"{large_field}."
                    ) from exc

                row_number = reader.line_num
                try:
                    adjustment_metrics = compute_adjustment_metrics(
                        row,
                        include_per_sample_details=include_per_sample_details,
                        checker_cache=checker_cache,
                        use_fast_sigma_backdoor=use_fast_sigma_backdoor,
                    )
                    processed_row = compact_processed_row(
                        row=row,
                        adjustment_metrics=adjustment_metrics,
                        filename_metadata=filename_metadata,
                        filename_metadata_columns=METADATA_COLUMNS,
                        input_path=input_path,
                        row_number=row_number,
                        include_per_sample_details=include_per_sample_details,
                    )

                    mb_method = str(processed_row.get("mb_method", ""))
                    if mb_method not in accumulators:
                        accumulators[mb_method] = SummaryAccumulator(
                            filename_metadata, input_path, mb_method
                        )
                    accumulators[mb_method].add(processed_row)

                    if per_sample_writer is not None:
                        per_sample_writer.writerow(processed_row)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to process {input_path} row {row_number}: {exc}"
                    ) from exc
    except csv.Error as exc:
        raise ValueError(f"Failed to read {input_path}: {exc}") from exc

    return [accumulators[mb_method].to_summary() for mb_method in sorted(accumulators)]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    raise_csv_field_size_limit()
    load_runtime_dependencies()
    input_files = discover_input_files(args.inputs)
    out_path = resolve_output_path(args.out, input_files)
    use_fast_sigma_backdoor = not args.no_fast_sigma_backdoor

    summary_rows = []
    if args.per_sample_out:
        args.per_sample_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_out.open("w", newline="") as handle:
            per_sample_writer = csv.DictWriter(
                handle, fieldnames=PER_SAMPLE_COLUMNS, extrasaction="ignore"
            )
            per_sample_writer.writeheader()
            for input_path in input_files:
                summary_rows.extend(
                    summarize_file(
                        input_path,
                        include_per_sample_details=True,
                        per_sample_writer=per_sample_writer,
                        use_fast_sigma_backdoor=use_fast_sigma_backdoor,
                    )
                )
    else:
        for input_path in input_files:
            summary_rows.extend(
                summarize_file(
                    input_path,
                    include_per_sample_details=False,
                    per_sample_writer=None,
                    use_fast_sigma_backdoor=use_fast_sigma_backdoor,
                )
            )

    write_csv(out_path, summary_rows, SUMMARY_COLUMNS)

    print(f"Wrote summary metrics: {out_path}")
    if args.per_sample_out:
        print(f"Wrote per-sample metrics: {args.per_sample_out}")


if __name__ == "__main__":
    main()
