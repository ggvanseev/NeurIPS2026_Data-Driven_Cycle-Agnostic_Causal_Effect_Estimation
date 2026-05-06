#!/usr/bin/env python3
"""Run LSAS on generated NeurIPS 2026 CSV datasets.

The flow is intentionally plain:

1. Find every CSV below ``NeurIPS_2026/data/generated``.
2. Treat each CSV as one dataset.
3. Split that dataset into graph blocks using the metadata rows.
4. For every graph block and Markov blanket method, run LSAS.
5. Write one ``*_lsas.csv`` file next to the matching results subfolder.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = ANALYSIS_DIR.parent
DEFAULT_DATA_ROOT = NEURIPS_DIR / "data" / "generated"
DEFAULT_RESULTS_DIR = ANALYSIS_DIR / "results"
DEFAULT_LOG_DIR = ANALYSIS_DIR / "logs"
LSAS_DIR = NEURIPS_DIR / "Li_et_al_neurips_2025"

if str(LSAS_DIR) not in sys.path:
    sys.path.insert(0, str(LSAS_DIR))


NODE_COLUMN_RE = re.compile(r"^node_(\d+)$")
NA_VALUES = ["NA", "NaN", "nan", "N/A", "", "None", "null"]
DEFAULT_MB_METHODS = ["TC", "IAMB", "Fast_IAMB", "HITON_MB"]

METADATA_COLUMNS = [
    "graph_id",
    "n_graphs",
    "graph_family",
    "allows_two_cycles",
    "x_singleton_scc_flag",
    "scm_type",
    "treatment",
    "outcome",
    "treatment_outcome_edge",
    "n_nodes",
    "n_edges",
    "n_visible_nodes",
    "true_ce",
    "has_sigma_backdoor",
    "sigma_backdoor_checked",
    "graph_structure",
]

OUTPUT_COLUMNS = [
    "source_file",
    "dataset_name",
    "dataset_group",
    "n_graphs_in_dataset",
    "graph_index",
    "graph_id",
    "sample_size",
    "graph_structure",
    "treatment",
    "outcome",
    "treatment_column",
    "outcome_column",
    "visible_node_columns",
    "mb_method",
    "max_k",
    "markov_blanket_treatment",
    "markov_blanket_outcome",
    "suggested_valid_adjustment_sets",
    "selected_adjustment_set",
    "n_adjustment_sets",
    "n_independence_tests",
    "cpu_runtime_seconds",
    "wall_runtime_seconds",
    "lsas_status",
    "ce_estimates_for_adjustment_sets",
    "error_message",
    *[
        column
        for column in METADATA_COLUMNS
        if column not in {"graph_id", "graph_structure", "treatment", "outcome"}
    ],
]

pd: Any = None
LSAS: Any = None


def load_runtime_dependencies() -> None:
    """Import analysis dependencies after argparse has handled ``--help``."""
    global pd, LSAS
    import pandas as pandas_module
    from LSAS import LSAS as LSAS_class

    pd = pandas_module
    LSAS = LSAS_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root searched recursively for generated CSV datasets.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root where per-dataset LSAS CSVs are written.",
    )
    parser.add_argument(
        "--mb-methods",
        nargs="+",
        default=DEFAULT_MB_METHODS,
        help="Markov blanket discovery methods to run for every graph.",
    )
    parser.add_argument(
        "--max-k",
        default="auto",
        help=(
            "Maximum conditioning-set size. With 'auto', use the number of "
            "visible covariates: all node columns except treatment and outcome."
        ),
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of datasets.",
    )
    parser.add_argument(
        "--limit-graphs",
        type=int,
        default=None,
        help="Optional smoke-test limit on graph blocks per dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs that already exist.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Run on a single CSV file, skipping dataset discovery.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DEFAULT_LOG_DIR / f"run_analysis_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Log file: %s", log_path)


def discover_datasets(data_root: Path) -> list[Path]:
    """Return all generated CSV files below ``data_root``."""
    return sorted(path for path in data_root.rglob("*.csv") if path.is_file())


def dataset_group(source_file: Path, data_root: Path) -> str:
    relative = source_file.relative_to(data_root)
    return relative.parts[0] if len(relative.parts) > 1 else ""


def output_path_for(source_file: Path, data_root: Path, results_dir: Path) -> Path:
    relative = source_file.relative_to(data_root)
    return results_dir / relative.parent / f"{source_file.stem}_lsas.csv"


def node_columns(frame: Any) -> list[str]:
    columns = [column for column in frame.columns if NODE_COLUMN_RE.fullmatch(column)]
    return sorted(
        columns, key=lambda column: int(NODE_COLUMN_RE.fullmatch(column).group(1))
    )


def first_metadata_value(row: Any, column: str) -> Any:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return value


def int_metadata_value(row: Any, column: str) -> int | None:
    value = first_metadata_value(row, column)
    if value is None:
        return None
    return int(float(value))


def graph_metadata(graph_block: Any) -> dict[str, Any]:
    first_row = graph_block.iloc[0]
    metadata = {
        column: first_metadata_value(first_row, column) for column in METADATA_COLUMNS
    }
    for column in [
        "graph_id",
        "n_graphs",
        "treatment",
        "outcome",
        "n_nodes",
        "n_edges",
        "n_visible_nodes",
    ]:
        metadata[column] = int_metadata_value(first_row, column)
    return metadata


def split_into_graphs(dataset: Any) -> list[Any]:
    """Split a dataset where each graph starts at a non-empty ``n_edges`` row."""
    if "n_edges" not in dataset.columns:
        raise ValueError("Missing required metadata column: n_edges")

    starts = dataset.index[dataset["n_edges"].notna()].tolist()
    if not starts:
        raise ValueError("No graph blocks found; n_edges is empty for every row")

    graphs = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(dataset)
        graphs.append(dataset.iloc[start:stop].copy())
    return graphs


def max_k_for_graph(
    raw_max_k: str, columns: list[str], treatment: str, outcome: str
) -> int:
    if raw_max_k != "auto":
        return int(raw_max_k)
    return len([column for column in columns if column not in {treatment, outcome}])


def normalise_adjustment_sets(raw_vas: Any) -> list[list[str]]:
    if raw_vas is None:
        return []
    elif raw_vas == 0:
        return 0

    adjustment_sets: list[list[str]] = []
    for entry in raw_vas:
        values = entry.get("VAS", []) if isinstance(entry, dict) else entry
        adjustment_sets.append([str(value) for value in values])
    return adjustment_sets


def selected_adjustment_set(adjustment_sets: list[list[str]]) -> list[str]:
    if not adjustment_sets:
        return []
    return min(adjustment_sets, key=lambda values: (len(values), values))


def run_lsas(
    sample: Any,
    treatment_column: str,
    outcome_column: str,
    mb_method: str,
    max_k: int,
) -> dict[str, Any]:
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    model = LSAS(
        data=sample,
        Tr_X=treatment_column,
        Out_Y=outcome_column,
        mb_method=mb_method,
        max_k=max_k,
    )
    raw_vas = model.find_A()

    cpu_runtime = time.process_time() - cpu_start
    wall_runtime = time.perf_counter() - wall_start

    adjustment_sets = normalise_adjustment_sets(raw_vas)

    # Calc CE estimate for valid adjustment sets, if any, and add to the output metadata.
    ce_estimates = []
    if adjustment_sets or adjustment_sets == 0:
        if adjustment_sets == 0:
            ce_estimates.append(0.0)
        else:
            for vas in adjustment_sets:
                Z = sample[vas] if vas else None
                ce = ce_ols(sample[treatment_column], sample[outcome_column], Z)
                ce_estimates.append(ce)

    return {
        "markov_blanket_treatment": json.dumps(model.mb_set.get(treatment_column, [])),
        "markov_blanket_outcome": json.dumps(model.mb_set.get(outcome_column, [])),
        "suggested_valid_adjustment_sets": json.dumps(adjustment_sets),
        "selected_adjustment_set": "",
        "n_adjustment_sets": (
            len(adjustment_sets) if isinstance(adjustment_sets, list) else ""
        ),
        "n_independence_tests": model._get_ci_num(),
        "cpu_runtime_seconds": cpu_runtime,
        "wall_runtime_seconds": wall_runtime,
        "lsas_status": "ok",
        "ce_estimates_for_adjustment_sets": (json.dumps(ce_estimates)),
        "error_message": "",
    }


def empty_lsas_result(error: Exception) -> dict[str, Any]:
    return {
        "markov_blanket_treatment": json.dumps([]),
        "markov_blanket_outcome": json.dumps([]),
        "suggested_valid_adjustment_sets": json.dumps([]),
        "selected_adjustment_set": json.dumps([]),
        "n_adjustment_sets": 0,
        "n_independence_tests": "",
        "cpu_runtime_seconds": "",
        "wall_runtime_seconds": "",
        "lsas_status": "error",
        "ce_estimates_for_adjustment_sets": json.dumps([]),
        "error_message": repr(error),
    }


def base_output_row(
    source_file: Path,
    data_root: Path,
    n_graphs: int,
    graph_index: int,
    metadata: dict[str, Any],
    columns: list[str],
    treatment_column: str,
    outcome_column: str,
    mb_method: str,
    max_k: int,
    sample_size: int,
) -> dict[str, Any]:
    row = {
        "source_file": source_file.relative_to(data_root).as_posix(),
        "dataset_name": source_file.name,
        "dataset_group": dataset_group(source_file, data_root),
        "n_graphs_in_dataset": n_graphs,
        "graph_index": graph_index,
        "sample_size": sample_size,
        "treatment_column": treatment_column,
        "outcome_column": outcome_column,
        "visible_node_columns": json.dumps(columns),
        "mb_method": mb_method,
        "max_k": max_k,
    }
    row.update(metadata)
    return row


import numpy as np


def ce_ols(T, Y, Z=None, add_intercept=True):
    """
    Estimate causal effect via OLS given a valid adjustment set Z.

    Parameters
    ----------
    T : array (n,)
        Treatment
    Y : array (n,)
        Outcome
    Z : array (n, k) or None
        Adjustment variables
    add_intercept : bool
        Whether to include intercept

    Returns
    -------
    tau_hat : float
        Estimated causal effect (coefficient on T)
    """
    T = np.asarray(T).reshape(-1, 1)
    Y = np.asarray(Y).reshape(-1, 1)

    if Z is None:
        X = T
    else:
        Z = np.asarray(Z)
        X = np.column_stack([T, Z])

    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])

    # OLS: beta = (X^T X)^{-1} X^T Y
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]

    # coefficient on T:
    tau_idx = 1 if add_intercept else 0
    return float(beta[tau_idx][0])


def map_unobserved_nodes(nodes: int) -> int:
    """Map the number of unobserved nodes to a fixed number.
    IDeally this would not be hardcoded, but for now we did this as quick ffix"""
    if nodes < 10:
        return 2
    elif nodes < 20:
        return 3
    elif nodes < 30:
        return 4
    elif nodes < 40:
        return 5
    elif nodes < 50:
        return 6
    elif nodes < 110:
        return 10
    elif nodes < 210:
        return 20
    elif nodes < 510:
        return 30
    else:
        return 40


def run_dataset(
    source_file: Path,
    data_root: Path,
    results_dir: Path,
    mb_methods: list[str],
    raw_max_k: str,
    limit_graphs: int | None = None,
    overwrite: bool = False,
) -> Path:
    output_path = output_path_for(source_file, data_root, results_dir)
    if output_path.exists() and not overwrite:
        logging.info("Skipping existing output: %s", output_path)
        return output_path

    dataset = pd.read_csv(source_file, na_values=NA_VALUES, keep_default_na=True)
    columns = node_columns(dataset)
    if not columns:
        raise ValueError(f"No node_<id> sample columns found in {source_file}")

    graph_blocks = split_into_graphs(dataset)
    n_graphs_in_dataset = len(graph_blocks)
    if limit_graphs is not None:
        graph_blocks = graph_blocks[:limit_graphs]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()

        for graph_index, graph_block in enumerate(graph_blocks):
            metadata = graph_metadata(graph_block)
            treatment_column = f"node_{metadata['treatment']}"
            outcome_column = f"node_{metadata['outcome']}"

            # Draw random nodes to make unobserved
            # Ensure that treatment and outcome are always observed,
            # and that the same columns are hidden for every graph in the dataset.
            unobserved_nodes = []
            n_hidden_nodes = map_unobserved_nodes(metadata["n_nodes"])
            while (
                len(unobserved_nodes) < n_hidden_nodes
            ):  # Adjust the number of unobserved nodes as needed
                random_node = np.random.choice(columns)
                if (
                    random_node not in {treatment_column, outcome_column}
                    and random_node not in unobserved_nodes
                ):
                    unobserved_nodes.append(random_node)

            visible_columns = [col for col in columns if col not in unobserved_nodes]

            sample = graph_block[visible_columns].apply(pd.to_numeric, errors="coerce")

            for mb_method in mb_methods:
                max_k = max_k_for_graph(
                    raw_max_k, visible_columns, treatment_column, outcome_column
                )
                row = base_output_row(
                    source_file=source_file,
                    data_root=data_root,
                    n_graphs=n_graphs_in_dataset,
                    graph_index=graph_index,
                    metadata=metadata,
                    columns=visible_columns,
                    treatment_column=treatment_column,
                    outcome_column=outcome_column,
                    mb_method=mb_method,
                    max_k=max_k,
                    sample_size=len(sample),
                )

                try:
                    if (
                        treatment_column not in sample.columns
                        or outcome_column not in sample.columns
                    ):
                        raise ValueError(
                            f"Missing treatment/outcome columns: "
                            f"{treatment_column}, {outcome_column}"
                        )
                    row.update(
                        run_lsas(
                            sample, treatment_column, outcome_column, mb_method, max_k
                        )
                    )
                except Exception as exc:
                    row.update(empty_lsas_result(exc))
                    logging.exception(
                        "LSAS failed for %s graph=%s method=%s",
                        source_file,
                        graph_index,
                        mb_method,
                    )

                writer.writerow(row)

    logging.info("Wrote %s", output_path)
    return output_path


def main() -> int:
    args = parse_args()
    setup_logging()
    load_runtime_dependencies()

    data_root = args.data_root.resolve()
    results_dir = args.results_dir.resolve()

    if args.file is not None:
        datasets = [args.file.resolve()]
        logging.info("Running on single file: %s", datasets[0])
    else:
        datasets = discover_datasets(data_root)
        if args.limit_files is not None:
            datasets = datasets[: args.limit_files]
        logging.info("Found %s datasets under %s", len(datasets), data_root)
    for source_file in datasets:
        try:
            run_dataset(
                source_file=source_file,
                data_root=data_root,
                results_dir=results_dir,
                mb_methods=args.mb_methods,
                raw_max_k=args.max_k,
                limit_graphs=args.limit_graphs,
                overwrite=args.overwrite,
            )
        except Exception:
            logging.exception("Dataset failed: %s", source_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
