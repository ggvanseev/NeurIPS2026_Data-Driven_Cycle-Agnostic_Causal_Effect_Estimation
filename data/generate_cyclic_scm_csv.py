"""Export sampled SCM data to CSV.

This utility generates one SCM instance, draws ``N`` observational
samples, estimates the true causal effect for a designated treatment/outcome
pair, and stores visible-node sample values in a CSV file.

The exporter writes one cohesive sample per row, with visible-node values
written into consecutive columns. Graph-level metadata such as whether the
graph is cyclic or acyclic and the treatment/outcome CE are stored in CSV
columns rather than encoded in the filename.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from calc_CE import ce_linear_scm
from synthetic_cyclic_scm import generate_synthetic_scm_linear
from synthetic_cyclic_scm_nonlinear import generate_synthetic_scm_nonlinear


@dataclass(frozen=True)
class GeneratedSCM:
    """Bundle a generated SCM instance and its sampled data."""

    scm_type: str
    W: np.ndarray
    samples_full: np.ndarray
    observed_nodes: np.ndarray


def _validate_positive_int(value: int, name: str) -> None:
    """Raise if ``value`` is not a strictly positive integer-like quantity."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_nonnegative_int(value: int, name: str) -> None:
    """Raise if ``value`` is not a non-negative integer-like quantity."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def _validate_node_index(node: int, n_nodes: int, name: str) -> None:
    """Raise if ``node`` is outside the valid node-index range ``[0, n_nodes)``."""
    if not isinstance(node, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(node).__name__}.")
    if not 0 <= int(node) < n_nodes:
        raise ValueError(f"{name} must be in [0, {n_nodes}), got {node}.")


def _observed_edges(W: np.ndarray, observed_nodes: np.ndarray) -> list[tuple[int, int]]:
    """Return observed directed edges as ``(parent, child)`` tuples.

    An edge is considered observed when both of its endpoint nodes belong to
    the sampled observed-node subset.
    """
    observed_set = set(int(node) for node in observed_nodes)
    edges: list[tuple[int, int]] = []
    n_nodes = W.shape[0]
    for child in range(n_nodes):
        for parent in range(n_nodes):
            if W[child, parent] == 0.0:
                continue
            if parent in observed_set and child in observed_set:
                edges.append((parent, child))
    return edges


def _visible_nodes(
    observed_nodes: np.ndarray,
    treatment: int,
    outcome: int,
) -> np.ndarray:
    """Return visible nodes, always including treatment and outcome."""
    visible = np.unique(
        np.concatenate(
            [
                np.asarray(observed_nodes, dtype=int),
                np.array([treatment, outcome], dtype=int),
            ]
        )
    )
    return np.sort(visible)


def _format_graph_structure(W: np.ndarray) -> str:
    """Serialize the weighted graph structure into a single CSV cell.

    The format is a set-like string of directed edges:
    ``{parent->child,...}``.
    """
    edges: list[str] = []
    n_nodes = W.shape[0]
    for child in range(n_nodes):
        for parent in np.flatnonzero(W[child] != 0.0):
            edges.append(f"{int(parent)}->{int(child)}")
    return "{" + ",".join(edges) + "}"


def estimate_true_ce(
    W: np.ndarray,
    scm_type: str,
    treatment: int,
    outcome: int,
    cyclic_flag: bool,
    intervention_low: float,
    intervention_high: float,
) -> float | str:
    """Return the exported true causal effect value."""
    n_nodes = W.shape[0]
    _validate_node_index(treatment, n_nodes, "treatment")
    _validate_node_index(outcome, n_nodes, "outcome")

    if scm_type == "linear":
        if not cyclic_flag:
            return float(W[outcome, treatment])
        return float(
            ce_linear_scm(
                W=W,
                treatment_idx=treatment,
                outcome_idx=outcome,
                t1=intervention_high,
                t0=intervention_low,
            )
        )
    if scm_type == "nonlinear":
        return "NA"
    raise ValueError(f"Unsupported scm_type {scm_type!r}.")


def _effect_output_path(
    output_path: str | Path,
    cyclic_flag: bool,
    scm_type: str,
    n_graphs: int,
    n_samples: int,
    n_nodes: int,
    n_edges: int,
    treatment: int,
    outcome: int,
) -> Path:
    """Append dataset metadata to the requested output path."""
    output_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    graph_family = "cyclic" if cyclic_flag else "acyclic"
    dataset_tag = (
        f"_{graph_family}"
        f"_{scm_type}"
        f"_graphs{n_graphs}"
        f"_n{n_samples}"
        f"_nodes{n_nodes}"
        f"_edges{n_edges}"
        f"_t{treatment}"
        f"_y{outcome}"
        f"_{timestamp}"
    )
    return output_path.with_name(
        f"{output_path.stem}{dataset_tag}{output_path.suffix or '.csv'}"
    )


def _generate_linear_dataset(seed: int, **kwargs: object) -> GeneratedSCM:
    data = generate_synthetic_scm_linear(rng_seed=seed, **kwargs)
    return GeneratedSCM(
        scm_type="linear",
        W=data.W,
        samples_full=data.samples_full,
        observed_nodes=data.observed_nodes,
    )


def _generate_nonlinear_dataset(seed: int, **kwargs: object) -> GeneratedSCM:
    data = generate_synthetic_scm_nonlinear(rng_seed=seed, **kwargs)
    return GeneratedSCM(
        scm_type="nonlinear",
        W=data.W,
        samples_full=data.samples_full,
        observed_nodes=data.observed_nodes,
    )


def generate_scm_dataframe(
    output_path: str | Path,
    scm_type: str,
    n_nodes: int,
    n_edges: int,
    n_samples: int,
    n_observed: int,
    n_graphs: int = 1,
    treatment: int = 0,
    outcome: int = 1,
    sigma_range: tuple[float, float] | list[float] = (1.0, 1.0),
    noise_type: str = "gaussian",
    target_operator_norm: float = 0.8,
    rng_seed: Optional[int] = None,
    n_mc_samples: int = 5000,
    intervention_low: float = 0.0,
    intervention_high: float = 1.0,
    pretreatment_flag: bool = True,
    treatment_outcome_edge: bool = True,
    cyclic_flag: bool = True,
    allow_two_cycles: bool = False,
    x_singleton_scc_flag: bool = True,
    require_weak_connectivity: bool = True,
    sigma_backdoor_flag: bool = False,
    require_observed_edge: bool = True,
    max_generation_attempts: int = 200,
    debug: bool = False,
    target_radius: Optional[float] = None,
) -> pd.DataFrame:
    """Generate an SCM and export visible-node sample data to CSV.

    The exported table uses one row per observational sample. Metadata such as
    the treatment, outcome, total number of nodes, total number of edges, and
    graph structure are repeated on each row, while visible-node values are
    stored in consecutive columns named ``node_<index>``. The visible-node set
    always includes the treatment and outcome nodes.
    """
    _validate_positive_int(n_nodes, "n_nodes")
    _validate_nonnegative_int(n_edges, "n_edges")
    _validate_positive_int(n_samples, "n_samples")
    _validate_nonnegative_int(n_observed, "n_observed")
    _validate_positive_int(n_graphs, "n_graphs")
    _validate_positive_int(n_mc_samples, "n_mc_samples")
    _validate_positive_int(max_generation_attempts, "max_generation_attempts")
    _validate_node_index(treatment, n_nodes, "treatment")
    _validate_node_index(outcome, n_nodes, "outcome")

    if treatment == outcome:
        raise ValueError("treatment and outcome must be distinct.")
    if n_observed > n_nodes:
        raise ValueError(
            f"n_observed cannot exceed n_nodes; got {n_observed} > {n_nodes}."
        )
    if target_radius is not None:
        target_operator_norm = target_radius

    generator = np.random.default_rng(rng_seed)
    requested_parameters = {
        "output_path": str(output_path),
        "scm_type": scm_type,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_samples": n_samples,
        "n_observed": n_observed,
        "n_graphs": n_graphs,
        "treatment": treatment,
        "outcome": outcome,
        "sigma_range": [float(sigma_range[0]), float(sigma_range[1])],
        "noise_type": noise_type,
        "target_operator_norm": target_operator_norm,
        "rng_seed": rng_seed,
        "n_mc_samples": n_mc_samples,
        "intervention_low": intervention_low,
        "intervention_high": intervention_high,
        "pretreatment_flag": pretreatment_flag,
        "treatment_outcome_edge": treatment_outcome_edge,
        "cyclic_flag": cyclic_flag,
        "allow_two_cycles": allow_two_cycles,
        "x_singleton_scc_flag": x_singleton_scc_flag,
        "require_weak_connectivity": require_weak_connectivity,
        "sigma_backdoor_flag": sigma_backdoor_flag,
        "require_observed_edge": require_observed_edge,
        "max_generation_attempts": max_generation_attempts,
        "debug": debug,
    }
    generation_kwargs = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_samples": n_samples,
        "n_observed": n_observed,
        "target_operator_norm": target_operator_norm,
        "sigma_range": sigma_range,
        "noise_type": noise_type,
    }

    dataset_builder: Callable[..., GeneratedSCM]
    if scm_type == "linear":
        dataset_builder = _generate_linear_dataset
        generation_kwargs["enforce_operator_norm_condition"] = True
        generation_kwargs["pretreatment_flag"] = pretreatment_flag
        generation_kwargs["treatment"] = treatment
        generation_kwargs["outcome"] = outcome
        generation_kwargs["treatment_outcome_edge"] = treatment_outcome_edge
        generation_kwargs["cyclic_flag"] = cyclic_flag
        generation_kwargs["allow_two_cycles"] = allow_two_cycles
        generation_kwargs["x_singleton_scc_flag"] = x_singleton_scc_flag
        generation_kwargs["require_weak_connectivity"] = require_weak_connectivity
        generation_kwargs["sigma_backdoor_flag"] = sigma_backdoor_flag
    elif scm_type == "nonlinear":
        dataset_builder = _generate_nonlinear_dataset
        generation_kwargs["pretreatment_flag"] = pretreatment_flag
        generation_kwargs["treatment"] = treatment
        generation_kwargs["outcome"] = outcome
        generation_kwargs["treatment_outcome_edge"] = treatment_outcome_edge
        generation_kwargs["cyclic_flag"] = cyclic_flag
        generation_kwargs["allow_two_cycles"] = allow_two_cycles
        generation_kwargs["x_singleton_scc_flag"] = x_singleton_scc_flag
        generation_kwargs["require_weak_connectivity"] = require_weak_connectivity
        generation_kwargs["sigma_backdoor_flag"] = sigma_backdoor_flag
    else:
        raise ValueError("scm_type must be either 'linear' or 'nonlinear'.")

    if require_observed_edge and (n_observed < 2 or n_edges == 0):
        require_observed_edge = False

    final_output_path = _effect_output_path(
        output_path=output_path,
        cyclic_flag=cyclic_flag,
        scm_type=scm_type,
        n_graphs=n_graphs,
        n_samples=n_samples,
        n_nodes=n_nodes,
        n_edges=n_edges,
        treatment=treatment,
        outcome=outcome,
    )
    requested_parameters_text = json.dumps(requested_parameters, sort_keys=True)

    rows: list[dict[str, float | int | str]] = []
    true_ce_values: list[float | str] = []
    for graph_id in range(n_graphs):
        dataset: Optional[GeneratedSCM] = None
        visible_nodes: Optional[np.ndarray] = None
        for _ in range(max_generation_attempts):
            seed = int(generator.integers(0, np.iinfo(np.int32).max))
            candidate = dataset_builder(seed=seed, **generation_kwargs)
            candidate_visible_nodes = _visible_nodes(
                candidate.observed_nodes,
                treatment=treatment,
                outcome=outcome,
            )
            candidate_observed_edges = _observed_edges(
                candidate.W,
                candidate_visible_nodes,
            )
            if not require_observed_edge or candidate_observed_edges:
                dataset = candidate
                visible_nodes = candidate_visible_nodes
                break

        if dataset is None or visible_nodes is None:
            raise RuntimeError(
                "Failed to generate graph "
                f"{graph_id} with at least one observed edge within "
                f"{max_generation_attempts} attempts."
            )

        true_ce = estimate_true_ce(
            W=dataset.W,
            scm_type=scm_type,
            treatment=treatment,
            outcome=outcome,
            cyclic_flag=cyclic_flag,
            intervention_low=intervention_low,
            intervention_high=intervention_high,
        )
        true_ce_values.append(true_ce)
        graph_structure_text = _format_graph_structure(dataset.W)

        for sample_id, sample in enumerate(dataset.samples_full):
            is_graph_metadata_row = sample_id == 0
            row: dict[str, float | int | str] = {
                "graph_id": graph_id,
                "sample_id": sample_id,
                "n_graphs": n_graphs if is_graph_metadata_row else "NA",
                "graph_family": (
                    ("cyclic" if cyclic_flag else "acyclic")
                    if is_graph_metadata_row
                    else "NA"
                ),
                "allows_two_cycles": (
                    allow_two_cycles if is_graph_metadata_row else "NA"
                ),
                "x_singleton_scc_flag": (
                    x_singleton_scc_flag if is_graph_metadata_row else "NA"
                ),
                "scm_type": scm_type if is_graph_metadata_row else "NA",
                "treatment": treatment if is_graph_metadata_row else "NA",
                "outcome": outcome if is_graph_metadata_row else "NA",
                "treatment_outcome_edge": (
                    treatment_outcome_edge if is_graph_metadata_row else "NA"
                ),
                "n_nodes": n_nodes if is_graph_metadata_row else "NA",
                "n_edges": n_edges if is_graph_metadata_row else "NA",
                "n_visible_nodes": (
                    int(len(visible_nodes)) if is_graph_metadata_row else "NA"
                ),
                "true_ce": true_ce if is_graph_metadata_row else "NA",
                "has_sigma_backdoor": (
                    True if is_graph_metadata_row and sigma_backdoor_flag else "NA"
                ),
                "sigma_backdoor_checked": (
                    sigma_backdoor_flag if is_graph_metadata_row else "NA"
                ),
                "graph_structure": (
                    graph_structure_text if is_graph_metadata_row else "NA"
                ),
            }
            if debug:
                row["debug"] = True if is_graph_metadata_row else "NA"
                row["requested_parameters"] = (
                    requested_parameters_text if is_graph_metadata_row else "NA"
                )
            for visible_node in visible_nodes:
                row[f"node_{int(visible_node)}"] = float(sample[int(visible_node)])
            rows.append(row)

    frame = pd.DataFrame(rows)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(final_output_path, index=False)
    frame.attrs["output_path"] = str(final_output_path)
    frame.attrs["true_ce"] = true_ce_values[0]
    frame.attrs["true_ce_by_graph"] = true_ce_values
    return frame


generate_cyclic_scm_edge_dataframe = generate_scm_dataframe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SCM samples and export one row per sample to CSV."
    )
    parser.add_argument("output_path", type=Path, help="Destination CSV file.")
    parser.add_argument(
        "--scm-type",
        choices=("linear", "nonlinear"),
        default="linear",
        help="Type of SCM to generate.",
    )
    parser.add_argument("--n-nodes", type=int, required=True, help="Number of nodes.")
    parser.add_argument("--n-edges", type=int, required=True, help="Number of edges.")
    parser.add_argument(
        "--n-samples", type=int, required=True, help="Number of observational samples."
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=1,
        help="Number of independently generated graphs to append to the CSV.",
    )
    parser.add_argument(
        "--n-observed",
        type=int,
        required=True,
        help="Number of observed nodes.",
    )
    parser.add_argument(
        "--treatment",
        type=int,
        default=0,
        help="Treatment node index used for the true causal effect.",
    )
    parser.add_argument(
        "--outcome",
        type=int,
        default=1,
        help="Outcome node index used for the true causal effect.",
    )
    parser.add_argument(
        "--sigma-low",
        type=float,
        default=1.0,
        help="Lower bound of the per-node sigma sampling range.",
    )
    parser.add_argument(
        "--sigma-high",
        type=float,
        default=1.0,
        help="Upper bound of the per-node sigma sampling range.",
    )
    parser.add_argument(
        "--noise-type",
        choices=("gaussian", "non_gaussian", "random"),
        default="gaussian",
        help="Noise-family assignment across nodes: all Gaussian, all uniform, or random per node.",
    )
    parser.add_argument(
        "--target-operator-norm",
        "--target-radius",
        dest="target_operator_norm",
        type=float,
        default=0.8,
        help="Target operator norm used when rescaling the graph.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-mc-samples",
        type=int,
        default=5000,
        help="Monte Carlo sample size for estimating the true causal effect.",
    )
    parser.add_argument(
        "--intervention-low",
        type=float,
        default=0.0,
        help="Lower intervention value for the causal effect.",
    )
    parser.add_argument(
        "--intervention-high",
        type=float,
        default=1.0,
        help="Higher intervention value for the causal effect.",
    )
    sink_group = parser.add_mutually_exclusive_group()
    sink_group.add_argument(
        "--pretreatment-flag",
        dest="pretreatment_flag",
        action="store_true",
        help="Enforce the special treatment/outcome pretreatment structure.",
    )
    sink_group.add_argument(
        "--no-pretreatment-flag",
        dest="pretreatment_flag",
        action="store_false",
        help="Disable the special treatment/outcome pretreatment structure.",
    )
    parser.set_defaults(pretreatment_flag=True)
    treatment_outcome_group = parser.add_mutually_exclusive_group()
    treatment_outcome_group.add_argument(
        "--treatment-outcome-edge",
        dest="treatment_outcome_edge",
        action="store_true",
        help="Enforce a direct edge from the treatment to the outcome.",
    )
    treatment_outcome_group.add_argument(
        "--no-treatment-outcome-edge",
        dest="treatment_outcome_edge",
        action="store_false",
        help="Forbid a direct edge from the treatment to the outcome.",
    )
    parser.set_defaults(treatment_outcome_edge=True)
    cyclic_group = parser.add_mutually_exclusive_group()
    cyclic_group.add_argument(
        "--cyclic",
        dest="cyclic_flag",
        action="store_true",
        help="Generate from the existing cyclic graph construction.",
    )
    cyclic_group.add_argument(
        "--acyclic",
        dest="cyclic_flag",
        action="store_false",
        help="Generate an acyclic graph by using a strict lower-triangular DAG order and mapping back to the original node labels.",
    )
    parser.set_defaults(cyclic_flag=True)
    parser.add_argument(
        "--allow-two-cycles",
        action="store_true",
        help="Allow reciprocal edge pairs in cyclic graphs, enabling directed cycles of length two.",
    )
    x_scc_group = parser.add_mutually_exclusive_group()
    x_scc_group.add_argument(
        "--x-singleton-scc",
        dest="x_singleton_scc",
        action="store_true",
        help="Require the treatment node to have a singleton SCC.",
    )
    x_scc_group.add_argument(
        "--x-non-singleton-scc",
        dest="x_singleton_scc",
        action="store_false",
        help="Require the treatment node to belong to a non-singleton SCC.",
    )
    parser.set_defaults(x_singleton_scc=True)
    weak_group = parser.add_mutually_exclusive_group()
    weak_group.add_argument(
        "--require-weak-connectivity",
        dest="require_weak_connectivity",
        action="store_true",
        help="Enforce weak connectivity of the generated graph.",
    )
    weak_group.add_argument(
        "--allow-disconnected",
        dest="require_weak_connectivity",
        action="store_false",
        help="Allow the generated graph to have multiple weakly disconnected components.",
    )
    parser.set_defaults(require_weak_connectivity=True)
    sigma_group = parser.add_mutually_exclusive_group()
    sigma_group.add_argument(
        "--sigma-backdoor",
        dest="sigma_backdoor_flag",
        action="store_true",
        help="Require the generated graph to admit at least one valid sigma-backdoor set.",
    )
    sigma_group.add_argument(
        "--no-sigma-backdoor",
        dest="sigma_backdoor_flag",
        action="store_false",
        help="Skip sigma-backdoor checking during graph generation.",
    )
    parser.set_defaults(sigma_backdoor_flag=False)
    parser.add_argument(
        "--allow-zero-observed-edges",
        action="store_true",
        help="Allow export even if the visible-node subgraph has no observed edges.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Store the full requested parameter set in extra CSV metadata columns.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    frame = generate_scm_dataframe(
        output_path=args.output_path,
        scm_type=args.scm_type,
        n_nodes=args.n_nodes,
        n_edges=args.n_edges,
        n_samples=args.n_samples,
        n_observed=args.n_observed,
        n_graphs=args.n_graphs,
        treatment=args.treatment,
        outcome=args.outcome,
        sigma_range=(args.sigma_low, args.sigma_high),
        noise_type=args.noise_type,
        target_operator_norm=args.target_operator_norm,
        rng_seed=args.rng_seed,
        n_mc_samples=args.n_mc_samples,
        intervention_low=args.intervention_low,
        intervention_high=args.intervention_high,
        pretreatment_flag=args.pretreatment_flag,
        treatment_outcome_edge=args.treatment_outcome_edge,
        cyclic_flag=args.cyclic_flag,
        allow_two_cycles=args.allow_two_cycles,
        x_singleton_scc_flag=args.x_singleton_scc,
        require_weak_connectivity=args.require_weak_connectivity,
        sigma_backdoor_flag=args.sigma_backdoor_flag,
        require_observed_edge=not args.allow_zero_observed_edges,
        debug=args.debug,
    )

    print(f"Wrote {len(frame)} rows to {frame.attrs['output_path']}.")
    print(f"Graphs: {args.n_graphs}")
    print(f"Visible nodes: {frame['n_visible_nodes'].iloc[0]}")
    print(
        "Tracked treatment/outcome pair: "
        f"{frame['treatment'].iloc[0]} -> {frame['outcome'].iloc[0]}"
    )
    true_ce = frame.attrs["true_ce"]
    if isinstance(true_ce, (int, float, np.floating)):
        print(f"True CE: {true_ce:.6f}")
    else:
        print(f"True CE: {true_ce}")


if __name__ == "__main__":
    main()

    # Execute example:
    # python3 generate_cyclic_scm_csv.py generated/output.csv --scm-type linear --n-nodes 8 --n-edges 12 --n-samples 1000 --n-observed 5 --treatment 0 --outcome 1 --sigma-low 0.5 --sigma-high 1.0 --noise-type gaussian --cyclic --require-weak-connectivity --rng-seed 42
