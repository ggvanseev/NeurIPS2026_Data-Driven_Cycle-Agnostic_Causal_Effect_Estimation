"""Utilities for generating synthetic data from a nonlinear SCM.

This module deliberately reuses the graph-construction logic from
``synthetic_cyclic_scm.py`` so that the linear and nonlinear generators obey
the same rules for:

1. edge orientation and adjacency-matrix semantics,
2. weak connectivity,
3. the optional treatment/outcome pretreatment constraint, and
4. optional rejection of acyclic draws when a directed cycle is required.

The only nonlinear-specific part is the sampling model. Here the equilibrium
equations are

    X = tanh(W X) + U,

where ``W[i, j] != 0`` denotes the directed edge ``j -> i`` and ``U`` is
Gaussian noise. After generating a graph skeleton, the weights are rescaled so
that the operator norm is strictly below one, which makes the fixed-point map
contractive enough for stable iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from synthetic_cyclic_scm import (
    _has_sigma_backdoor,
    _has_directed_cycle,
    _treatment_children_stay_in_scc,
    _treatment_scc,
    _validate_nonnegative_int,
    _validate_positive_int,
    _validate_noise_type,
    _validate_sigma_range,
    choose_observed_nodes,
    random_weighted_digraph,
    sample_exogenous_noise,
    largest_W_operator_norm,
)


@dataclass(frozen=True)
class SyntheticCyclicSCMNonlinearData:
    """Bundle the graph and sampled data returned by the nonlinear generator."""

    W: np.ndarray
    samples_full: np.ndarray
    samples_observed: np.ndarray
    observed_nodes: np.ndarray


def _coerce_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Return the caller-provided random generator or create a fresh one."""
    return np.random.default_rng() if rng is None else rng


def rescale_to_simple(
    W: np.ndarray,
    target_operator_norm: float = 0.6,
    *,
    target_radius: Optional[float] = None,
) -> np.ndarray:
    """Rescale ``W`` so its operator norm is at most ``target_operator_norm``.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix to rescale.
    target_operator_norm:
        Desired upper bound on the largest singular value after rescaling.
        Smaller values lead to a more contractive nonlinear fixed-point map.
    target_radius:
        Deprecated alias for ``target_operator_norm``.
    """
    if target_radius is not None:
        target_operator_norm = target_radius

    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2D array, got shape {W.shape}.")
    if W.size == 0:
        raise ValueError("W must be non-empty.")
    if not (0.0 <= target_operator_norm < 1.0):
        raise ValueError(
            "target_operator_norm must satisfy 0 <= target_operator_norm < 1, "
            f"got {target_operator_norm}."
        )

    operator_norm = largest_W_operator_norm(W)
    if operator_norm == 0.0:
        return W.copy()

    return W * (target_operator_norm / operator_norm)


def sample_nonlinear_scm(
    W: np.ndarray,
    n_samples: int,
    sigma_range: tuple[float, float] | list[float] = (1.0, 1.0),
    noise_type: str = "gaussian",
    max_iter: int = 500,
    tol: float = 1e-8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample observations from the nonlinear equilibrium SCM.

    For each Gaussian noise draw ``U``, the function solves

        X = tanh(W X) + U

    by fixed-point iteration from the zero vector until successive iterates are
    within ``tol`` in max norm, or until ``max_iter`` iterations are exceeded.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix of the nonlinear SCM.
    n_samples:
        Number of equilibrium samples to generate.
    sigma_range:
        Two-element range ``[low, high]`` used to sample one sigma value per
        node uniformly before drawing node-wise exogenous noise.
    noise_type:
        Exogenous-noise family selector: ``"gaussian"``, ``"non_gaussian"``,
        or ``"random"``.
    max_iter:
        Maximum number of fixed-point iterations allowed per sample.
    tol:
        Convergence tolerance on the maximum absolute coordinate update.
    rng:
        Optional random generator controlling the noise draws.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2D array, got shape {W.shape}.")
    if W.size == 0:
        raise ValueError("W must be non-empty.")

    _validate_positive_int(n_samples, "n_samples")
    _validate_positive_int(max_iter, "max_iter")
    _validate_sigma_range(sigma_range)
    _validate_noise_type(noise_type)
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}.")

    operator_norm = largest_W_operator_norm(W)
    if operator_norm >= 1.0:
        raise ValueError(
            "W does not define a conservative simple SCM because its operator "
            f"norm is not strictly less than 1 (got {operator_norm:.6f})."
        )

    generator = _coerce_rng(rng)
    n_nodes = W.shape[0]
    noise = sample_exogenous_noise(
        n_nodes=n_nodes,
        n_samples=n_samples,
        sigma_range=sigma_range,
        noise_type=noise_type,
        rng=generator,
    )
    samples = np.zeros((n_samples, n_nodes), dtype=float)

    for sample_idx in range(n_samples):
        u = noise[sample_idx]
        x = np.zeros(n_nodes, dtype=float)

        for _ in range(max_iter):
            x_next = np.tanh(W @ x) + u
            if np.max(np.abs(x_next - x)) < tol:
                samples[sample_idx] = x_next
                break
            x = x_next
        else:
            raise RuntimeError(
                "Fixed-point iteration failed to converge for sample "
                f"{sample_idx} within {max_iter} iterations."
            )

    return samples


def generate_synthetic_scm_nonlinear(
    n_nodes: int,
    n_edges: int,
    n_samples: int,
    n_observed: int,
    target_operator_norm: float = 0.6,
    sigma_range: tuple[float, float] | list[float] = (1.0, 1.0),
    noise_type: str = "gaussian",
    pretreatment_flag: bool = True,
    treatment: int = 0,
    outcome: int = 1,
    treatment_outcome_edge: bool = True,
    cyclic_flag: bool = True,
    allow_two_cycles: bool = False,
    x_singleton_scc_flag: bool = True,
    require_weak_connectivity: bool = True,
    require_directed_cycle: bool = True,
    sigma_backdoor_flag: bool = False,
    max_graph_attempts: int = 1000,
    rng_seed: Optional[int] = None,
    target_radius: Optional[float] = None,
) -> SyntheticCyclicSCMNonlinearData:
    """Generate a complete synthetic dataset from a nonlinear SCM.

    This function uses the same graph-construction code as the linear generator,
    then swaps in nonlinear fixed-point sampling with
    ``X = tanh(W X) + U``.

    Parameters
    ----------
    n_nodes:
        Number of variables/nodes in the SCM.
    n_edges:
        Number of directed edges to place in the generated graph.
    n_samples:
        Number of observational samples to generate after graph construction.
    n_observed:
        Number of nodes to retain in ``samples_observed``.
    target_operator_norm:
        Operator-norm target used to make the nonlinear fixed-point map
        contractive enough for stable iteration.
    target_radius:
        Deprecated alias for ``target_operator_norm``.
    sigma_range:
        Two-element range ``[low, high]`` used to sample one sigma value per
        node uniformly before drawing node-wise exogenous noise.
    noise_type:
        Exogenous-noise family selector: ``"gaussian"``, ``"non_gaussian"``,
        or ``"random"``.
    pretreatment_flag:
        Whether to impose the special treatment/outcome structure in which
        ``treatment`` points only to ``outcome`` and ``outcome`` has no
        outgoing edges.
    treatment:
        Node index used as the designated treatment node when the pretreatment
        constraint is enabled.
    outcome:
        Node index used as the designated outcome node when the pretreatment
        constraint is enabled.
    treatment_outcome_edge:
        If ``True``, enforce a direct edge from ``treatment`` to ``outcome``.
        If ``False``, forbid that direct edge. The rule applies regardless of
        ``pretreatment_flag``.
    cyclic_flag:
        If ``True``, use the existing graph sampler that can produce cycles and
        optionally reject acyclic draws. If ``False``, build an acyclic graph
        by allowing only strict lower-triangular off-diagonal entries in an
        internal DAG order and then mapping back to the original node labels.
        In this case, ``require_directed_cycle`` is ignored.
    allow_two_cycles:
        Whether cyclic graphs may include reciprocal edge pairs, allowing
        directed cycles of length two. Ignored when ``cyclic_flag=False``.
    x_singleton_scc_flag:
        If ``True``, require ``|SCC(treatment)| == 1``. If ``False``, require
        ``1 < |SCC(treatment)| < 4/5 * n_nodes`` and require every
        non-outcome child of ``treatment`` to lie inside ``SCC(treatment)``.
        This SCC-size rule is independent of ``pretreatment_flag``.
    require_weak_connectivity:
        Whether the underlying undirected skeleton must be connected.
    require_directed_cycle:
        Whether to reject acyclic graph draws until a genuine directed cycle is
        present.
    sigma_backdoor_flag:
        If ``True``, graph draws are rejected until there exists at least one
        valid sigma-backdoor adjustment set for the treatment/outcome pair. If
        ``False``, no sigma-backdoor check is performed.
    max_graph_attempts:
        Maximum number of candidate graph draws to try before giving up on the
        requested structural constraints.
    rng_seed:
        Optional seed used to initialize the NumPy random generator for the
        full generation process.
    """
    _validate_positive_int(n_nodes, "n_nodes")
    _validate_nonnegative_int(n_edges, "n_edges")
    _validate_positive_int(n_samples, "n_samples")
    _validate_nonnegative_int(n_observed, "n_observed")
    _validate_positive_int(max_graph_attempts, "max_graph_attempts")
    _validate_sigma_range(sigma_range)
    _validate_noise_type(noise_type)
    if target_radius is not None:
        target_operator_norm = target_radius

    if n_observed > n_nodes:
        raise ValueError(
            f"n_observed cannot exceed n_nodes; got {n_observed} > {n_nodes}."
        )
    if not x_singleton_scc_flag and n_nodes < 3:
        raise ValueError(
            "x_singleton_scc_flag=False requires enough nodes for "
            "1 < |SCC(treatment)| < 4/5 * n_nodes; got n_nodes < 3."
        )
    if cyclic_flag and require_directed_cycle:
        if pretreatment_flag:
            if not x_singleton_scc_flag:
                min_nodes_for_cycle = 2 if allow_two_cycles else 3
                min_edges_for_cycle = max(
                    n_nodes - 1 if require_weak_connectivity else 0,
                    2 if allow_two_cycles else 3,
                )
            else:
                min_nodes_for_cycle = 4 if allow_two_cycles else 5
                min_edges_for_cycle = max(
                    (
                        n_nodes
                        if (require_weak_connectivity and allow_two_cycles)
                        else (n_nodes - 1 if require_weak_connectivity else 0)
                    ),
                    3 if allow_two_cycles else 4,
                )
            if n_nodes < min_nodes_for_cycle:
                raise ValueError(
                    "A directed cycle is impossible with pretreatment_flag=True "
                    f"unless n_nodes >= {min_nodes_for_cycle} under the requested "
                    "constraints, because treatment and outcome cannot belong "
                    "to the required recurrent structure under that pretreatment convention."
                )
        else:
            min_nodes_for_cycle = 2 if allow_two_cycles else 3
            if n_nodes < min_nodes_for_cycle:
                raise ValueError(
                    "A directed cycle is impossible without self-loops unless "
                    f"n_nodes >= {min_nodes_for_cycle} under the requested "
                    "constraints."
                )
            min_edges_for_cycle = max(
                (
                    n_nodes
                    if (require_weak_connectivity and allow_two_cycles)
                    else (n_nodes - 1 if require_weak_connectivity else 0)
                ),
                2 if allow_two_cycles else 3,
            )

        if n_edges < min_edges_for_cycle:
            raise ValueError(
                f"At least {min_edges_for_cycle} edges are required to enforce "
                f"a directed cycle under the requested constraints; got {n_edges}."
            )

    generator = np.random.default_rng(rng_seed)

    for _ in range(max_graph_attempts):
        W_raw = random_weighted_digraph(
            n_nodes=n_nodes,
            n_edges=n_edges,
            allow_self_loops=False,
            pretreatment_flag=pretreatment_flag,
            treatment=treatment,
            outcome=outcome,
            treatment_outcome_edge=treatment_outcome_edge,
            cyclic_flag=cyclic_flag,
            allow_two_cycles=allow_two_cycles,
            x_singleton_scc_flag=x_singleton_scc_flag,
            require_weak_connectivity=require_weak_connectivity,
            rng=generator,
        )
        if cyclic_flag and require_directed_cycle and not _has_directed_cycle(W_raw):
            continue
        treatment_scc = _treatment_scc(W_raw, treatment=treatment)
        treatment_scc_size = len(treatment_scc)
        if x_singleton_scc_flag and treatment_scc_size != 1:
            continue
        if (
            not x_singleton_scc_flag
            and (treatment_scc_size <= 1 or 5 * treatment_scc_size >= 4 * n_nodes)
        ):
            continue
        if not x_singleton_scc_flag and not _treatment_children_stay_in_scc(
            W_raw,
            treatment=treatment,
            outcome=outcome,
            treatment_scc=treatment_scc,
        ):
            continue
        if bool(W_raw[outcome, treatment] != 0.0) != treatment_outcome_edge:
            continue
        if sigma_backdoor_flag and not _has_sigma_backdoor(
            W_raw,
            treatment=treatment,
            outcome=outcome,
        ):
            continue
        W = rescale_to_simple(W_raw, target_operator_norm=target_operator_norm)
        break
    else:
        raise RuntimeError(
            "Failed to generate a weighted graph containing a directed cycle "
            f"within {max_graph_attempts} attempts."
        )

    samples_full = sample_nonlinear_scm(
        W=W,
        n_samples=n_samples,
        sigma_range=sigma_range,
        noise_type=noise_type,
        rng=generator,
    )
    observed_nodes = choose_observed_nodes(
        n_nodes=n_nodes,
        n_observed=n_observed,
        rng=generator,
    )
    samples_observed = samples_full[:, observed_nodes]

    return SyntheticCyclicSCMNonlinearData(
        W=W,
        samples_full=samples_full,
        samples_observed=samples_observed,
        observed_nodes=observed_nodes,
    )


generate_synthetic_cyclic_scm_nonlinear = generate_synthetic_scm_nonlinear
SyntheticSCMNonlinearData = SyntheticCyclicSCMNonlinearData


if __name__ == "__main__":
    data = generate_synthetic_scm_nonlinear(
        n_nodes=6,
        n_edges=9,
        n_samples=200,
        n_observed=4,
        target_operator_norm=0.6,
        sigma_range=(0.5, 0.5),
        noise_type="gaussian",
        pretreatment_flag=True,
        treatment=0,
        outcome=1,
        require_weak_connectivity=True,
        rng_seed=7,
    )

    print("Adjacency shape:", data.W)
    print("Operator norm:", f"{largest_W_operator_norm(data.W):.3f}")
    print("Observed nodes:", data.observed_nodes)
    print("Observed sample matrix shape:", data.samples_observed.shape)
