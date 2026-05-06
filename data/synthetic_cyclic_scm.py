"""Utilities for generating synthetic data from a linear SCM.

This module builds random weighted directed graphs and interprets them as the
coefficient matrix of the equilibrium model

    X = W X + U,

where ``W[i, j] != 0`` denotes the directed edge ``j -> i`` and ``U`` is
Gaussian noise. The helper functions in this file cover four main steps:

1. Construct a directed graph with optional structural constraints such as
   weak connectivity, a designated pretreatment/outcome pair, and the absence of
   reciprocal edges.
2. Optionally generate either a cyclic graph or an acyclic lower-triangular
   graph.
3. Rescale the edge weights so that the operator norm is below one, which
   makes ``I - W`` invertible and yields a unique equilibrium solution.
4. Draw observational samples from that equilibrium model and select a subset
   of observed nodes.

The public entry point is ``generate_synthetic_scm_linear(...)``, which
returns the adjacency matrix, the full sample matrix, and the observed subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import networkx as nx

from sigma_backdoor import exists_valid_sigma_backdoor_set


@dataclass(frozen=True)
class SyntheticCyclicSCMData:
    """Bundle the graph and sampled data returned by the linear generator.

    ``W`` stores the weighted adjacency matrix, ``samples_full`` contains all
    simulated node values, and ``samples_observed`` is the corresponding
    submatrix restricted to ``observed_nodes``.
    """

    W: np.ndarray
    samples_full: np.ndarray
    samples_observed: np.ndarray
    observed_nodes: np.ndarray


def _coerce_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Return the caller-provided random generator or create a fresh one."""
    return np.random.default_rng() if rng is None else rng


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


def _validate_sigma_range(
    sigma_range: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Validate and normalize a per-node sigma sampling range."""
    if len(sigma_range) != 2:
        raise ValueError(
            "sigma_range must contain exactly two values [low, high]; "
            f"got {sigma_range}."
        )
    sigma_low = float(sigma_range[0])
    sigma_high = float(sigma_range[1])
    if sigma_low <= 0.0 or sigma_high <= 0.0:
        raise ValueError(
            "sigma_range entries must both be positive; "
            f"got [{sigma_low}, {sigma_high}]."
        )
    if sigma_high < sigma_low:
        raise ValueError(
            "sigma_range must satisfy low <= high; " f"got [{sigma_low}, {sigma_high}]."
        )
    return sigma_low, sigma_high


def _validate_noise_type(noise_type: str) -> str:
    """Validate and normalize the exogenous-noise family selector."""
    allowed = {"gaussian", "non_gaussian", "random"}
    if noise_type not in allowed:
        raise ValueError(
            f"noise_type must be one of {sorted(allowed)}, got {noise_type!r}."
        )
    return noise_type


def sample_exogenous_noise(
    n_nodes: int,
    n_samples: int,
    sigma_range: tuple[float, float] | list[float] = (1.0, 1.0),
    noise_type: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw exogenous noise with per-node sigma values and selectable families.

    For each node ``j``, a scale ``sigma_j`` is sampled uniformly from
    ``sigma_range``. The node's noise family is then chosen according to
    ``noise_type``:

    - ``"gaussian"``: all nodes use ``N(0, sigma_j^2)``
    - ``"non_gaussian"``: all nodes use ``U[-sigma_j, sigma_j]``
    - ``"random"``: each node independently chooses between the two families

    Parameters
    ----------
    n_nodes:
        Number of SCM nodes receiving exogenous noise.
    n_samples:
        Number of noise draws to generate.
    sigma_range:
        Two-element range ``[low, high]`` used to sample one sigma value per
        node uniformly.
    noise_type:
        Noise-family selector. Allowed values are ``"gaussian"``,
        ``"non_gaussian"``, and ``"random"``.
    rng:
        Optional random generator controlling sigma draws and noise samples.
    """
    _validate_positive_int(n_nodes, "n_nodes")
    _validate_positive_int(n_samples, "n_samples")
    sigma_low, sigma_high = _validate_sigma_range(sigma_range)
    noise_type = _validate_noise_type(noise_type)

    generator = _coerce_rng(rng)
    sigmas = generator.uniform(sigma_low, sigma_high, size=n_nodes)

    if noise_type == "gaussian":
        family_is_gaussian = np.ones(n_nodes, dtype=bool)
    elif noise_type == "non_gaussian":
        family_is_gaussian = np.zeros(n_nodes, dtype=bool)
    else:
        family_is_gaussian = generator.integers(0, 2, size=n_nodes).astype(bool)

    noise = np.zeros((n_samples, n_nodes), dtype=float)
    gaussian_nodes = np.flatnonzero(family_is_gaussian)
    uniform_nodes = np.flatnonzero(~family_is_gaussian)

    if gaussian_nodes.size > 0:
        noise[:, gaussian_nodes] = generator.normal(
            loc=0.0,
            scale=sigmas[gaussian_nodes],
            size=(n_samples, gaussian_nodes.size),
        )
    if uniform_nodes.size > 0:
        low = -sigmas[uniform_nodes]
        high = sigmas[uniform_nodes]
        noise[:, uniform_nodes] = generator.uniform(
            low=low,
            high=high,
            size=(n_samples, uniform_nodes.size),
        )

    return noise


def _is_weakly_connected(adjacency: np.ndarray) -> bool:
    """Return whether the graph is connected after ignoring edge direction.

    This is the connectivity notion used by ``require_weak_connectivity``:
    every node must belong to a single connected component in the undirected
    skeleton of the generated directed graph.
    """
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return False

    undirected = (adjacency != 0.0) | (adjacency.T != 0.0)
    visited = np.zeros(n_nodes, dtype=bool)
    stack = [0]
    visited[0] = True

    while stack:
        node = stack.pop()
        neighbors = np.flatnonzero(undirected[node])
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(int(neighbor))

    return bool(np.all(visited))


def _has_directed_cycle(adjacency: np.ndarray) -> bool:
    """Return whether the graph contains at least one directed cycle.

    The function runs a depth-first search over the directed edges encoded by
    the adjacency matrix and detects a cycle via the current recursion stack.
    It is used to enforce that "cyclic SCM" means an actually cyclic graph
    rather than merely a graph in which cycles would be allowed.
    """
    adjacency = np.asarray(adjacency)
    n_nodes = adjacency.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    in_stack = np.zeros(n_nodes, dtype=bool)

    def dfs(node: int) -> bool:
        visited[node] = True
        in_stack[node] = True

        children = np.flatnonzero(adjacency[:, node] != 0.0)
        for child in children:
            child = int(child)
            if not visited[child]:
                if dfs(child):
                    return True
            elif in_stack[child]:
                return True

        in_stack[node] = False
        return False

    for node in range(n_nodes):
        if not visited[node] and dfs(node):
            return True

    return False


def _treatment_scc_is_singleton(
    W: np.ndarray,
    treatment: int,
) -> bool:
    """Return whether the strongly connected component of ``treatment`` has size 1."""
    return _treatment_scc_size(W, treatment=treatment) == 1


def _treatment_scc_size(
    W: np.ndarray,
    treatment: int,
) -> int:
    """Return the size of the strongly connected component of ``treatment``."""
    return len(_treatment_scc(W, treatment=treatment))


def _treatment_scc(
    W: np.ndarray,
    treatment: int,
) -> set[int]:
    """Return the strongly connected component containing ``treatment``."""
    graph = _adjacency_to_digraph(W)
    for component in nx.strongly_connected_components(graph):
        if treatment in component:
            return {int(node) for node in component}
    return set()


def _treatment_children_stay_in_scc(
    W: np.ndarray,
    treatment: int,
    outcome: int,
    treatment_scc: set[int],
) -> bool:
    """Return whether treatment's non-outcome children all remain in SCC(treatment)."""
    children = np.flatnonzero(W[:, treatment] != 0.0)
    return all(
        int(child) == outcome or int(child) in treatment_scc for child in children
    )


def _adjacency_to_digraph(W: np.ndarray) -> nx.DiGraph:
    """Convert an SCM adjacency matrix into a NetworkX directed graph."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2D array, got shape {W.shape}.")

    graph = nx.DiGraph()
    n_nodes = W.shape[0]
    graph.add_nodes_from(range(n_nodes))
    for child in range(n_nodes):
        for parent in np.flatnonzero(W[child] != 0.0):
            graph.add_edge(int(parent), int(child))
    return graph


def _has_sigma_backdoor(W: np.ndarray, treatment: int, outcome: int) -> bool:
    """Return whether the graph admits at least one valid sigma-backdoor set."""
    graph = _adjacency_to_digraph(W)
    return bool(exists_valid_sigma_backdoor_set(graph, treatment, outcome))


def random_weighted_digraph(
    n_nodes: int,
    n_edges: int,
    weight_low: float = 0.2,
    weight_high: float = 0.8,
    allow_self_loops: bool = False,
    pretreatment_flag: bool = False,
    treatment: int = 0,
    outcome: int = 1,
    treatment_outcome_edge: bool = True,
    cyclic_flag: bool = True,
    allow_two_cycles: bool = False,
    x_singleton_scc_flag: bool = True,
    require_weak_connectivity: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create a random weighted directed graph with exactly ``n_edges`` edges.

    The returned matrix uses the convention ``W[i, j] != 0`` for the directed
    edge ``j -> i``. Edge magnitudes are drawn uniformly from
    ``[weight_low, weight_high]`` and then assigned a random sign.

    When ``cyclic_flag=True``, the generator uses the existing orientation
    logic. By default, each unordered node pair can be oriented in either
    direction, but reciprocal pairs are forbidden. If
    ``allow_two_cycles=True``, both directions may be present on the same node
    pair, which allows directed 2-cycles. When ``cyclic_flag=False``, the generator
    builds a strict lower-triangular DAG in an internal topological order and
    then maps it back to the original node labels. This makes the graph
    acyclic by construction while still allowing special treatment/outcome
    placement.

    When ``require_weak_connectivity=True``, the function first builds a
    spanning-tree-like backbone so all nodes lie in one weakly connected
    component, then adds extra edges. When ``pretreatment_flag=True``, it
    additionally restricts the graph so that ``treatment`` points only to
    ``outcome`` and ``outcome`` has no outgoing edges, matching the special
    treatment/outcome convention used elsewhere in the project.

    Parameters
    ----------
    n_nodes:
        Number of graph nodes.
    n_edges:
        Exact number of directed edges to include.
    weight_low:
        Lower bound for the absolute edge weight. Must be non-negative.
    weight_high:
        Upper bound for the absolute edge weight. Must exceed `weight_low`.
    allow_self_loops:
        Whether diagonal edges are permitted.
    pretreatment_flag:
        If ``True``, enforce a special pair ``treatment -> outcome`` such that
        ``outcome`` has no outgoing edges and ``treatment`` has no outgoing
        edges other than to ``outcome``. Consequently, ``outcome`` is the only
        descendant of ``treatment``.
    treatment:
        Index of the special treatment node ``X`` when `pretreatment_flag=True`.
    outcome:
        Index of the special outcome node ``Y`` when `pretreatment_flag=True`.
    treatment_outcome_edge:
        If ``True``, force the direct edge ``treatment -> outcome``. If
        ``False``, prohibit that direct edge. This restriction is enforced both
        with and without the pretreatment structure.
    cyclic_flag:
        If ``True``, allow the existing cyclic graph construction logic. If
        ``False``, force an acyclic graph by allowing only strict
        lower-triangular off-diagonal entries in an internal DAG order.
    allow_two_cycles:
        Whether cyclic graphs may include both directions on the same node
        pair, creating a directed cycle of length two. Ignored when
        ``cyclic_flag=False``.
    x_singleton_scc_flag:
        If ``True``, require ``|SCC(treatment)| == 1``. If ``False``, require
        ``1 < |SCC(treatment)| < 4/5 * n_nodes`` and require every
        non-outcome child of ``treatment`` to lie inside ``SCC(treatment)``.
        This SCC-size rule is independent of ``pretreatment_flag``.
    require_weak_connectivity:
        If ``True``, ensure the generated directed graph is weakly connected.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    np.ndarray
        A square weighted adjacency matrix of shape ``(n_nodes, n_nodes)``.
    """
    _validate_positive_int(n_nodes, "n_nodes")
    _validate_nonnegative_int(n_edges, "n_edges")

    if weight_low < 0:
        raise ValueError(f"weight_low must be non-negative, got {weight_low}.")
    if weight_high <= weight_low:
        raise ValueError(
            "weight_high must be strictly greater than weight_low; "
            f"got {weight_high} <= {weight_low}."
        )
    if pretreatment_flag and allow_self_loops:
        raise ValueError("pretreatment_flag requires allow_self_loops=False.")
    if not cyclic_flag and allow_self_loops:
        raise ValueError("cyclic_flag=False requires allow_self_loops=False.")
    if not cyclic_flag and allow_two_cycles:
        raise ValueError("allow_two_cycles=True requires cyclic_flag=True.")
    if not x_singleton_scc_flag and not cyclic_flag:
        raise ValueError("x_singleton_scc_flag=False requires cyclic_flag=True.")

    _validate_node_index(treatment, n_nodes, "treatment")
    _validate_node_index(outcome, n_nodes, "outcome")
    if int(treatment) == int(outcome):
        raise ValueError("treatment and outcome must be distinct.")

    treatment = int(treatment)
    outcome = int(outcome)
    treatment_outcome_pair = (outcome, treatment)
    reverse_treatment_outcome_pair = (treatment, outcome)

    if treatment_outcome_edge and n_edges == 0:
        raise ValueError("treatment_outcome_edge=True requires at least 1 edge.")
    if pretreatment_flag:
        if n_nodes < 2:
            raise ValueError("pretreatment_flag requires at least 2 nodes.")
        if (
            not treatment_outcome_edge
            and require_weak_connectivity
            and x_singleton_scc_flag
            and n_nodes < 3
        ):
            raise ValueError(
                "pretreatment_flag=True with treatment_outcome_edge=False and "
                "require_weak_connectivity=True requires at least 3 nodes."
            )
    if cyclic_flag and allow_two_cycles:
        max_edges = n_nodes * n_nodes if allow_self_loops else n_nodes * (n_nodes - 1)
    else:
        max_edges = (
            n_nodes * (n_nodes + 1) // 2
            if allow_self_loops
            else n_nodes * (n_nodes - 1) // 2
        )
    generator = _coerce_rng(rng)
    W = np.zeros((n_nodes, n_nodes), dtype=float)

    all_pairs = []
    if cyclic_flag:
        if allow_two_cycles:
            for i in range(n_nodes):
                if allow_self_loops:
                    all_pairs.append((i, i))
                for j in range(i + 1, n_nodes):
                    all_pairs.append((i, j))
                    all_pairs.append((j, i))
        else:
            for i in range(n_nodes):
                if allow_self_loops:
                    all_pairs.append((i, i))
                for j in range(i + 1, n_nodes):
                    if generator.integers(0, 2) == 0:
                        all_pairs.append((i, j))
                    else:
                        all_pairs.append((j, i))
    else:
        if pretreatment_flag:
            other_nodes = [
                node for node in range(n_nodes) if node not in {treatment, outcome}
            ]
            generator.shuffle(other_nodes)
            ordered_nodes = other_nodes + [int(treatment), int(outcome)]
        else:
            ordered_nodes = list(range(n_nodes))
            generator.shuffle(ordered_nodes)

        ordered_to_actual = np.array(ordered_nodes, dtype=int)
        actual_to_ordered = np.empty(n_nodes, dtype=int)
        actual_to_ordered[ordered_to_actual] = np.arange(n_nodes, dtype=int)
        if not pretreatment_flag and treatment_outcome_edge:
            treatment_pos = int(actual_to_ordered[treatment])
            outcome_pos = int(actual_to_ordered[outcome])
            if treatment_pos > outcome_pos:
                ordered_to_actual[treatment_pos], ordered_to_actual[outcome_pos] = (
                    ordered_to_actual[outcome_pos],
                    ordered_to_actual[treatment_pos],
                )
                actual_to_ordered[ordered_to_actual] = np.arange(n_nodes, dtype=int)
        elif not pretreatment_flag and not treatment_outcome_edge:
            treatment_pos = int(actual_to_ordered[treatment])
            outcome_pos = int(actual_to_ordered[outcome])
            if outcome_pos > treatment_pos:
                ordered_to_actual[treatment_pos], ordered_to_actual[outcome_pos] = (
                    ordered_to_actual[outcome_pos],
                    ordered_to_actual[treatment_pos],
                )
                actual_to_ordered[ordered_to_actual] = np.arange(n_nodes, dtype=int)
        all_pairs = [
            (
                int(ordered_to_actual[child_pos]),
                int(ordered_to_actual[parent_pos]),
            )
            for child_pos in range(n_nodes)
            for parent_pos in range(child_pos)
        ]

    if pretreatment_flag:
        allowed_pairs = []
        if cyclic_flag:
            for node_a in range(n_nodes):
                for node_b in range(node_a + 1, n_nodes):
                    if {node_a, node_b} == {treatment, outcome}:
                        if treatment_outcome_edge:
                            allowed_pairs.append(treatment_outcome_pair)
                    elif treatment in {node_a, node_b}:
                        other_node = node_b if node_a == treatment else node_a
                        if not x_singleton_scc_flag:
                            allowed_pairs.append((treatment, other_node))
                            allowed_pairs.append((other_node, treatment))
                        else:
                            allowed_pairs.append((treatment, other_node))
                    elif outcome in {node_a, node_b}:
                        other_node = node_b if node_a == outcome else node_a
                        if not x_singleton_scc_flag:
                            allowed_pairs.append((outcome, other_node))
                        else:
                            allowed_pairs.append((outcome, other_node))
                    else:
                        if allow_two_cycles:
                            allowed_pairs.append((node_a, node_b))
                            allowed_pairs.append((node_b, node_a))
                        else:
                            if generator.integers(0, 2) == 0:
                                allowed_pairs.append((node_a, node_b))
                            else:
                                allowed_pairs.append((node_b, node_a))
        else:
            for child_pos in range(n_nodes):
                child = int(ordered_to_actual[child_pos])
                for parent_pos in range(child_pos):
                    parent = int(ordered_to_actual[parent_pos])
                    pair = (child, parent)
                    if pair == treatment_outcome_pair and not treatment_outcome_edge:
                        continue
                    allowed_pairs.append(pair)

        max_edges = len(allowed_pairs)
        min_edges = (
            n_nodes - 1 if require_weak_connectivity else int(treatment_outcome_edge)
        )
        if n_edges < min_edges:
            raise ValueError(
                f"n_edges must be at least {min_edges} when "
                f"pretreatment_flag={pretreatment_flag} and "
                f"require_weak_connectivity={require_weak_connectivity}; got {n_edges}."
            )
        if n_edges > max_edges:
            raise ValueError(
                f"Cannot place {n_edges} directed edges on {n_nodes} nodes under the "
                f"requested pretreatment constraint; maximum is {max_edges}."
            )

        selected_pairs = {treatment_outcome_pair} if treatment_outcome_edge else set()

        if require_weak_connectivity:
            if cyclic_flag:
                connected_nodes = (
                    [treatment, outcome] if treatment_outcome_edge else [treatment]
                )
                remaining_nodes = [
                    node for node in range(n_nodes) if node not in connected_nodes
                ]
                generator.shuffle(remaining_nodes)
                if not treatment_outcome_edge and outcome in remaining_nodes:
                    remaining_nodes.remove(outcome)
                    remaining_nodes.append(outcome)

                for node in remaining_nodes:
                    candidate_parents = connected_nodes
                    if not treatment_outcome_edge and node == outcome:
                        candidate_parents = [
                            parent for parent in connected_nodes if parent != treatment
                        ]
                        if not candidate_parents:
                            raise ValueError(
                                "Cannot weakly connect outcome under "
                                "pretreatment_flag=True and "
                                "treatment_outcome_edge=False."
                            )
                    parent = int(
                        generator.choice(np.array(candidate_parents, dtype=int))
                    )
                    if parent == treatment and x_singleton_scc_flag:
                        selected_pairs.add((treatment, node))
                    elif parent == treatment and not x_singleton_scc_flag:
                        selected_pairs.add((node, treatment))
                    elif parent == outcome and x_singleton_scc_flag:
                        selected_pairs.add((outcome, node))
                    elif parent == outcome and not x_singleton_scc_flag:
                        selected_pairs.add((outcome, node))
                    elif node == outcome:
                        selected_pairs.add((outcome, parent))
                    else:
                        if generator.integers(0, 2) == 0:
                            selected_pairs.add((node, parent))
                        else:
                            selected_pairs.add((parent, node))
                    connected_nodes.append(node)
            else:
                treatment_pos = int(actual_to_ordered[treatment])
                outcome_pos = int(actual_to_ordered[outcome])
                for child_pos in range(1, treatment_pos + 1):
                    child = int(ordered_to_actual[child_pos])
                    if child == outcome:
                        continue
                    if child == treatment:
                        parent_pos = int(generator.integers(0, child_pos))
                    else:
                        parent_pos = int(generator.integers(0, child_pos))
                    parent = int(ordered_to_actual[parent_pos])
                    selected_pairs.add((child, parent))
                if treatment_outcome_edge:
                    selected_pairs.add(treatment_outcome_pair)

        available_extra_pairs = []
        for pair in allowed_pairs:
            if pair in selected_pairs:
                continue
            if cyclic_flag and not allow_two_cycles:
                i, j = pair
                selected_unordered_pairs = {
                    frozenset((src, dst)) for src, dst in selected_pairs if src != dst
                }
                if i != j and frozenset((i, j)) in selected_unordered_pairs:
                    continue
            available_extra_pairs.append(pair)
        extra_needed = n_edges - len(selected_pairs)
        if extra_needed > 0:
            extra_indices = generator.choice(
                len(available_extra_pairs), size=extra_needed, replace=False
            )
            for pair_idx in np.atleast_1d(extra_indices):
                selected_pairs.add(available_extra_pairs[int(pair_idx)])
    else:
        if treatment_outcome_edge:
            if treatment_outcome_pair not in all_pairs:
                all_pairs = [
                    pair for pair in all_pairs if pair != reverse_treatment_outcome_pair
                ]
                all_pairs.append(treatment_outcome_pair)
        else:
            if treatment_outcome_pair in all_pairs:
                all_pairs = [
                    pair for pair in all_pairs if pair != treatment_outcome_pair
                ]
                if cyclic_flag and not allow_two_cycles:
                    all_pairs.append(reverse_treatment_outcome_pair)
        max_edges = len(all_pairs)

        if n_edges > max_edges:
            raise ValueError(
                f"Cannot place {n_edges} directed edges on {n_nodes} nodes when "
                f"allow_self_loops={allow_self_loops}"
                f"{' and reciprocal pairs are forbidden' if not allow_two_cycles else ''}; "
                f"maximum is {max_edges}."
            )
        min_edges = n_nodes - 1 if require_weak_connectivity else 0
        if n_edges < min_edges:
            raise ValueError(
                f"n_edges must be at least {min_edges} when "
                f"require_weak_connectivity={require_weak_connectivity}; got {n_edges}."
            )

        if n_edges == 0:
            return W

        selected_pairs = {treatment_outcome_pair} if treatment_outcome_edge else set()
        if require_weak_connectivity:
            if cyclic_flag:
                if treatment_outcome_edge:
                    connected_nodes = [treatment, outcome]
                    node_order = [
                        node for node in range(n_nodes) if node not in connected_nodes
                    ]
                    generator.shuffle(node_order)
                else:
                    node_order = list(range(n_nodes))
                    generator.shuffle(node_order)
                    connected_nodes = [node_order.pop()]
                for node in node_order:
                    candidate_parents = connected_nodes
                    parent = int(
                        generator.choice(np.array(candidate_parents, dtype=int))
                    )
                    candidate_pair = (
                        (node, parent)
                        if generator.integers(0, 2) == 0
                        else (parent, node)
                    )
                    if candidate_pair == treatment_outcome_pair:
                        candidate_pair = reverse_treatment_outcome_pair
                    if (
                        treatment_outcome_edge
                        and not allow_two_cycles
                        and candidate_pair == reverse_treatment_outcome_pair
                    ):
                        candidate_pair = treatment_outcome_pair
                    selected_pairs.add(candidate_pair)
                    connected_nodes.append(node)
            else:
                for child_pos in range(1, n_nodes):
                    child = int(ordered_to_actual[child_pos])
                    if child == outcome and treatment_outcome_edge:
                        selected_pairs.add(treatment_outcome_pair)
                        continue
                    parent_pos = int(generator.integers(0, child_pos))
                    parent = int(ordered_to_actual[parent_pos])
                    pair = (child, parent)
                    if pair == treatment_outcome_pair and not treatment_outcome_edge:
                        continue
                    selected_pairs.add(pair)

        available_extra_pairs = []
        for pair in all_pairs:
            if pair in selected_pairs:
                continue
            if pair == treatment_outcome_pair and not treatment_outcome_edge:
                continue
            if (
                pair == reverse_treatment_outcome_pair
                and treatment_outcome_edge
                and cyclic_flag
                and not allow_two_cycles
            ):
                continue
            if cyclic_flag and not allow_two_cycles:
                i, j = pair
                selected_unordered_pairs = {
                    frozenset((src, dst)) for src, dst in selected_pairs if src != dst
                }
                if i != j and frozenset((i, j)) in selected_unordered_pairs:
                    continue
            available_extra_pairs.append(pair)
        extra_needed = n_edges - len(selected_pairs)
        if extra_needed > 0:
            extra_indices = generator.choice(
                len(available_extra_pairs), size=extra_needed, replace=False
            )
            for pair_idx in np.atleast_1d(extra_indices):
                selected_pairs.add(available_extra_pairs[int(pair_idx)])

    selected_pairs_list = list(selected_pairs)
    magnitudes = generator.uniform(
        weight_low, weight_high, size=len(selected_pairs_list)
    )
    signs = generator.choice(np.array([-1.0, 1.0]), size=len(selected_pairs_list))

    for edge_idx, (i, j) in enumerate(selected_pairs_list):
        W[i, j] = magnitudes[edge_idx] * signs[edge_idx]

    if require_weak_connectivity and not _is_weakly_connected(W):
        raise RuntimeError("Internal error: generated graph is not weakly connected.")

    return W


def largest_W_operator_norm(W: np.ndarray) -> float:
    """Return the Euclidean operator norm of ``W``.

    Concretely, this function returns the largest singular value of ``W``,
    which is the induced matrix 2-norm:

        ||W||_2 = sup_{x != 0} ||W x||_2 / ||x||_2.


    Parameters
    ----------
    W:
        Square weighted adjacency matrix whose largest singular value is
        computed.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2D array, got shape {W.shape}.")
    if W.size == 0:
        raise ValueError("W must be non-empty.")

    return float(np.linalg.svd(W, compute_uv=False)[0])


def has_nonsingular_system_matrix(W: np.ndarray) -> bool:
    """Return whether the linear equilibrium system ``(I - W) X = U`` is solvable.

    If ``I - W`` is non-singular, the SCM has a unique equilibrium solution for
    every exogenous noise draw even when the stricter operator-norm condition
    is not enforced.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix defining the linear SCM.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2D array, got shape {W.shape}.")
    if W.size == 0:
        raise ValueError("W must be non-empty.")

    identity = np.eye(W.shape[0], dtype=float)
    system_matrix = identity - W
    return bool(np.linalg.matrix_rank(system_matrix) == W.shape[0])


def rescale_to_simple(
    W: np.ndarray,
    target_operator_norm: float = 0.8,
    *,
    target_radius: Optional[float] = None,
) -> np.ndarray:
    """Rescale ``W`` so its operator norm is at most ``target_operator_norm``.

    This keeps the graph structure and edge-sign pattern fixed while shrinking
    all edge weights by a common factor. If ``W`` is the zero matrix, there is
    nothing to rescale and a copy is returned unchanged.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix to rescale.
    target_operator_norm:
        Desired upper bound on the largest singular value after rescaling.
        Values closer to one preserve stronger edges; smaller values produce a
        more conservative, more weakly coupled system.
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

    max_svd = largest_W_operator_norm(W)
    if max_svd == 0.0:
        return W.copy()

    scale = target_operator_norm / max_svd
    return W * scale


def sample_linear_scm(
    W: np.ndarray,
    n_samples: int,
    sigma_range: tuple[float, float] | list[float] = (1.0, 1.0),
    noise_type: str = "gaussian",
    enforce_operator_norm_condition: bool = True,
    rng: Optional[np.random.Generator] = None,
    enforce_spectral_radius_condition: Optional[bool] = None,
) -> np.ndarray:
    """Sample observations from the linear equilibrium SCM ``X = W X + U``.

    Each sample is produced by drawing Gaussian noise ``U`` and solving the
    linear system ``(I - W) X = U``. When
    ``enforce_operator_norm_condition=True``, the function requires
    ``||W||_2 < 1``; otherwise it only checks that ``I - W`` is
    non-singular.

    Parameters
    ----------
    W:
        Square weighted adjacency matrix of the linear SCM.
    n_samples:
        Number of equilibrium samples to draw.
    sigma_range:
        Two-element range ``[low, high]`` used to sample one sigma value per
        node uniformly before drawing node-wise exogenous noise.
    noise_type:
        Exogenous-noise family selector: ``"gaussian"``, ``"non_gaussian"``,
        or ``"random"``.
    enforce_operator_norm_condition:
        If ``True``, require ``||W||_2 < 1`` before sampling. If
        ``False``, only require that ``I - W`` is invertible.
    rng:
        Optional random generator controlling the noise draws.
    enforce_spectral_radius_condition:
        Deprecated alias for ``enforce_operator_norm_condition``.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2D array, got shape {W.shape}.")
    if W.size == 0:
        raise ValueError("W must be non-empty.")

    _validate_positive_int(n_samples, "n_samples")
    _validate_sigma_range(sigma_range)
    _validate_noise_type(noise_type)
    if enforce_spectral_radius_condition is not None:
        enforce_operator_norm_condition = enforce_spectral_radius_condition

    generator = _coerce_rng(rng)
    n_nodes = W.shape[0]
    identity = np.eye(n_nodes, dtype=float)
    system_matrix = identity - W

    if enforce_operator_norm_condition:
        operator_norm = largest_W_operator_norm(W)
        if operator_norm >= 1.0:
            raise ValueError(
                "W does not define a simple SCM because its operator norm is not "
                f"strictly less than 1 (got {operator_norm:.6f})."
            )
    elif not has_nonsingular_system_matrix(W):
        raise ValueError(
            "I - W must be non-singular when operator-norm checks are disabled."
        )

    noise = sample_exogenous_noise(
        n_nodes=n_nodes,
        n_samples=n_samples,
        sigma_range=sigma_range,
        noise_type=noise_type,
        rng=generator,
    )

    samples = np.linalg.solve(system_matrix, noise.T).T
    return samples


def choose_observed_nodes(
    n_nodes: int,
    n_observed: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly choose which nodes are observed in the final dataset.

    The returned indices are sorted so downstream code can rely on a stable
    column ordering when subsetting the full sample matrix.

    Parameters
    ----------
    n_nodes:
        Total number of nodes in the SCM.
    n_observed:
        Number of node indices to keep in the observed subset.
    rng:
        Optional random generator controlling which nodes are selected.
    """
    _validate_positive_int(n_nodes, "n_nodes")
    _validate_nonnegative_int(n_observed, "n_observed")

    if n_observed > n_nodes:
        raise ValueError(
            f"n_observed cannot exceed n_nodes; got {n_observed} > {n_nodes}."
        )

    generator = _coerce_rng(rng)
    observed_nodes = generator.choice(n_nodes, size=n_observed, replace=False)
    return np.sort(observed_nodes.astype(int))


def generate_synthetic_scm_linear(
    n_nodes: int,
    n_edges: int,
    n_samples: int,
    n_observed: int,
    target_operator_norm: float = 0.8,
    sigma_range: tuple[float, float] | list[float] = (1.0, 1.0),
    noise_type: str = "gaussian",
    enforce_operator_norm_condition: bool = True,
    pretreatment_flag: bool = False,
    treatment: int = 0,
    outcome: int = 1,
    treatment_outcome_edge: bool = True,
    cyclic_flag: bool = True,
    allow_two_cycles: bool = False,
    x_singleton_scc_flag: bool = False,
    require_weak_connectivity: bool = False,
    require_directed_cycle: bool = True,
    sigma_backdoor_flag: bool = False,
    max_graph_attempts: int = 1000,
    rng_seed: Optional[int] = None,
    target_radius: Optional[float] = None,
    enforce_spectral_radius_condition: Optional[bool] = None,
) -> SyntheticCyclicSCMData:
    """Generate a complete synthetic dataset from a linear SCM.

    The function repeatedly samples a candidate directed graph until it
        satisfies the requested structural constraints, such as weak connectivity,
        the optional pretreatment/outcome pattern, and the presence of at least
        one directed cycle when requested, or produces a lower-triangular DAG in
        an internal topological order when
    ``cyclic_flag=False``. It then rescales the graph if requested, draws
    equilibrium samples, and selects a subset of observed nodes.

    The return value contains:
    - ``W``: the final weighted adjacency matrix
    - ``samples_full``: all simulated node values
    - ``samples_observed``: the observed-node subset of those samples
    - ``observed_nodes``: the node indices retained in the observed dataset

    Parameters
    ----------
    n_nodes:
        Number of variables/nodes in the SCM.
    n_edges:
        Number of directed edges to place in the generated graph.
    n_samples:
        Number of observational samples to generate after the graph is built.
    n_observed:
        Number of nodes to expose in ``samples_observed``.
    target_operator_norm:
        Operator-norm target used when shrinking edge weights to a stable
        linear system. Ignored when ``enforce_operator_norm_condition=False``
        and a non-singular unscaled graph is accepted directly.
    sigma_range:
        Two-element range ``[low, high]`` used to sample one sigma value per
        node uniformly before drawing node-wise exogenous noise.
    noise_type:
        Exogenous-noise family selector: ``"gaussian"``, ``"non_gaussian"``,
        or ``"random"``.
    enforce_operator_norm_condition:
        Whether to require and enforce ``||W||_2 < 1`` before
        sampling. If ``False``, the function only needs ``I - W`` to be
        non-singular.
    target_radius:
        Deprecated alias for ``target_operator_norm``.
    enforce_spectral_radius_condition:
        Deprecated alias for ``enforce_operator_norm_condition``.
    pretreatment_flag:
        Whether to impose the special pretreatment structure in which
        ``treatment`` points only to ``outcome`` and ``outcome`` has no
        outgoing edges.
    treatment:
        Node index used as the designated treatment node when
        ``pretreatment_flag=True``.
    outcome:
        Node index used as the designated outcome node when
        ``pretreatment_flag=True``.
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
        Whether acyclic draws should be rejected until a graph with at least
        one directed cycle is found.
    sigma_backdoor_flag:
        If ``True``, graph draws are rejected until there exists at least one
        valid sigma-backdoor adjustment set for the treatment/outcome pair. If
        ``False``, no sigma-backdoor check is performed.
    max_graph_attempts:
        Maximum number of candidate graph draws to try before giving up on the
        requested structural constraints.
    rng_seed:
        Optional seed used to initialize the NumPy random generator for the
        entire generation process.
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
    if enforce_spectral_radius_condition is not None:
        enforce_operator_norm_condition = enforce_spectral_radius_condition

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
                min_nodes_for_cycle = 2 if allow_two_cycles else 3
                min_edges_for_cycle = max(
                    (
                        n_nodes
                        if (require_weak_connectivity and allow_two_cycles)
                        else (n_nodes - 1 if require_weak_connectivity else 0)
                    ),
                    2 if allow_two_cycles else 3,
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
        if enforce_operator_norm_condition:
            W = rescale_to_simple(W_raw, target_operator_norm=target_operator_norm)
            break
        if has_nonsingular_system_matrix(W_raw):
            W = W_raw
            break
    else:
        raise RuntimeError(
            "Failed to generate a weighted graph with non-singular I - W "
            f"within {max_graph_attempts} attempts."
        )

    samples_full = sample_linear_scm(
        W=W,
        n_samples=n_samples,
        sigma_range=sigma_range,
        noise_type=noise_type,
        enforce_operator_norm_condition=enforce_operator_norm_condition,
        rng=generator,
    )
    observed_nodes = choose_observed_nodes(
        n_nodes=n_nodes,
        n_observed=n_observed,
        rng=generator,
    )
    samples_observed = samples_full[:, observed_nodes]

    return SyntheticCyclicSCMData(
        W=W,
        samples_full=samples_full,
        samples_observed=samples_observed,
        observed_nodes=observed_nodes,
    )


generate_synthetic_cyclic_scm_linear = generate_synthetic_scm_linear
SyntheticSCMData = SyntheticCyclicSCMData


if __name__ == "__main__":
    dataset = generate_synthetic_scm_linear(
        n_nodes=6,
        n_edges=10,
        n_samples=1000,
        n_observed=4,
        target_operator_norm=0.8,
        sigma_range=(1.0, 1.0),
        noise_type="gaussian",
        enforce_operator_norm_condition=False,
        pretreatment_flag=True,
        treatment=0,
        outcome=1,
        require_weak_connectivity=True,
        require_directed_cycle=True,
        rng_seed=14,
    )

    print("Adjacency matrix shape:", dataset.W)
    print("Operator norm:", f"{largest_W_operator_norm(dataset.W):.4f}")
    print("Observed nodes:", dataset.observed_nodes)
    print("Full samples shape:", dataset.samples_full.shape)
    print("Observed samples shape:", dataset.samples_observed.shape)
    print("Source outgoing targets:", np.flatnonzero(dataset.W[:, 0]))
    print("Sink outgoing targets:", np.flatnonzero(dataset.W[:, 1]))
