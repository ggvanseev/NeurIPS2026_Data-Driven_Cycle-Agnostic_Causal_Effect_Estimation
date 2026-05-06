"""Unit tests for local Markov blanket discovery backends."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Utils.MB_discovery import fast_iamb, hiton_mb, hiton_pc, TC_mb


def _simulate_linear_gaussian_dag(
    edges: list[tuple[str, str]],
    n_samples: int = 5000,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate data from a linear-Gaussian DAG specified by directed edges."""
    rng = np.random.default_rng(seed)
    variables = sorted({node for edge in edges for node in edge})
    parents = {variable: [] for variable in variables}
    children = {variable: [] for variable in variables}
    indegree = {variable: 0 for variable in variables}
    for parent, child in edges:
        parents[child].append(parent)
        children[parent].append(child)
        indegree[child] += 1

    topo_order: list[str] = []
    ready = sorted([variable for variable, degree in indegree.items() if degree == 0])
    while ready:
        variable = ready.pop(0)
        topo_order.append(variable)
        for child in sorted(children[variable]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()

    if len(topo_order) != len(variables):
        raise ValueError("edges must define an acyclic graph for this test helper.")

    data: dict[str, np.ndarray] = {}
    for variable in topo_order:
        noise = rng.normal(scale=0.4, size=n_samples)
        if not parents[variable]:
            data[variable] = noise
            continue

        value = noise.copy()
        for idx, parent in enumerate(parents[variable]):
            weight = 0.8 + 0.1 * idx
            value += weight * data[parent]
        data[variable] = value

    return pd.DataFrame(data, columns=variables)


class MarkovBlanketDiscoveryTests(unittest.TestCase):
    """Regression tests for the Fisher-Z blanket learners."""

    def _assert_mb_case(
        self,
        edges: list[tuple[str | int, str | int]],
        target: str | int,
        expected_mb: set[str | int],
        seed: int,
        expected_pc: set[str | int] | None = None,
    ) -> None:
        """Check that all MB learners recover the expected blanket."""
        data = _simulate_linear_gaussian_dag(edges=edges, seed=seed)

        tc_mb, tc_ci_num = TC_mb(data, target=target, alpha=0.01)
        fast_mb, fast_ci_num = fast_iamb(
            data, target=target, alpha=0.01, return_ci_num=True
        )
        hiton_pc_set, hiton_pc_ci_num = hiton_pc(
            data, target=target, alpha=0.01, return_ci_num=True
        )
        hiton_mb_set, hiton_mb_ci_num = hiton_mb(
            data, target=target, alpha=0.01, return_ci_num=True
        )

        self.assertEqual(set(tc_mb), expected_mb)
        self.assertEqual(fast_mb, expected_mb)
        self.assertEqual(hiton_mb_set, expected_mb)
        self.assertGreater(tc_ci_num, 0)
        self.assertGreater(fast_ci_num, 0)
        self.assertGreater(hiton_pc_ci_num, 0)
        self.assertGreater(hiton_mb_ci_num, 0)

        if expected_pc is not None:
            self.assertEqual(hiton_pc_set, expected_pc)

    def test_original_known_blanket_structure(self) -> None:
        """Original baseline case.

        Graph:
            0 -> 1 -> 2 <- 3

        Target:
            1

        Expected MB:
            {0, 2, 3}

        Expected PC:
            {0, 2}
        """
        edges = [
            (0, 1),
            (1, 2),
            (3, 2),
        ]
        self._assert_mb_case(
            edges=edges,
            target=1,
            expected_mb={0, 2, 3},
            expected_pc={0, 2},
            seed=7,
        )

    def test_parent_child_spouse_structure(self) -> None:
        """Parent-child-spouse structure.

        Graph:
            A -> T <- B
            T -> C <- S

        Target:
            T

        Expected MB:
            {A, B, C, S}
        """
        edges = [
            ("A", "T"),
            ("B", "T"),
            ("T", "C"),
            ("S", "C"),
        ]
        self._assert_mb_case(
            edges=edges,
            target="T",
            expected_mb={"A", "B", "C", "S"},
            seed=11,
        )

    def test_chain_plus_fork_no_spouses(self) -> None:
        """Chain plus fork without spouses.

        Graph:
            A -> B -> T -> C -> D
            E -> T

        Target:
            T

        Expected MB:
            {B, E, C}
        """
        edges = [
            ("A", "B"),
            ("B", "T"),
            ("T", "C"),
            ("C", "D"),
            ("E", "T"),
        ]
        self._assert_mb_case(
            edges=edges,
            target="T",
            expected_mb={"B", "E", "C"},
            seed=13,
        )

    def test_two_children_two_spouses(self) -> None:
        """Two children and two spouses.

        Graph:
            A -> T -> C <- S1
            T -> D <- S2

        Target:
            T

        Expected MB:
            {A, C, D, S1, S2}
        """
        edges = [
            ("A", "T"),
            ("T", "C"),
            ("S1", "C"),
            ("T", "D"),
            ("S2", "D"),
        ]
        self._assert_mb_case(
            edges=edges,
            target="T",
            expected_mb={"A", "C", "D", "S1", "S2"},
            seed=17,
        )


if __name__ == "__main__":
    unittest.main()
