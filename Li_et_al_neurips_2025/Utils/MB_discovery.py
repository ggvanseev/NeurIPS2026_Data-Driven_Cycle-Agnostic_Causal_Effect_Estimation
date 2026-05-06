"""Markov blanket discovery utilities used by LSAS.

This module keeps the original TC-based blanket learner and adds Fisher-Z based
Fast-IAMB, HITON-PC, and HITON-MB implementations as selectable alternatives.

Assumptions
-----------
These local blanket learners rely on standard causal-discovery assumptions:

- Causal sufficiency over the observed variables used by the learner.
- Faithfulness / stability so conditional independences correspond to the DAG.
- Reliable conditional-independence tests.
- A Gaussian setting in which Fisher-Z is an appropriate CI test.

The new algorithms below intentionally follow a conservative, deterministic
translation of the requested pseudocode. They do not import external causal
discovery libraries.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Iterable, Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .CI_test import CItest_method

Variable = Union[int, str]
DataLike = Union[pd.DataFrame, np.ndarray]


class _CITestTracker:
    """Cache CI-test p-values and count unique conditional-independence tests."""

    def __init__(
        self,
        data: pd.DataFrame,
        alpha: float,
        order_map: Dict[Variable, int],
        method_type: str = "FisherZ",
    ):
        self._ci_tester = CItest_method(data, method_type=method_type, alpha=alpha)
        self._order_map = order_map
        self._cache: Dict[tuple[Variable, Variable, tuple[Variable, ...]], float] = {}

    @property
    def count(self) -> int:
        """Return the number of unique CI tests evaluated so far."""
        return len(self._cache)

    def p_value(
        self, x: Variable, y: Variable, conditioning_set: Iterable[Variable]
    ) -> float:
        """Return a cached p-value for ``x ⟂ y | conditioning_set``."""
        x_key, y_key = (x, y)
        if self._order_map[x_key] > self._order_map[y_key]:
            x_key, y_key = y_key, x_key
        cond_key = _sorted_conditioning_set(conditioning_set, self._order_map)
        key = (x_key, y_key, cond_key)
        if key not in self._cache:
            _, p_value = self._ci_tester(x, y, list(cond_key))
            self._cache[key] = float(p_value)
        return self._cache[key]


def _as_dataframe(data: DataLike) -> pd.DataFrame:
    """Return ``data`` as a DataFrame with deterministic column labels."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {array.shape}.")
    return pd.DataFrame(array, columns=list(range(array.shape[1])))


def variables(data: DataLike) -> list[Variable]:
    """Return variables in deterministic column order."""
    if isinstance(data, pd.DataFrame):
        return list(data.columns)
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {array.shape}.")
    return list(range(array.shape[1]))


def n_samples(data: DataLike) -> int:
    """Return the sample size."""
    return int(_as_dataframe(data).shape[0])


def powerset_limited(
    items: Sequence[Variable], max_size: Optional[int]
) -> Iterator[tuple[Variable, ...]]:
    """Yield subsets of ``items`` up to size ``max_size`` in deterministic order."""
    upper = len(items) if max_size is None else min(len(items), int(max_size))
    for size in range(upper + 1):
        yield from combinations(items, size)


def _sorted_conditioning_set(
    conditioning_set: Iterable[Variable], order_map: Dict[Variable, int]
) -> tuple[Variable, ...]:
    """Canonicalize a conditioning set for caching."""
    return tuple(sorted(conditioning_set, key=lambda value: order_map[value]))


def ci_test_cached(
    data: pd.DataFrame,
    x: Variable,
    y: Variable,
    conditioning_set: Iterable[Variable],
    tracker: _CITestTracker,
) -> float:
    """Return a cached Fisher-Z p-value for ``x ⟂ y | conditioning_set``."""
    del data
    return tracker.p_value(x, y, conditioning_set)


def assoc_score(
    data: pd.DataFrame,
    x: Variable,
    y: Variable,
    conditioning_set: Iterable[Variable],
    tracker: _CITestTracker,
) -> float:
    """Association score ``-log(max(p_value, 1e-300))`` used for ranking."""
    p_value = ci_test_cached(data, x, y, conditioning_set, tracker)
    return -math.log(max(p_value, 1e-300))


def _dependent(
    data: pd.DataFrame,
    x: Variable,
    y: Variable,
    conditioning_set: Iterable[Variable],
    alpha: float,
    tracker: _CITestTracker,
) -> bool:
    """Return whether Fisher-Z declares dependence at level ``alpha``."""
    return ci_test_cached(data, x, y, conditioning_set, tracker) <= alpha


def _normalize_method(method: str) -> str:
    """Map user-facing MB discovery names to canonical identifiers."""
    normalized = method.lower().replace("-", "_").replace(" ", "")
    aliases = {
        "tc": "tc",
        "tc_mb": "tc",
        "fastiamb": "fast_iamb",
        "fast_iamb": "fast_iamb",
        "fastiamb_mb": "fast_iamb",
        "iamb": "iamb",
        "iamb_mb": "iamb",
        "hitonmb": "hiton_mb",
        "hiton_mb": "hiton_mb",
    }
    if normalized not in aliases:
        raise ValueError(
            "Invalid MB discovery method. Choose from 'TC', 'IAMB', "
            "'Fast_IAMB', or 'HITON_MB'."
        )
    return aliases[normalized]


def TC_mb(
    data: DataLike,
    target: Variable,
    alpha: float = 0.05,
    method_type: str = "FisherZ",
    use_lasso: bool = False,
) -> tuple[list[Variable], int]:
    """Original TC algorithm retained as the default LSAS behavior.

    Notes
    -----
    This preserves the original total-conditioning blanket check, including the
    sample-size-dependent alpha shrinkage from the previous implementation.
    ``use_lasso`` is kept only for backward API compatibility and is ignored.
    """
    del use_lasso

    df = _as_dataframe(data)
    sample_size = df.shape[0]
    do_n = sample_size / 10
    ceil_result = math.floor(do_n)
    adjusted_alpha = alpha / (ceil_result * 10) if ceil_result > 0 else alpha

    mb: list[Variable] = []
    ntest = 0
    candidates = list(df.columns.difference([target]))
    cit = CItest_method(df, method_type=method_type, alpha=adjusted_alpha)

    for candidate in candidates:
        ntest += 1
        conditioning = [variable for variable in candidates if variable != candidate]
        if not cit(target, candidate, conditioning)[0]:
            mb.append(candidate)

    return mb, ntest


def iamb(
    data: DataLike,
    target: Variable,
    alpha: float = 0.05,
    method_type: str = "FisherZ",
    max_k: Optional[int] = None,
    return_ci_num: bool = False,
) -> Union[set[Variable], tuple[set[Variable], int]]:
    """Classical IAMB grow-shrink blanket discovery using ``CItest_method``."""
    df = _as_dataframe(data)
    vars_in_order = variables(df)
    order_map = {variable: index for index, variable in enumerate(vars_in_order)}
    tracker = _CITestTracker(
        df, alpha=alpha, order_map=order_map, method_type=method_type
    )

    mb: list[Variable] = []
    changed = True
    while changed:
        changed = False
        best_candidate: Optional[Variable] = None
        best_score = -math.inf

        for variable in vars_in_order:
            if variable == target or variable in mb:
                continue
            if max_k is not None and len(mb) >= max_k:
                break
            if _dependent(df, target, variable, mb, alpha, tracker):
                score = assoc_score(df, target, variable, mb, tracker)
                if score > best_score:
                    best_candidate = variable
                    best_score = score

        if best_candidate is not None:
            mb.append(best_candidate)
            changed = True

        for variable in mb.copy():
            conditioning = [member for member in mb if member != variable]
            if not _dependent(df, target, variable, conditioning, alpha, tracker):
                mb.remove(variable)
                changed = True

    mb_set = set(mb)
    if return_ci_num:
        return mb_set, tracker.count
    return mb_set


def fast_iamb(
    data: DataLike,
    target: Variable,
    alpha: float = 0.05,
    method_type: str = "FisherZ",
    max_k: Optional[int] = None,
    min_df: int = 5,
    return_ci_num: bool = False,
) -> Union[set[Variable], tuple[set[Variable], int]]:
    """Fast-IAMB blanket discovery using ``CItest_method``.

    The implementation follows the requested procedure:
    start from marginally dependent variables, sort by conditional association,
    speculatively add candidates while degrees of freedom remain sufficient,
    then run a shrink phase and recompute the active candidate set.
    """
    df = _as_dataframe(data)
    vars_in_order = variables(df)
    order_map = {variable: index for index, variable in enumerate(vars_in_order)}
    tracker = _CITestTracker(
        df, alpha=alpha, order_map=order_map, method_type=method_type
    )
    sample_size = n_samples(df)

    mb: list[Variable] = []
    active = [
        variable
        for variable in vars_in_order
        if variable != target
        and _dependent(df, target, variable, [], alpha, tracker)
    ]

    while active:
        previous_mb = tuple(mb)
        ranked = sorted(
            active,
            key=lambda variable: (
                -assoc_score(df, target, variable, mb, tracker),
                order_map[variable],
            ),
        )

        for variable in ranked:
            if variable in mb:
                continue
            if max_k is not None and len(mb) >= max_k:
                break
            if sample_size - len(mb) - 3 <= min_df:
                break
            if _dependent(df, target, variable, mb, alpha, tracker):
                mb.append(variable)

        # Shrink phase: remove variables rendered independent by the rest of MB.
        for variable in mb.copy():
            conditioning = [member for member in mb if member != variable]
            if not _dependent(df, target, variable, conditioning, alpha, tracker):
                mb.remove(variable)

        active = [
            variable
            for variable in vars_in_order
            if variable != target
            and variable not in mb
            and _dependent(df, target, variable, mb, alpha, tracker)
        ]

        if tuple(mb) == previous_mb:
            break

    mb_set = set(mb)
    if return_ci_num:
        return mb_set, tracker.count
    return mb_set


def _hiton_pc_impl(
    df: pd.DataFrame,
    target: Variable,
    alpha: float,
    max_k: int,
    tracker: _CITestTracker,
    order_map: Dict[Variable, int],
) -> list[Variable]:
    """Internal HITON-PC implementation sharing a cache across nested calls."""
    vars_in_order = variables(df)
    remaining = [
        variable
        for variable in vars_in_order
        if variable != target
        and _dependent(df, target, variable, [], alpha, tracker)
    ]
    remaining.sort(
        key=lambda variable: (
            -assoc_score(df, target, variable, [], tracker),
            order_map[variable],
        )
    )

    pc: list[Variable] = []
    removed: set[Variable] = set()

    while remaining:
        candidate = remaining.pop(0)
        if candidate in removed:
            continue
        pc.append(candidate)

        changed = True
        while changed:
            changed = False
            for variable in pc.copy():
                others = [member for member in pc if member != variable]
                for subset in powerset_limited(others, max_k):
                    if not _dependent(
                        df,
                        target,
                        variable,
                        subset,
                        alpha,
                        tracker,
                    ):
                        pc.remove(variable)
                        removed.add(variable)
                        changed = True
                        break
                if changed:
                    break

        remaining = [variable for variable in remaining if variable not in removed]

    return pc


def hiton_pc(
    data: DataLike,
    target: Variable,
    alpha: float = 0.05,
    method_type: str = "FisherZ",
    max_k: int = 3,
    return_ci_num: bool = False,
) -> Union[set[Variable], tuple[set[Variable], int]]:
    """Return the HITON-PC parents-and-children set for ``target``."""
    df = _as_dataframe(data)
    vars_in_order = variables(df)
    order_map = {variable: index for index, variable in enumerate(vars_in_order)}
    tracker = _CITestTracker(
        df, alpha=alpha, order_map=order_map, method_type=method_type
    )
    pc_set = set(_hiton_pc_impl(df, target, alpha, max_k, tracker, order_map))
    if return_ci_num:
        return pc_set, tracker.count
    return pc_set


def hiton_mb(
    data: DataLike,
    target: Variable,
    alpha: float = 0.05,
    method_type: str = "FisherZ",
    max_k: int = 3,
    return_ci_num: bool = False,
) -> Union[set[Variable], tuple[set[Variable], int]]:
    """Return the HITON-MB Markov blanket for ``target``.

    HITON-MB first estimates the local parents-and-children set and then adds
    spouse candidates discovered through the children. A final shrink step
    removes variables that become conditionally independent from the target
    given the rest of the blanket.
    """
    df = _as_dataframe(data)
    vars_in_order = variables(df)
    order_map = {variable: index for index, variable in enumerate(vars_in_order)}
    tracker = _CITestTracker(
        df, alpha=alpha, order_map=order_map, method_type=method_type
    )

    pc_target = _hiton_pc_impl(df, target, alpha, max_k, tracker, order_map)
    pc_target_set = set(pc_target)
    mb = set(pc_target)

    for child in pc_target:
        pc_child = _hiton_pc_impl(df, child, alpha, max_k, tracker, order_map)
        spouse_candidates = [
            variable
            for variable in pc_child
            if variable != target and variable not in pc_target_set
        ]

        local_candidates = [
            variable
            for variable in vars_in_order
            if variable not in {target, child}
            and (variable in pc_target_set or variable in set(pc_child))
        ]

        for candidate in spouse_candidates:
            search_pool = [variable for variable in local_candidates if variable != candidate]
            separated = False
            subset_size = max(0, max_k - 1)
            for subset in powerset_limited(search_pool, subset_size):
                conditioning = list(subset) + [child]
                if not _dependent(
                    df,
                    target,
                    candidate,
                    conditioning,
                    alpha,
                    tracker,
                ):
                    separated = True
                    break
            if not separated:
                mb.add(candidate)

    changed = True
    while changed:
        changed = False
        for variable in sorted(mb, key=lambda value: order_map[value]):
            conditioning = [member for member in mb if member != variable]
            if not _dependent(df, target, variable, conditioning, alpha, tracker):
                mb.remove(variable)
                changed = True
                break

    if return_ci_num:
        return mb, tracker.count
    return mb


def discover_mb(
    data: DataLike,
    target: Variable,
    alpha: float = 0.05,
    method_type: str = "FisherZ",
    mb_method: str = "TC",
    use_lasso: bool = False,
    max_k: int = 3,
    min_df: int = 5,
) -> tuple[list[Variable], int]:
    """Dispatch to the requested MB discovery backend and return CI test count."""
    normalized_method = _normalize_method(mb_method)
    if normalized_method == "tc":
        return TC_mb(
            data,
            target=target,
            alpha=alpha,
            method_type=method_type,
            use_lasso=use_lasso,
        )
    df = _as_dataframe(data)
    vars_in_order = variables(df)
    order_map = {variable: index for index, variable in enumerate(vars_in_order)}

    if normalized_method == "iamb":
        mb_set, n_test = iamb(
            df,
            target=target,
            alpha=alpha,
            method_type=method_type,
            max_k=max_k,
            return_ci_num=True,
        )
        mb = list(sorted(mb_set, key=lambda value: order_map[value]))
    elif normalized_method == "fast_iamb":
        mb_set, n_test = fast_iamb(
            df,
            target=target,
            alpha=alpha,
            method_type=method_type,
            max_k=max_k,
            min_df=min_df,
            return_ci_num=True,
        )
        mb = list(sorted(mb_set, key=lambda value: order_map[value]))
    else:
        mb_set, n_test = hiton_mb(
            df,
            target=target,
            alpha=alpha,
            method_type=method_type,
            max_k=max_k,
            return_ci_num=True,
        )
        mb = list(sorted(mb_set, key=lambda value: order_map[value]))

    return mb, n_test
