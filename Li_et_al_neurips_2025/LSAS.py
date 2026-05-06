import re
from typing import Dict, Set, Tuple, Union
import numpy as np
import pandas as pd
from Utils.CI_test import CItest_method
from Utils.Est_Effect import est_reg_con
from Utils.MB_discovery import discover_mb
from itertools import combinations


def subsets1(candidate_set, subset_size):
    """
    Generate all subsets of a given size from the candidate set.

    Parameters:
    candidate_set -- List of elements to generate subsets from
    subset_size -- Size of the subsets to generate

    Returns:
    List of subsets
    """
    subsets_tuples = list(combinations(candidate_set, subset_size))
    subsets_lists = [list(subset) for subset in subsets_tuples]
    return subsets_lists


def list_difference(A, B):
    """
    Return all elements in list A that are not in list B.

    Parameters:
    A -- First list
    B -- Second list

    Returns:
    List of elements in A but not in B
    """
    setB = {tuple(item) if isinstance(item, list) else item for item in B}
    return [
        item
        for item in A
        if (tuple(item) not in setB if isinstance(item, list) else item not in setB)
    ]


class LSAS:
    def __init__(
        self,
        data: pd.DataFrame,
        Tr_X: Union[int, str],
        Out_Y: Union[int, str],
        alpha: float = 0.01,
        max_k: int = 6,
        CI_type: str = "FisherZ",
        mb_method: str = "TC",
        verbose: bool = False,
    ):
        """
        Initialize the LSAS model.

        Parameters:
        data -- Input dataset as a pandas DataFrame
        Tr_X -- Treatment variable
        Out_Y -- Outcome variable
        alpha -- Significance level for CI tests
        max_k -- Maximum size of conditioning sets
        CI_type -- Type of CI test ('FisherZ')
        mb_method -- MB discovery backend ('TC', 'IAMB', 'Fast_IAMB', 'HITON_MB')
        verbose -- Whether to print detailed logs
        """
        self.data = data
        self.Out_Y = Out_Y
        self.Tr_X = Tr_X
        self.alpha = alpha
        self.maxK = max_k
        self.verbose = verbose
        self.CI_type = CI_type
        self.mb_method = mb_method

        self.var_names = data.columns.tolist()
        self.ci_num = 0
        self.CI_test = CItest_method(data, method_type=CI_type, alpha=alpha)
        self.mb_set: Dict[Union[str, int], list[Union[str, int]]] = {}
        self._var_index = {var: i for i, var in enumerate(data.columns)}
        self.VAS: list = []

    def _learn_markov_blanket(self, node: str) -> None:
        """
        Learn the Markov Blanket of a target variable using the configured backend.
        """
        if node in self.mb_set:
            return
        mb, n_test = discover_mb(
            self.data,
            target=node,
            alpha=self.alpha,
            method_type=self.CI_type,
            mb_method=self.mb_method,
            max_k=self.maxK,
        )
        self.mb_set[node] = sorted(mb, key=lambda x: self._var_index[x])
        self.ci_num += n_test

        if self.verbose:
            print(
                f"MB of {node} using {self.mb_method} is {mb}, with {n_test} tests."
            )

    def _get_ci_num(self) -> int:
        """
        Get the total number of conditional independence tests performed.
        """
        return self.ci_num + self.CI_test._ci_num

    def _Rule_one(self, S: str, Z: list = []) -> Dict[str, Union[bool, float]]:
        """
        Apply Rule One to test conditional independence.

        Parameters:
        S -- Variable to test
        Z -- Conditioning set

        Returns:
        Dictionary containing CI test results
        """
        if self.verbose:
            print(f"Testing Rule One: S: {S}, Z: {Z}")
        CI_one, P_one = self.CI_test(S, self.Out_Y, Z)
        CI_two, P_two = (
            self.CI_test(S, self.Out_Y, Z + [self.Tr_X]) if not CI_one else (None, None)
        )
        return {"CI_one": CI_one, "P_one": P_one, "CI_two": CI_two, "P_two": P_two}

    def _Rule_two_case1(self, Z: list = []) -> bool:
        """
        Apply Rule Two (Case 1) to test conditional independence.

        Parameters:
        Z -- Conditioning set

        Returns:
        Boolean indicating whether the rule holds
        """
        if self.verbose:
            print(f"Testing Rule Two (Case 1): Z: {Z}")
        CI, _ = self.CI_test(self.Tr_X, self.Out_Y, Z)
        return CI

    def _Rule_two_case2(self, S: str, Z: list = []) -> bool:
        """
        Apply Rule Two (Case 2) to test conditional independence.

        Parameters:
        S -- Variable to test
        Z -- Conditioning set

        Returns:
        Boolean indicating whether the rule holds
        """
        if self.verbose:
            print(f"Testing Rule Two (Case 2): S: {S}, Z: {Z}")
        return (
            not self.CI_test(S, self.Tr_X, Z)[0] and self.CI_test(S, self.Out_Y, Z)[0]
        )

    def _is_Z_in_VAS(self, Z: list) -> bool:
        """
        Check if a conditioning set Z is already in the VAS.

        Parameters:
        Z -- Conditioning set

        Returns:
        Boolean indicating whether Z is in the VAS
        """
        return any(entry["VAS"] == Z for entry in self.VAS)

    def find_A(self) -> Union[list, int]:
        """
        Find the adjustment set (VAS) for causal discovery.

        Returns:
        Adjustment set or 0 if the causal effect is zero
        """

        # Learn Markov Blankets
        self._learn_markov_blanket(self.Tr_X)
        self._learn_markov_blanket(self.Out_Y)

        MB_X = self.mb_set[self.Tr_X].copy()
        if self.Out_Y in MB_X:
            MB_X.remove(self.Out_Y)
        else:
            return 0

        MB_Y = self.mb_set[self.Out_Y].copy()
        if self.Tr_X in MB_Y:
            MB_Y.remove(self.Tr_X)
        else:
            return 0

        if len(MB_X) == 0:
            return None

        # Rule One
        for S in MB_X:
            candidate_set = list_difference(MB_Y, [S]) if S in MB_Y else MB_Y.copy()
            conset_size = 1
            candidate_Z = subsets1(candidate_set, conset_size)

            while (
                conset_size <= len(candidate_set)
                and conset_size <= self.maxK
                and len(candidate_Z) > 0
            ):
                for Z in candidate_Z:
                    Z = sorted(Z, key=lambda x: self._var_index[x])
                    if self._is_Z_in_VAS(Z):
                        continue

                    res = self._Rule_one(S, Z)
                    if not res["CI_one"] and res["CI_two"]:
                        self.VAS.append({"VAS": Z})

                conset_size += 1
                candidate_Z = subsets1(candidate_set, conset_size)
                if self.verbose:
                    print(f"Generated candidate sets: {candidate_Z}")

        if len(self.VAS) > 0:
            return self.VAS

        # Rule Two (Case 1 and 2)
        if self._Rule_two_case1():
            return 0

        for S in MB_X:
            if self._Rule_two_case2(S):
                return 0

            candidate_set = list_difference(MB_Y, [S]) if S in MB_Y else MB_Y.copy()
            conset_size = 1
            candidate_Z = subsets1(candidate_set, conset_size)

            while (
                conset_size <= len(candidate_set)
                and conset_size <= self.maxK
                and len(candidate_Z) > 0
            ):
                for Z in candidate_Z:
                    if self._Rule_two_case1(Z) or self._Rule_two_case2(S, Z):
                        return 0

                conset_size += 1
                candidate_Z = subsets1(candidate_set, conset_size)
                if self.verbose:
                    print(f"Generated candidate sets: {candidate_Z}")
        return None


def alg_LSAS(data, Tr_X, Out_Y, mb_method="TC"):
    """
    LSAS algorithm.

    Parameters:
        data (pd.DataFrame): The dataset as a pandas DataFrame.
        Tr_X (str): The treatment variable.
        Out_Y (str): The outcome variable.
        mb_method (str): MB discovery backend ('TC', 'IAMB', 'Fast_IAMB', 'HITON_MB').
    Returns:
        dict: A dictionary containing the VAS, average treatment effect (ATE), and number of CI tests performed.
    """
    model = LSAS(data, Tr_X=Tr_X, Out_Y=Out_Y, mb_method=mb_method)
    VASs = model.find_A()
    CI_num = model._get_ci_num()

    effects_VASs = []
    if VASs is not None and VASs != 0:
        for vas in VASs:
            effect, _ = est_reg_con(Tr=Tr_X, Y=Out_Y, adjset=vas["VAS"], dataset=data)
            effects_VASs.append(effect)
    avg_effect_VASs = np.mean(effects_VASs) if len(effects_VASs) > 0 else None
    res = {"VASs": VASs, "ATE": avg_effect_VASs, "CI_num": CI_num}
    return res
