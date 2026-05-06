'''
Tests the null hypothesis that X is independent from Y given S.
"""
This file contains implementations of conditional independence tests.
Some functions are adapted from the Python package `causallearn` and the R package `pcalg`.

- URL: https://github.com/py-why/causal-learn
- URL: https://github.com/cran/pcalg

All borrowed code are modified for specific use cases.
"""
'''

from numpy.linalg import LinAlgError
import numpy as np
from scipy.stats import norm
from typing import Union, Set, Tuple, Dict, List
import pandas as pd
import time
from collections import OrderedDict


def CItest_method(
    data: Union[np.ndarray, pd.DataFrame],
    method_type: str = "FisherZ",
    alpha: float = 0.05,
    **kwargs,
):
    """
    Perform conditional independence test.

    Parameters:
    data: np.ndarray, pd.DataFrame - DataFrame containing the data to compute the test.
    method_type: str - Method type for CI test. Options are 'FisherZ', 'G_sq', or 'D_sep'.
    alpha: float - Significance level.

    """
    if method_type == "FisherZ":
        return FisherZ_Test_(data, alpha=alpha, **kwargs)
    elif method_type == "G_sq":
        raise NotImplementedError("G_sq test is not implemented yet.")
    else:
        raise ValueError("Invalid method_type. Choose 'FisherZ', 'G_sq'.")


class CI_Test_:
    def __init__(self, data: Union[np.ndarray, pd.DataFrame], **kwargs):

        if isinstance(data, np.ndarray):
            self.data = pd.DataFrame(
                data, columns=[f"V{i+1}" for i in range(data.shape[1])]
            )
        else:
            self.data = data
        self.data_columns = {col: idx for idx, col in enumerate(data.columns)}
        self.data = self.data.to_numpy()
        self.sample_size, self.num_nodes = data.shape
        if kwargs.get("alpha") is None:
            self.alpha = 0.05
        else:
            self.alpha = kwargs["alpha"]

        self._validate_data()
        self._ci_cache = OrderedDict()  # LRU cache
        self.max_cache_size = kwargs.get("Max_cache_size", 3000)  # Maximum cache size
        self._ci_num = 0  # The number of tests.
        self.Max_time = kwargs.get("Max_time", None)
        if self.Max_time is not None:
            self.start_time = time.time()  # Start time for the simulation.

    def _validate_data(self) -> bool:
        """
        Validate that the data does not contain missing or infinite values.

        Returns:
        bool - True if the data is valid, False otherwise.
        """
        if np.isnan(self.data).any():  # Use np.isnan for NumPy arrays
            raise ValueError("Data contains missing values (NaN).")
        if np.isinf(self.data).any():  # Use np.isinf for infinite values
            raise ValueError("Data contains infinite values.")
        return True

    def _generate_cache_key(self, X: int, Y: int, S: List[int]) -> str:
        """
        Generate a unique cache key for the given variables.

        Parameters:
        ----------
        X, Y: int
            Variables to test for conditional independence.
        S: List[int]
            Conditioning set.

        Returns:
        -------
        str - A unique cache key.
        """
        S_sorted = sorted(S)  # Ensure consistent order
        return f"{X}-{Y}|{'-'.join(map(str, S_sorted))}"

    def _add_to_cache(self, key: str, p_value: float):
        """
        Add a new entry to the cache. If the cache exceeds the maximum size, remove the least recently used item.

        Parameters:
        ----------
        key: str
            Cache key.
        value: dict
            Cache value.
        """
        if key in self._ci_cache:
            # Move the key to the end to mark it as recently used
            self._ci_cache.move_to_end(key)
        self._ci_cache[key] = p_value
        if len(self._ci_cache) > self.max_cache_size:
            # Remove the least recently used item
            self._ci_cache.popitem(last=False)

    def _get_from_cache(self, key: str) -> Union[float, None]:
        """
        Retrieve a value from the cache.

        Parameters:
        ----------
        key: str
            Cache key.

        Returns:
        -------
        float or None - The cached value, or None if the key is not in the cache.
        """
        if key in self._ci_cache:
            # Move the key to the end to mark it as recently used
            self._ci_cache.move_to_end(key)
            return self._ci_cache[key]
        return None

    def _formatted_XYS(
        self,
        X: Union[int, str],
        Y: Union[int, str],
        condition_set: Union[list[Union[int, str]], None] = None,
    ) -> Tuple[int, int, List[int]]:
        """
        Format the input variables X, Y, and condition_set.
        Convert string variable names to their corresponding indices in the data.
        """

        if isinstance(X, str) and isinstance(Y, str):
            X = self.data_columns[X]
            Y = self.data_columns[Y]

        if X > Y:
            X, Y = Y, X

        if condition_set is not None:
            if not isinstance(condition_set, list):
                raise ValueError("The condition_set of CI_test must is list.")

            if len(condition_set) > 0:
                if all(isinstance(s, str) for s in condition_set):
                    condition_set = [self.data_columns[s] for s in condition_set]
                elif all(isinstance(s, int) for s in condition_set):
                    condition_set = condition_set
                else:
                    raise ValueError(
                        "The condition_set of CI_test must is list of str or int."
                    )
                condition_set = sorted(condition_set)

            if any(not isinstance(s, int) for s in condition_set):
                raise ValueError(
                    "The _formatted_XYS of the CI_test function contains an error."
                )

        if not isinstance(X, int) or not isinstance(Y, int):
            raise ValueError("The formatted function has lost its validity.")

        return X, Y, condition_set


class FisherZ_Test_(CI_Test_):

    def __init__(self, data: Union[np.ndarray, pd.DataFrame], **kwargs):

        super().__init__(data, **kwargs)
        # Correlation matrix
        self.Corr_mat = np.corrcoef(self.data, rowvar=False)

    def __call__(
        self,
        X: Union[int, str],
        Y: Union[int, str],
        S: List[Union[int, str]] = [],
        **kwargs,
    ) -> Tuple[bool, float]:
        """
        Source code from the R package pcalg. 'https://github.com/cran/pcalg/blob/930fe476875ddc00528dbaacb6f48f2791ab12e3/R/pcalg.R#L2839'

        gaussCItest <- function(x,y,S,suffStat)
        Perform Gaussian Conditional Independence Test.

        Parameters:
        X, Y: int or str - Variables to test for conditional independence.
        S: list - Conditioning set.

        Returns:
        bool - CI is True means independent, False means dependent.
        float - p-value of the test.
        """

        def zStat(X: int, Y: int, S: list, C: np.ndarray, n: int) -> float:
            """
            Calculate Fisher's z-transform statistic of partial correlation.

            Parameters:
            X, Y: int - Variables to test for conditional independence.
            S: list - Conditioning set.
            C: np.ndarray - Correlation matrix.
            n: int - Sample size.

            Returns:
            float - z-statistic.
            """
            r = pcorOrder(X, Y, S, C)
            if r is None:
                return 0
            return np.sqrt(n - len(S) - 3) * 0.5 * np.log((1 + r) / (1 - r))

        def pcorOrder(
            i: int, j: int, k: list, C: np.ndarray, cut_at: float = 0.9999999
        ) -> float:
            """
            Compute partial correlation.

            Parameters:
            i, j: int - Variables to compute partial correlation.
            k: list - Conditioning set.
            C: np.ndarray - Correlation matrix.

            Returns:
            float - Partial correlation coefficient.
            """
            if len(k) == 0:
                r = C[i, j]
            elif len(k) == 1:
                r = (C[i, j] - C[i, k[0]] * C[j, k[0]]) / np.sqrt(
                    (1 - C[j, k[0]] ** 2) * (1 - C[i, k[0]] ** 2)
                )
            else:
                try:
                    sub_matrix = C[np.ix_([i, j] + k, [i, j] + k)]
                    PM = np.linalg.pinv(sub_matrix)
                    r = -PM[0, 1] / np.sqrt(PM[0, 0] * PM[1, 1])
                except LinAlgError:
                    return None
            if np.isnan(r):
                return 0
            return min(cut_at, max(-cut_at, r))

        if self.Max_time is not None:
            end_time = time.time()
            if end_time - self.start_time > self.Max_time:
                raise TimeoutError(
                    f"The simulation exceeded maximum time limit of {self.Max_time} seconds. The running time is {end_time - self.start_time} seconds."
                )

        X, Y, S = self._formatted_XYS(X, Y, S)
        key = self._generate_cache_key(X, Y, S)
        p_value = self._get_from_cache(key)
        if p_value is not None:
            CI = p_value > self.alpha
            return CI, p_value
        z = zStat(X, Y, S, self.Corr_mat, self.sample_size)
        p_value = 2 * norm.cdf(-abs(z))
        CI = p_value > self.alpha
        self._add_to_cache(key, p_value)
        self._ci_num += 1
        return CI, p_value
