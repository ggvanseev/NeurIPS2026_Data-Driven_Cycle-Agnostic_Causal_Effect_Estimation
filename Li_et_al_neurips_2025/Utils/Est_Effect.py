import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def est_reg_con(
    Tr: str, Y: str, adjset: list[str], dataset: pd.DataFrame
) -> tuple[float, float]:
    """
    Estimate the Average Treatment Effect (ATE) using regression adjustment.
    """

    if isinstance(dataset, np.ndarray):
        dataset = pd.DataFrame(
            dataset, columns=[f"V{i}" for i in range(dataset.shape[1])]
        )

        # Convert Tr and Y indices to column names
        Tr_col = dataset.columns[Tr]
        Y_col = dataset.columns[Y]

    else:
        Tr_col = Tr
        Y_col = Y

    if any(pd.isna(adjset)):
        est_ATE = np.nan
        est_sd = np.nan

    else:
        if len(adjset) == 0:
            formula = f"{Y_col} ~ {Tr_col}"
        else:
            formula = f"{Y_col} ~ {Tr_col} + " + " + ".join(adjset)
        model = smf.ols(formula, data=dataset).fit()
        est_ATE = model.params[Tr_col]
        est_sd = model.bse[Tr_col]

    return est_ATE, est_sd
