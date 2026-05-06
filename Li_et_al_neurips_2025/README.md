# LSAS Method Code

This folder contains the local copy of the LSAS implementation used by the
NeurIPS 2026 experiments.

LSAS stands for Local Search Adjustment Set. In this project, it is used to
suggest adjustment sets for estimating the effect of a treatment variable on an
outcome variable when some variables may be hidden.

## Local Augmentation

This copy has been augmented relative to the original LSAS code. The main
change is support for multiple Markov blanket discovery backends instead of a
single original blanket learner.

Supported methods are:

- `TC`
  Total-conditioning learner, used to preserve the original behavior.
- `IAMB`
  Incremental Association Markov Blanket.
- `Fast_IAMB`
  Fast-IAMB.
- `HITON_MB`
  HITON Markov blanket discovery.

The active method is selected with the `mb_method` argument.

## Files

- `LSAS.py`
  Main LSAS implementation, augmented with selectable Markov blanket discovery.
- `example.py`
  Minimal example for running LSAS directly.
- `test_mb_discovery.py`
  Checks for Markov blanket discovery methods.
- `Utils/CI_test.py`
  Conditional independence testing helpers.
- `Utils/Est_Effect.py`
  Effect-estimation helpers.
- `Utils/MB_discovery.py`
  Markov blanket discovery methods added for these experiments.

## Main Entry Point

The main function is `alg_LSAS` in `LSAS.py`.

Parameters:

- `data`
  A pandas DataFrame containing the observed variables.
- `Tr_X`
  Treatment column name.
- `Out_Y`
  Outcome column name.
- `mb_method`
  Markov blanket discovery backend. Supported values are `TC`, `IAMB`,
  `Fast_IAMB`, and `HITON_MB`. Use `TC` for the behavior closest to the
  original implementation.

Returns:

- `VASs`
  Suggested valid adjustment sets.
- `ATE`
  Estimated average treatment effects for the adjustment sets.
- `CI_num`
  Number of conditional independence tests.

## Example

```python
import re
import pandas as pd
from LSAS import alg_LSAS

data_v = pd.read_csv("Data/example_data_5000.csv")
data_o = data_v[[col for col in data_v.columns if not re.match(r"^L\d+$", col)]]

result = alg_LSAS(
    data_o,
    Tr_X="X",
    Out_Y="Y",
    mb_method="Fast_IAMB",
)

print(result)
```

## Use Inside This Project

Most project runs should use the wrapper in:

```text
NeurIPS_2026/analysis/run_analysis.py
```

That wrapper loads generated CSV datasets, prepares the visible node columns,
runs this augmented LSAS code for each graph block and each selected Markov
blanket method, and stores standardised result files.

Use this folder directly only when you want to inspect or test the LSAS method
itself.

## Citation

If you use the LSAS method code, cite:

```bibtex
@inproceedings{li2025local,
  title={Local Learning for Covariate Selection in Nonparametric Causal Effect Estimation with Latent Variables},
  author={Zheng Li, Xichen Guo, Feng Xie, Yan Zeng, Hao Zhang, and Zhi Geng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
