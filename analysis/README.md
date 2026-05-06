# Analysis

This folder runs LSAS on generated datasets, summarises the results, and creates
figures for inspection or paper output.

The usual flow is:

```text
generated data -> LSAS result CSVs -> metric summaries -> figures/tables
```

Generated data is expected under:

```text
NeurIPS_2026/data/generated
```

## Files

- `run_analysis.py`
  Runs LSAS on generated dataset CSVs.
- `run_dataset_workflow.ipynb`
  Notebook for running LSAS on one chosen dataset.
- `summarise_lsas_metrics.py`
  Converts LSAS result CSVs into metric summaries.
- `create_paper_metric_summary.py`
  Pools metric summaries into a paper-ready table.
- `plot_lsas_metrics.py`
  Creates metric figures.
- `plot_runtime_by_nodes.py`
  Creates runtime-by-node-count figures.
- `rename_result_csvs.py`
  Converts long result filenames to compact names.
- `results/`
  Default location for per-dataset LSAS outputs.
- `results_metric/`
  Default location for metric summaries and paper summary tables.
- `figures/`
  Default location for generated plots.
- `logs/`
  Default location for analysis logs.

## Run LSAS

From the repository root:

```bash
python NeurIPS_2026/analysis/run_analysis.py
```

This recursively finds CSV files below `NeurIPS_2026/data/generated`, splits
each dataset into graph blocks, runs LSAS once per Markov blanket method, and
writes one `*_lsas.csv` file per input dataset.

The LSAS code used here is the local augmented copy in
`NeurIPS_2026/Li_et_al_neurips_2025`. Compared with the original LSAS code, it
supports multiple Markov blanket discovery methods. This is why one graph can
produce several LSAS rows, one for each selected method.

Small smoke test:

```bash
python NeurIPS_2026/analysis/run_analysis.py --limit-files 1 --limit-graphs 1 --overwrite
```

Run one file:

```bash
python NeurIPS_2026/analysis/run_analysis.py --file NeurIPS_2026/data/generated/gaussian/output.csv
```

Choose Markov blanket methods:

```bash
python NeurIPS_2026/analysis/run_analysis.py --mb-methods TC IAMB Fast_IAMB HITON_MB
```

Use different input or output folders:

```bash
python NeurIPS_2026/analysis/run_analysis.py --data-root NeurIPS_2026/data/generated/random --results-dir NeurIPS_2026/analysis/results_random
```

## LSAS Output

Outputs are written under:

```text
NeurIPS_2026/analysis/results
```

The input subfolder is preserved and `_lsas.csv` is added to the filename.

Example:

```text
input:
NeurIPS_2026/data/generated/gaussian/output.csv

output:
NeurIPS_2026/analysis/results/gaussian/output_lsas.csv
```

Each output row represents one graph block and one Markov blanket method.
Important columns include:

- source dataset and dataset group
- graph block index and graph metadata
- visible node columns
- treatment and outcome columns
- Markov blanket method
- learned treatment and outcome Markov blankets
- suggested and selected adjustment sets
- number of independence tests
- CPU and wall-clock runtime
- LSAS status and error message

## Summarise Metrics

After LSAS result CSVs exist, create a metric summary:

```bash
python NeurIPS_2026/analysis/summarise_lsas_metrics.py NeurIPS_2026/analysis/results --out NeurIPS_2026/analysis/results_metric
```

When `--out` is a directory, the output file is:

```text
metrics_summary.csv
```

The input can be a CSV file, a folder, or multiple files/folders. Folder inputs
are searched recursively.

The metric summary has one row per input file per Markov blanket method. It
includes dataset metadata, adjustment-set quality metrics, edge-identification
metrics, independence-test counts, and runtime summaries.

Optionally write per-sample metrics:

```bash
python NeurIPS_2026/analysis/summarise_lsas_metrics.py NeurIPS_2026/analysis/results --per-sample-out NeurIPS_2026/analysis/results_metric/per_sample_metrics.csv
```

By default, sigma-backdoor validation uses the cached checker in
`NeurIPS_2026/data/sigma_backdoor.py`. To compare with the slower original
path-enumeration check:

```bash
python NeurIPS_2026/analysis/summarise_lsas_metrics.py NeurIPS_2026/analysis/results --out NeurIPS_2026/analysis/results_metric/metrics_summary_slow_check.csv --no-fast-sigma-backdoor
```

## Paper Metric Summary

Create a pooled paper-ready table:

```bash
python NeurIPS_2026/analysis/create_paper_metric_summary.py --with-xy NeurIPS_2026/analysis/results_metric/metrics_summary_medium_xy_edge.csv --without-xy NeurIPS_2026/analysis/results_metric/metrics_summary_medium_no_xy_edge.csv --out NeurIPS_2026/analysis/results_metric/paper_metric_summary.csv
```

If there is no separate no-edge summary, provide only the with-edge summary:

```bash
python NeurIPS_2026/analysis/create_paper_metric_summary.py NeurIPS_2026/analysis/results_metric/metrics_summary_medium_xy_edge.csv --out NeurIPS_2026/analysis/results_metric/paper_metric_summary.csv
```

The paper summary groups by node count, sample size, graph type, pretreatment
setting, and Markov blanket method.

## Plot Metrics

Create figures from the default metric summary:

```bash
python NeurIPS_2026/analysis/plot_lsas_metrics.py
```

Choose paths explicitly:

```bash
python NeurIPS_2026/analysis/plot_lsas_metrics.py --summary NeurIPS_2026/analysis/results_metric/metrics_summary.csv --out NeurIPS_2026/analysis/figures
```

Useful plotting options:

- `--formats png pdf svg`
  Write one or more image formats.
- `--lines`
  Connect points with lines.
- `--comparison-mb-method Fast_IAMB`
  Choose the Markov blanket method used in comparison figures.
- `--one-mb-only`
  Use only the comparison method in all figures.
- `--paper-mode`
  Use larger text and paper-oriented filenames.
- `--no-legends`
  Omit legends.

## Rename Result CSVs

Preview compact filenames:

```bash
python NeurIPS_2026/analysis/rename_result_csvs.py NeurIPS_2026/analysis/results --recursive
```

Apply the renames:

```bash
python NeurIPS_2026/analysis/rename_result_csvs.py NeurIPS_2026/analysis/results --recursive --apply
```

The renamer skips files that already have compact names. Use `--debug` to show
examples of skip reasons and `--list-found` to print every discovered CSV.

## Notes

- `--max-k auto` in `run_analysis.py` uses all visible covariates except the
  treatment and outcome.
- Filename-derived metadata is preferred by the metric summariser when compact
  result filenames are available.
- Empty selected sets are counted separately in `empty_set_fraction`.
- Invalid non-empty adjustment sets are excluded from adjustment metrics.
