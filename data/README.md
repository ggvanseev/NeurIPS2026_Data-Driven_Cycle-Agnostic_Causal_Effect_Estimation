# Data

This folder creates the synthetic datasets used by the NeurIPS 2026 LSAS
experiments. It can generate linear or nonlinear structural causal models
(SCMs), sample data from them, and export the samples plus graph metadata to
CSV files.

The analysis code reads the generated CSV files from:

```text
NeurIPS_2026/data/generated
```

## Files

- `generate_cyclic_scm_csv.py`
  Command-line exporter for generated datasets.
- `synthetic_cyclic_scm.py`
  Linear SCM graph generation and sampling.
- `synthetic_cyclic_scm_nonlinear.py`
  Nonlinear SCM generation using the same graph construction as the linear
  generator.
- `calc_CE.py`
  Helper functions for causal-effect calculations.
- `sigma_backdoor.py`
  Sigma-backdoor existence and adjustment-set checks.
- `visualize_synthetic_cyclic_scm.ipynb`
  Notebook for visually checking generated graph structures.

## Generate Data

Run the exporter from the repository root:

```bash
python NeurIPS_2026/data/generate_cyclic_scm_csv.py NeurIPS_2026/data/generated/output.csv --scm-type linear --n-nodes 8 --n-edges 12 --n-samples 1000 --n-observed 5 --treatment 0 --outcome 1 --sigma-low 0.5 --sigma-high 1.0 --noise-type gaussian --cyclic --rng-seed 42
```

Nonlinear example:

```bash
python NeurIPS_2026/data/generate_cyclic_scm_csv.py NeurIPS_2026/data/generated/output.csv --scm-type nonlinear --n-nodes 8 --n-edges 12 --n-samples 1000 --n-observed 5 --treatment 0 --outcome 1 --sigma-low 0.5 --sigma-high 1.0 --noise-type random --cyclic --rng-seed 42
```

The output filename is automatically expanded with useful metadata such as graph
family, SCM type, sample count, node count, edge count, treatment, outcome, and
timestamp.

## Important Options

- `--scm-type linear` or `--scm-type nonlinear`
  Choose the sampling model.
- `--cyclic` or `--acyclic`
  Choose whether the graph may contain directed cycles.
- `--noise-type gaussian`, `--noise-type non_gaussian`, or `--noise-type random`
  Choose the node-noise family.
- `--no-pretreatment-flag`
  Disable the default pretreatment graph structure.
- `--allow-disconnected`
  Allow disconnected graphs. By default, weak connectivity is required.
- `--allow-two-cycles`
  Allow reciprocal directed pairs in cyclic graphs.
- `--sigma-backdoor`
  Require at least one valid sigma-backdoor adjustment set during generation.
  By default, this check is skipped.
- `--x-non-singleton-scc`
  Allow the treatment node to be part of a larger strongly connected component.
- `--target-operator-norm 0.8`
  Change the operator-norm target used when edge weights are rescaled.
- `--debug`
  Store the full requested parameter set in the first CSV metadata row.

## CSV Contents

Each generated CSV stores one sample per row. The first row for each graph
contains graph-level metadata; repeated metadata fields are written as `NA` in
later rows for the same graph.

Common columns include:

- `sample_id`
- `graph_family`
- `scm_type`
- `treatment`
- `outcome`
- `n_nodes`
- `n_edges`
- `n_visible_nodes`
- `true_ce`
- `has_sigma_backdoor`
- `sigma_backdoor_checked`
- `graph_structure`
- one column per visible node, such as `node_0`, `node_1`, or `node_4`

The visible node set always includes the treatment and outcome nodes.

The `graph_structure` column stores directed edges in a compact form:

```text
{0->1,2->0,4->3}
```

## Graph Conventions

The adjacency matrix convention is:

```text
W[i, j] != 0 means j -> i
```

When pretreatment structure is enabled:

- the treatment node is allowed to point to the outcome node
- the outcome remains a descendant of the treatment
- other nodes may point into the treatment
- the outcome has no outgoing edges

For cyclic graphs, `--allow-two-cycles` controls whether both `i -> j` and
`j -> i` may appear for the same node pair.

## Noise

The generators sample one sigma value per node from `--sigma-low` to
`--sigma-high`.

- `gaussian`
  Every node uses Gaussian noise.
- `non_gaussian`
  Every node uses uniform noise from `[-sigma, sigma]`.
- `random`
  Each node independently chooses Gaussian or uniform noise.

## Causal Effect Column

The `true_ce` column depends on the SCM type:

- cyclic linear SCMs: hard-intervention effect from the linear system
- acyclic linear SCMs: direct matrix entry for treatment to outcome
- nonlinear SCMs: `NA`

## Visual Check

Open `visualize_synthetic_cyclic_scm.ipynb` to inspect generated graphs. The
notebook draws the graph, labels the treatment as `X`, labels the outcome as
`Y`, and helps check edge directions, connectivity, and pretreatment behavior.
