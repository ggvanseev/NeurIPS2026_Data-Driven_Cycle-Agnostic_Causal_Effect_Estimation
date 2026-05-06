# NeurIPS 2026 Project

This folder contains the work for the NeurIPS 2026 LSAS experiments.

The goal is simple: create artificial causal datasets where the true answer is
known, run LSAS on those datasets, and check how well LSAS performs.

## The Big Picture

The project has three main parts:

- `data/`
  Makes the synthetic datasets.
- `Li_et_al_neurips_2025/`
  Contains the augmented LSAS method code.
- `analysis/`
  Runs LSAS and turns the results into tables and figures.

In plain language, the workflow is:

1. Make synthetic causal data.
2. Run LSAS on that data.
3. Compare the LSAS output with the known truth.
4. Summarise the results.
5. Make figures for inspection or the paper.

## Start Here

For an interactive overview, open:

```text
lsas_data_workflow.ipynb
```

That notebook connects the main steps in one place.

## Where To Go Next

Use the folder READMEs for technical details.

### Create Data

Go to:

```text
data/README.md
```

This explains how to generate synthetic datasets and what the generated CSV
files contain.

### Understand The LSAS Code

Go to:

```text
Li_et_al_neurips_2025/README.md
```

This explains the local LSAS implementation and how it is called by the rest of
the project. The local copy has been extended compared with the original LSAS
code so experiments can run several Markov blanket discovery methods.

### Run Experiments

Go to:

```text
analysis/README.md
```

This explains how to run LSAS on generated datasets, summarise metrics, and
create figures.

## Setup

This project uses Python 3.11 and the packages listed in:

```text
requirements.txt
```

The technical setup commands are kept in the folder-level READMEs so this page
can stay easy to scan.
