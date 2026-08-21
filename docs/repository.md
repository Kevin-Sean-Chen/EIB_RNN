# Repository organization

This document defines the role of each top-level directory. Apply these rules to new files. Move old files only after a scientific result check.

## `configs/`

Store YAML files that define simulation, scan, and task parameters. A configuration must not contain Python code or generated arrays.

## `src/`

Store reusable scientific functions and classes. Code belongs here when two scripts can use it, when a test must call it directly, or when it defines a model, stimulus, metric, or plot operation.

Source code must not use a fixed output path. It must not read command-line arguments.

## `scripts/`

Store executable scientific workflows. A script can run one simulation, run a parameter scan, train a model, or evaluate a result.

A script loads configuration, calls code from `src/`, and writes results. Keep new scripts thin.

Use these name prefixes:

- `run_` for one simulation or analysis.
- `scan_` for a parameter scan.
- `train_` for model training.
- `evaluate_` for model evaluation.

## `notebooks/`

Store demonstration and exploration notebooks. A notebook must call shared code from `src/`. Do not use a notebook as the only location of a scientific method.

## `data/`

Store required input data and small reference data. Do not store generated figures, videos, model states, or scan results here.

## `output/`

Store all generated results. This includes figures, arrays, tables, model states, logs, and videos. Git ignores generated files in this directory.

Each run should have one directory that contains its resolved configuration and results.

## `tests/`

Store fast scientific and software checks. Use small networks and fixed random seeds. The standard test set must not run a long parameter scan.

## `archive/`

Store code that is not part of the current workflow. Do not refactor archived code. Add a note that explains its source and status.

## `docs/`

Store model definitions, workflow instructions, and reproduction procedures.
