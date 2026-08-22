# Configurations

Store YAML configuration files by scientific topic:

- `baseline/` for local-network simulations and scans.
- `disorder/` for non-local connectivity simulations and scans.
- `driven/` for input-driven simulations and tasks.
- `working_memory/` for RLS/FORCE training and evaluation.
- `analyses/` for supporting spectral and dynamic analyses.
- `extensions/` for secondary scientific workflows.

Each script must state which configuration it uses. Use `_quick` for a small fixed-seed check.

The mode scan configurations use these sections:

- `run` defines the output root, experiment name, and optional run ID.
- `network` defines connectivity and network size.
- `simulation` defines time, seeds, and stability limits.
- `scan` defines scanned parameter values.
- `analysis` defines metric and plot limits.

Command-line arguments can override the output root, run ID, seed, and `N`.
