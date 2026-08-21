# Configurations

Store YAML configuration files in these groups:

- `simulations/` for one simulation.
- `scans/` for parameter scans.
- `tasks/` for training and evaluation tasks.

Each script must state which configuration it uses.

The mode scan configurations use these sections:

- `run` defines the output root, experiment name, and optional run ID.
- `network` defines connectivity and network size.
- `simulation` defines time, seeds, and stability limits.
- `scan` defines scanned parameter values.
- `analysis` defines metric and plot limits.

Command-line arguments can override the output root, run ID, seed, and `N`.
