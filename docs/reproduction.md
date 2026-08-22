# Reproduce an analysis

Create and activate the environment:

```bash
conda env create -f environment.yaml
conda activate EI2D
```

Run the canonical baseline local network:

```bash
python scripts/run_local.py \
  --config configs/simulations/local.yaml \
  --show
```

For a faster `N=15` check, use `configs/simulations/local_quick.yaml`.

Run the local-network mode scan from the repository root:

```bash
python scripts/scan_local_network_modes.py \
  --config configs/scans/local_network_modes.yaml
```

Run the related scans:

```bash
python scripts/scan_K_rhoF_modes.py \
  --config configs/scans/K_rhoF_modes.yaml

python scripts/scan_spatial_statistics.py \
  --config configs/scans/spatial_statistics.yaml
```

Each command creates one directory under `output/scans/`. The directory contains:

- `config.yaml`: resolved scientific parameters.
- `metadata.json`: creation time, Git commit, command, and software versions.
- `metrics.csv`: one summary row for each scan point.
- `results.npz`: complete numerical arrays.
- `summary.png`: summary figure.

Use `--run-id` only when you need a fixed directory name. The command stops if that directory already exists. This rule prevents accidental result replacement.

## Driven moving-dot task

Run one driven moving-dot simulation:

```bash
python scripts/run_driven_dot.py \
  --config configs/simulations/driven_dot.yaml
```

For a faster `N=15` check, use `configs/simulations/driven_dot_quick.yaml`.

Scan moving-dot tracking across `K`:

```bash
python scripts/scan_driven_dot.py \
  --config configs/tasks/driven_dot_tracking.yaml
```

For a short scan, use `configs/tasks/driven_dot_tracking_quick.yaml`.

These workflows use input and response center of mass. The tracking scan measures peak lag and peak overlap-normalized cross-correlation.

## Low-rank disorder simulation

Run the legacy-matching disorder simulation:

```bash
python scripts/run_disorder.py \
  --config configs/simulations/disorder.yaml \
  --show
```

For a faster `N=15` check, use `configs/simulations/disorder_quick.yaml`.

Set `pattern_type` to `random` or `gabor`. Set `rank` to `1` or `2`.

Scan rank-two random disorder across `K` and strength:

```bash
python scripts/scan_disorder_strength.py \
  --config configs/scans/disorder_strength.yaml
```

For a short `N=15` scan, use `configs/scans/disorder_strength_quick.yaml`.
