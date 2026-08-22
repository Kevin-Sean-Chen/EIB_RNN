# Reproduce an analysis

Create and activate the environment:

```bash
conda env create -f environment.yaml
conda activate EI2D
```

Run the canonical baseline local network:

```bash
python scripts/baseline/run_local.py \
  --config configs/baseline/local.yaml \
  --show
```

For a faster `N=15` check, use `configs/baseline/local_quick.yaml`.

Run the local-network mode scan from the repository root:

```bash
python scripts/baseline/scan_local_network_modes.py \
  --config configs/baseline/local_network_modes.yaml
```

Run the related scans:

```bash
python scripts/baseline/scan_K_rhoF_modes.py \
  --config configs/baseline/K_rhoF_modes.yaml

python scripts/baseline/scan_spatial_statistics.py \
  --config configs/baseline/spatial_statistics.yaml
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
python scripts/driven/run_driven_dot.py \
  --config configs/driven/dot.yaml
```

For a faster `N=15` check, use `configs/driven/dot_quick.yaml`.

Scan moving-dot tracking across `K`:

```bash
python scripts/driven/scan_driven_dot.py \
  --config configs/driven/dot_tracking.yaml
```

For a short scan, use `configs/driven/dot_tracking_quick.yaml`.

These workflows use input and response center of mass. The tracking scan measures peak lag and peak overlap-normalized cross-correlation.

## Low-rank disorder simulation

Run the legacy-matching disorder simulation:

```bash
python scripts/disorder/run_disorder.py \
  --config configs/disorder/run.yaml \
  --show
```

For a faster `N=15` check, use `configs/disorder/run_quick.yaml`.

Set `pattern_type` to `random` or `gabor`. Set `rank` to `1` or `2`.

Scan rank-two random disorder across `K` and strength:

```bash
python scripts/disorder/scan_disorder_strength.py \
  --config configs/disorder/strength_scan.yaml
```

For a short `N=15` scan, use `configs/disorder/strength_scan_quick.yaml`.

Scan rank-one Gabor disorder across `K` and phase:

```bash
python scripts/disorder/scan_rank_one.py \
  --config configs/disorder/rank_one.yaml
```

For a short `N=15` scan, use `configs/disorder/rank_one_quick.yaml`.

## Supporting spectral analysis

Run the legacy block-operator spectrum analysis:

```bash
python scripts/analyses/run_spectral.py \
  --config configs/analyses/spectral.yaml
```

For a faster `N=15` check, use `configs/analyses/spectral_quick.yaml`.
