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

## Figure 1 panels B--G

Create the matched spontaneous-activity panels:

```bash
python scripts/figures/figure1_spontaneous.py \
  --config configs/figures/figure1_spontaneous.yaml
```

Use `configs/figures/figure1_spontaneous_quick.yaml` for a short workflow check.
The production configuration uses `K = [1, 100, 10000]`, `tau_e = 0.01`, and
`tau_i = 0.02`. It saves each panel as PDF and PNG under
`output/figures/figure1/`. Numerical results and provenance are in the
`data/` subfolder. The workflow does not assemble the full figure.

Assemble the approved panel PNG files without rerunning simulations:

```bash
python scripts/figures/assemble_figure1.py
```

This command writes `figure1.png` and `figure1.pdf` beside the separate panels.

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

Run same-time rigid-shift reconstruction:

```bash
python scripts/driven/train_rigid_reconstruction.py \
  --config configs/driven/rigid_reconstruction.yaml
```

Use `configs/driven/rigid_reconstruction_quick.yaml` for an `N=15` check.

The default configuration uses one-pass RLS and the canonical driven-network condition: `dt=0.0001`, `K=100`, additive baseline input, stimulus gain `10`, and initial scale `0.1`. Use `configs/driven/rigid_reconstruction_adam.yaml` only to compare with the replacement-baseline optimizer in the original ReLU script.

Run the same-size non-spatial control:

```bash
python scripts/driven/train_rigid_reconstruction.py \
  --config configs/driven/rigid_reconstruction_non_spatial.yaml
```

The control has `N²` random recurrent units and the same `N²+1` readout features as the spatial excitatory population. The extra feature is the bias. Both models multiply the baseline and stimulus by `sqrt(K)`. They use the same movie, target, timing, RLS rule, initialization scale, and evaluation perturbation. Each row of the random recurrent matrix has zero sum. This condition gives an explicit balance between its positive and negative recurrent weights. The non-spatial recurrent gain is `0.8`, and its connection probability is `0.5`.

Run the strongly coupled random E/I control:

```bash
python scripts/driven/train_rigid_reconstruction.py \
  --config configs/driven/rigid_reconstruction_random_ei.yaml
```

This control has the same E and I population sizes, in-degree `K`, coupling values, time constants, strong-drive scaling, and excitatory readout size as the spatial network. Each random connectivity row has exactly `K` inputs. Only the recurrent topology changes from local spatial kernels to random connections.

Use `configs/driven/rigid_reconstruction_random_ei_quick.yaml` for a short check.

The metrics file includes a driven finite-time Lyapunov exponent. A positive value indicates perturbation growth under the fixed task input. A negative value indicates contraction toward an input-locked trajectory.

Scan the spatial smoothing width for both models:

```bash
python scripts/driven/scan_rigid_smoothing.py \
  --config configs/driven/rigid_reconstruction_smoothing_scan.yaml
```

The scan reports smoothing width in grid pixels. Larger values produce smoother textures with longer spatial correlations.

Compare raw input with one fixed spatial permutation:

```bash
python scripts/driven/scan_rigid_smoothing.py \
  --config configs/driven/rigid_reconstruction_shuffle_scan.yaml
```

The fixed permutation is identical at all time steps. It preserves pixel values and temporal identity, but it removes local spatial correlations.

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

## Dynamic mode decomposition

Analyze excitatory activity from the local spatial network:

```bash
python scripts/analyses/run_dmd.py \
  --config configs/analyses/dmd.yaml
```

For a faster `N=15` check, use `configs/analyses/dmd_quick.yaml`. Frequencies and growth rates use units of inverse seconds. The legacy script analyzed the `mue_all` field. That field analysis is not yet replaced.

## Working memory with RLS

Train the spatial reservoir with online RLS/FORCE updates:

```bash
python scripts/working_memory/train_force.py \
  --config configs/working_memory/force_spatial.yaml
```

Train the non-spatial control with the same learning rule:

```bash
python scripts/working_memory/train_force.py \
  --config configs/working_memory/force_non_spatial.yaml
```

Use `force_spatial_quick.yaml` or `force_non_spatial_quick.yaml` for a short check. These workflows have fixed recurrent weights. Only the output and memory readouts change through direct RLS updates. They do not use Adam or gradient updates.

Reproduce the legacy spatial ridge performance across `K`:

```bash
python scripts/working_memory/scan_K.py \
  --config configs/working_memory/K_scan.yaml
```

Use `configs/working_memory/K_scan_quick.yaml` for a short check. The scan reports post-go MSE and R2 for the output and memory readouts. The legacy-matched configuration uses ridge fitting, no feedback, and no field clamp. Set `learning_method: rls`, `feedback_gain: 0.001`, and `field_clip: 100.0` only for a separate FORCE/RLS comparison.
