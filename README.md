# EIB_RNN

EIB_RNN studies excitation-inhibition balanced recurrent neural networks with two-dimensional spatial connectivity. The repository contains reproducible workflows for spontaneous dynamics, low-rank disorder, input-driven computation, working memory, and mode analysis.

## Model

The basic rate equations are

$$
\tau_e \frac{\partial h_e(x,t)}{\partial t}
= -h_e + W_{ee}g_e*\phi(h_e) + W_{ei}g_i*\phi(h_i) + \mu_e,
$$

$$
\tau_i \frac{\partial h_i(x,t)}{\partial t}
= -h_i + W_{ie}g_e*\phi(h_e) + W_{ii}g_i*\phi(h_i) + \mu_i.
$$

Here, $\phi$ is the ReLU function, and $g_e$ and $g_i$ are spatial Gaussian kernels. The weights satisfy the E/I balance condition. The coupling and external input scale with $\sqrt{K}$, where $K$ is the connection scale.

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yaml
conda activate EI2D
```

Run commands from the repository root.

## Quick start

Run a small baseline simulation:

```bash
python scripts/baseline/run_local.py \
  --config configs/baseline/local_quick.yaml \
  --show
```

The workflow writes its results under `output/`. It saves the resolved configuration, run metadata, numerical results, metrics, and a summary figure.

## Main workflows

| Topic | Entry point | Purpose |
|---|---|---|
| Baseline dynamics | `scripts/baseline/run_local.py` | Simulate the local spatial E/I network. |
| Spatial mode scans | `scripts/baseline/scan_local_network_modes.py` | Compare local, network, and data-driven modes. |
| Low-rank disorder | `scripts/disorder/run_disorder.py` | Measure the effect of non-local structure. |
| Moving-dot response | `scripts/driven/run_driven_dot.py` | Measure input tracking without a trained readout. |
| Rigid reconstruction | `scripts/driven/train_rigid_reconstruction.py` | Train and compare spatial and random E/I reservoirs. |
| Working memory | `scripts/working_memory/train_force.py` | Train readouts with online RLS/FORCE updates. |
| Dynamic mode decomposition | `scripts/analyses/run_dmd.py` | Analyze excitatory activity modes and rates. |
| Spectral analysis | `scripts/analyses/run_spectral.py` | Analyze the connectivity operator spectrum. |

Each main workflow has a YAML file in `configs/`. Most workflows also have a `_quick.yaml` file for a short, fixed-seed check. See [Reproduce an analysis](docs/reproduction.md) for full commands and output details.

## Repository structure

- `src/`: Reusable models, tasks, learning rules, metrics, and analysis code.
- `scripts/`: Executable simulation, scan, training, and analysis workflows.
- `configs/`: YAML configuration files for supported workflows.
- `tests/`: Fast scientific and software checks.
- `docs/`: Scientific scope, repository rules, and reproduction instructions.
- `notebooks/`: Demonstrations and exploratory analyses that use shared code.
- `output/`: Generated results. Git ignores generated files in this directory.
- `archive/`: Preserved legacy code that is not part of the current workflow.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest
```

## Documentation

- [Scientific program and workflow status](docs/scientific_program.md)
- [Reproduction guide](docs/reproduction.md)
- [Repository organization](docs/repository.md)
- [Script inventory](docs/script_inventory.md)
- [Configuration guide](configs/README.md)
