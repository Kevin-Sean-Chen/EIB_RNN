# Scientific program and workflow map

This file links each main scientific question to its required workflow. Update it when a workflow changes status or when the scientific program changes.

Status terms:

- **Ready**: reproducible script, YAML configuration, saved output, and tests exist.
- **Refactor pending**: relevant exploratory code exists, but the workflow is not reproducible yet.
- **Decision pending**: the scientific role or scope is not final.

## 1. Balance and spatial dynamics

Scientific question: How do E/I balance and connection scale `K` control spontaneous spatial and temporal activity?

| Workflow | Script | Intended output | Status |
|---|---|---|---|
| Baseline local simulation | `scripts/baseline/run_local.py` | E/I activity, population rates, dimension, spatial coherence, and activity frames | **Ready** |
| Local-to-network mode transition | `scripts/baseline/scan_local_network_modes.py` | Reconstruction curves, mode advantage, dimension, and example activity | **Ready** |
| Transition across `K` | `scripts/baseline/scan_K_rhoF_modes.py` | Transition strength and mode metrics across `K` | **Ready** |
| Direct spatial statistics | `scripts/baseline/scan_spatial_statistics.py` | Neighbor correlation, correlation length, and low-wave-number power | **Ready** |

These workflows define the baseline for all disorder and task comparisons.

## 2. Disorder and non-local connectivity

Scientific question: How does low-rank non-local connectivity change spatial organization and network modes?

| Workflow | Script | Intended output | Status |
|---|---|---|---|
| One disorder simulation | `scripts/disorder/run_disorder.py` | Activity, fields, disorder patterns, dimension, and coherence | **Ready** |
| Disorder-strength scan | `scripts/disorder/scan_disorder_strength.py` | Dimension, spatial coherence, and latent coherence across `K` and strength | **Ready** |
| Rank-one structure scan | `scripts/disorder/scan_rank_one.py` | Mode alignment, latent dynamics, and phase dependence | **Ready** |
| Connectivity spectrum | `scripts/analyses/run_spectral.py` | Complex spectrum, spectral abscissa, and stability measures | **Ready** |

The rank-one and spectral workflows must connect the connectivity structure to the observed activity transition.

## 3. Input-driven computation

Scientific question: How does the spatial E/I network represent and track structured input?

| Workflow | Script | Intended output | Status |
|---|---|---|---|
| Moving-dot response | `scripts/driven/run_driven_dot.py` | Stimulus, E activity, center-of-mass traces, and tracking metrics | **Ready** |
| Tracking across `K` | `scripts/driven/scan_driven_dot.py` | Peak lag and cross-correlation across `K` | **Ready** |
| Time-shifted dot prediction | Planned extension of the moving-dot workflow | Prediction error across target lead time | **Refactor pending** |
| Same-time rigid-shift reconstruction | `scripts/driven/train_rigid_reconstruction.py` | Rigid-shift target, RLS readout, reconstruction error, and activity | **Ready** |
| Direction discrimination | `scripts/alternative_tasks/relu2D_ds.py` | Secondary direction-decoding experiment | **Decision pending** |
| Two-dot transient response | Preserved in `archive/legacy_driven/relu2D_driven.py` | Response during input and after input removal | **Decision pending** |

The moving-dot workflow will support time-shifted prediction. Rigid-shift reconstruction measures same-time input representation and uses an RLS readout. Direction discrimination is a secondary task.

## 4. Working memory with RLS

Scientific question: Can fixed spatial recurrent dynamics support working memory through learned readouts and feedback?

| Workflow | Source script | Intended replacement | Status |
|---|---|---|---|
| Working-memory trial | `scripts/WM_res.py` and archived `WM_force.py` | `src/tasks/working_memory.py` | **Ready** |
| Spatial reservoir | Archived `WM_force.py` | `src/models/working_memory.py` | **Ready** |
| RLS/FORCE learning | Archived `WM_force.py` | `src/learning/` and `scripts/working_memory/train_force.py` | **Ready** |
| Performance across `K` | `scripts/working_memory/scan_K.py` | Legacy-matched ridge MSE and R2 across `K`; optional RLS comparison | **Ready** |
| Random-RNN comparison | `scripts/WM_rnn.py` | `scripts/working_memory/train_force.py` with the non-spatial configuration | **Ready** |

The supported learning method is recursive least squares through FORCE. The spatial and non-spatial workflows use the same RLS update. They do not use Adam, gradient descent, back-propagation through time, or offline ridge initialization.

## 5. Dynamic and spectral interpretation

Scientific question: Which spatial and temporal modes explain spontaneous, disordered, and driven activity?

| Workflow | Source script | Intended output | Status |
|---|---|---|---|
| Dynamic mode decomposition | `scripts/relu2D_DMD.py` | DMD modes, frequencies, growth rates, prediction error, and dispersion | **Refactor pending** |
| Connectivity spectrum | `scripts/analyses/run_spectral.py` | Eigenvalue spectrum and leading stability measures | **Ready** |
| Activity-mode comparison | Current mode scan scripts | Local, network, and PCA reconstruction curves | **Ready** |

DMD describes activity dynamics. Spectral analysis describes the connectivity operator. Keep these results separate in saved output.

## 6. Asymmetry and waves

Scientific question: Does asymmetric local connectivity create directed propagation or wave-like activity?

| Workflow | Source script | Intended output | Status |
|---|---|---|---|
| One asymmetric simulation | `scripts/relu2D_asym.py` | Activity movie, direction, speed, and temporal statistics | **Refactor pending** |
| Asymmetry scan | `scripts/scan_asym.py` | Wave speed, autocorrelation, and dimension across asymmetry | **Refactor pending** |

This program item is secondary to the baseline, disorder, input-driven, and working-memory workflows.

## 7. Adaptation and development

Scientific question: How do adaptation or plasticity rules change the spatial E/I dynamics?

Only older tanh experiments exist. No current ReLU workflow supports this item.

Status: **Decision pending**. Keep this item outside the core workflow until its model and scientific claim are defined.

## Refactoring priority

1. Baseline local dynamics: complete.
2. Disorder simulation and strength scan: complete.
3. Rank-one and spectral analysis.
4. Driven moving-dot computation: complete; direction discrimination pending.
5. RLS/FORCE working memory.
6. DMD analysis.
7. Asymmetry and waves.
8. Adaptation or development only after a scope decision.

## Required output standard

Each supported workflow must save:

- Resolved YAML configuration.
- Runtime metadata and software versions.
- Numerical arrays in NPZ format.
- Summary metrics in CSV format.
- One summary figure.
- Optional videos under the same run directory.

Each workflow must also have a fixed-seed `N=15` check or a smaller unit test.
