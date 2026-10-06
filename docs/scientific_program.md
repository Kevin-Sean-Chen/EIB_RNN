# Scientific program and workflow map

This file links each main scientific question to its required workflow. Update it when a workflow changes status or when the scientific program changes.

The paper-level panel plan is in `docs/figure_plan.md`.

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

### Figure 1 target

Figure 1 uses the canonical slice `u_e = 10` and `u_i = 0`, with `K` as the main scan parameter. It combines example activity, activity dimension, spatial correlation, temporal correlation, mean spatial and temporal spectra `P(k)` and `P(f)`, and E/I balance. A pilot scan showed that positive `u_e` mainly rescales activity and does not define a useful second regime axis. Panel A therefore uses only the model schematic. The complete panel-level plan is in `docs/figure_plan.md`.

A pilot with `tau_i / tau_e = [1, 2, 4]` and fixed `tau_e = 0.01` showed that larger `K` increases irregularity and localization within each row. Slower inhibition shifts fluctuation onset to smaller `K`; at ratio 4, fluctuations are present at `K = 0.1`. Keep this comparison in the supplementary information. The main figure uses `tau_i / tau_e = 2` to show a wider dynamic range across `K`.

The Figure 1 workflow produces matched results at `K = [1, 100, 10000]`. It saves separate panel files, averaged `P(k)` and `P(f)`, current-balance diagnostics, and the underlying numerical results.

## 2. Disorder and non-local connectivity

Scientific question: How does low-rank non-local connectivity change spatial organization and network modes?

| Workflow | Script | Intended output | Status |
|---|---|---|---|
| One disorder simulation | `scripts/disorder/run_disorder.py` | Activity, fields, disorder patterns, dimension, and coherence | **Ready** |
| Disorder-strength scan | `scripts/disorder/scan_disorder_strength.py` | Dimension, spatial coherence, and latent coherence across `K` and strength | **Ready** |
| Rank-one structure scan | `scripts/disorder/scan_rank_one.py` | Mode alignment, latent dynamics, and phase dependence | **Ready** |
| Connectivity spectrum | `scripts/analyses/run_spectral.py` | Complex spectrum, spectral abscissa, and stability measures | **Ready** |

The rank-one and spectral workflows must connect the connectivity structure to the observed activity transition.

### Figure 4 target

Figure 4 uses the local-to-network mode workflow to test spatial-pattern robustness to low-rank non-local recurrence. It scans relative perturbation strength `rho_F` and shows separate curves for different `K` values at `u_e = 10` and `u_i = 0`. The zero-perturbation local network, `rho_F = 0`, is the reference. The core figure uses example activity, mode reconstruction, full-network mode advantage, and non-local subspace alignment. It does not repeat the Figure 2 task scans.

The main claim is that local spatial activity structure changes less with `rho_F` at large `K`. This is distinct from a claim that the non-local term has no input or current effect.

Full-network mode advantage and non-local activity alignment are the mechanism measurements. Direct spatial statistics are validation or supplementary results.

An optional consequence is reduced closed-loop FORCE efficacy at large `K`. A scalar readout with a fixed feedback map creates an effective rank-1 recurrent term. Test closed-loop FORCE against a no-feedback readout across `K`, with feedback strength normalized relative to the `sqrt(K)`-scaled recurrent operator. Include this result in Figure 4 only if it is robust; otherwise keep it in the supplementary information.

## 3. Input-driven computation

Scientific question: How does the spatial E/I network represent and track structured input?

| Workflow | Script | Intended output | Status |
|---|---|---|---|
| Moving-dot response | `scripts/driven/run_driven_dot.py` | Stimulus, E activity, center-of-mass traces, and tracking metrics | **Ready** |
| Tracking across `K` | `scripts/driven/scan_driven_dot.py` | Peak lag and cross-correlation across `K` | **Ready** |
| Same-time rigid-shift reconstruction | `scripts/driven/train_rigid_reconstruction.py` | Matched spatial and random E/I RLS reconstruction controls with measured driven stability | **Ready** |
| Direction discrimination | `scripts/alternative_tasks/relu2D_ds.py` | Secondary direction-decoding experiment | **Decision pending** |
| Two-dot transient response | Preserved in `archive/legacy_driven/relu2D_driven.py` | Response during input and after input removal | **Decision pending** |

The moving-dot workflow measures tracking without a trained readout. It compares the stimulus and excitatory-response centers of mass, then measures cross-correlation height and lag across `K`. Rigid-shift reconstruction uses a separate random E/I topology control because that task compares general reservoir representations. Direction discrimination is a secondary task.

### Figure 2 target

Figure 2 compares signal decoding, prediction, and working memory at `u_e = 10` and `u_i = 0`. Each task uses `K = [0.1, 1, 10, 100, 1000, 10000]`. Signal decoding uses same-time rigid-shift reconstruction. Prediction uses moving-dot cross-correlation. Working memory uses only the final cue readout from the cue-readout task. Each task is one panel with a small task schematic, one representative example, and a performance measure across `K`. Disorder robustness remains in Figure 4.

The current task configurations do not use one common `K` list. Align them before the final Figure 2 runs.

### Figure 3 target

Figure 3 gives one mechanism analysis for each Figure 2 task. It measures stimulus-locked signal-to-noise ratio for same-time decoding, empirical recurrent contribution for moving-dot prediction, and empirical metastability for the final cue readout. Each mechanism measure must use the same `K` scan as Figure 2 and must link directly to its matching performance curve.

The decoding signal is the temporal variance of the trial-mean response to a repeated stimulus. The decoding noise is the time-mean trial variance around that mean. The prediction analysis compares intact spatial recurrence with a spatially shuffled control that preserves E/I strength. It also measures recurrent field power relative to input field power.

The metastability analysis is unsupervised. It uses delay-period E/I activity without cue labels to infer recurrent states and estimate their dwell times, self-transition probabilities, and transition timescales. PCA, DMD, or time-lagged modes can supply reduced coordinates, but the scientific object is the inferred state and its empirical persistence.

## 4. Working memory with RLS

Scientific question: Can fixed spatial recurrent dynamics support working memory through learned readouts and feedback?

| Workflow | Source script | Intended replacement | Status |
|---|---|---|---|
| Working-memory trial | Archived `WM_res.py` and `WM_force.py` | `src/tasks/working_memory.py` | **Ready** |
| Spatial reservoir | Archived `WM_force.py` | `src/models/working_memory.py` | **Ready** |
| RLS/FORCE learning | Archived `WM_force.py` | `src/learning/` and `scripts/working_memory/train_force.py` | **Ready** |
| Performance across `K` | `scripts/working_memory/scan_K.py` | Legacy-matched ridge MSE and R2 across `K`; optional RLS comparison | **Ready** |
| Random-RNN comparison | Archived `WM_rnn.py` | `scripts/working_memory/train_force.py` with the non-spatial configuration | **Ready** |

The supported learning method is recursive least squares. The canonical spatial and non-spatial workflows train readouts without output or memory feedback. They use the same RLS update and do not use Adam, gradient descent, back-propagation through time, or offline ridge initialization.

## 5. Dynamic and spectral interpretation

Scientific question: Which spatial and temporal modes explain spontaneous, disordered, and driven activity?

| Workflow | Source script | Intended output | Status |
|---|---|---|---|
| Dynamic mode decomposition | `scripts/analyses/run_dmd.py` | DMD modes, frequencies, growth rates, prediction error, and dispersion | **Excitatory-rate analysis ready; field analysis pending** |
| Connectivity spectrum | `scripts/analyses/run_spectral.py` | Eigenvalue spectrum and leading stability measures | **Ready** |
| Activity-mode comparison | Current mode scan scripts | Local, network, and PCA reconstruction curves | **Ready** |
| Spontaneous Lyapunov spectrum | Legacy prototype in `old_tanh_code/Lyapunov/LE_test.py` | Leading spectrum at `K = 1`, `100`, and `10000`; convergence and Kaplan--Yorke dimension when supported | **Refactor pending** |

DMD describes activity dynamics. Spectral analysis describes the connectivity operator. Keep these results separate in saved output.

The Figure 1 supplement will compare spontaneous Lyapunov spectra at small, medium, and large `K`. The current driven finite-time Lyapunov measure is not a replacement for this spectrum.

## 6. Asymmetry and waves

Scientific question: Does asymmetric local connectivity create directed propagation or wave-like activity?

| Workflow | Source script | Intended output | Status |
|---|---|---|---|
| One asymmetric simulation | `scripts/relu2D_asym.py` | Activity movie, direction, speed, and temporal statistics | **Refactor pending** |
| Asymmetry scan | `scripts/scan_asym.py` | Wave speed, autocorrelation, and dimension across asymmetry | **Refactor pending** |

This program item is secondary to the baseline, disorder, input-driven, and working-memory workflows.

### Figure 5 target

Figure 5 scans `K = [1, 10, 100, 1000]` and excitatory-kernel shift `b`. It maps stationary, coherent-wave, and irregular regimes, then shows representative activity. A noise-robustness panel adds spatially and temporally independent Gaussian noise to the excitatory drive, with amplitude relative to `u_e`, and tests the prediction that noise disrupts waves at small `K` but not at large `K`. The wave classifier must combine speed with directional consistency or another coherence measure. Full speed, coherence, and dimension curves belong in the supplementary information.

## 7. Adaptation and development

Scientific question: How do adaptation or plasticity rules change the spatial E/I dynamics?

Only older tanh experiments exist. No current ReLU workflow supports this item.

Status: **Decision pending**. Keep this item outside the core workflow until its model and scientific claim are defined.

## Refactoring priority

1. Figure 1 unified `K` scan, `K` by `u_e` pilot, and activity spectra.
2. Figure 2 task scans on one common `K` grid.
3. Figure 3 task-specific mechanism analyses.
4. Figure 4 assembly from the `rho_F` mode workflow; optional closed-loop FORCE test.
5. Figure 5 reproducible asymmetry and noise workflow.
6. Figure 1 Lyapunov-spectrum supplement.
7. Other supplementary validation and controls.
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
