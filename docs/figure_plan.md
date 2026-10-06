# Figure plan

This file defines the current five-figure paper structure. Update it when a panel, claim, or required workflow changes.

Status terms:

- **Agreed**: the scientific content is set.
- **Explore**: run a small scan before the panel enters the main figure.
- **Open**: the panel content is not set.
- **Test first**: the hypothesis is set, but a pilot result will decide whether it enters the main figure or supplementary information.

## Figure 1. Spontaneous dynamics across connection scale

Main question: How does the connection scale `K` organize spontaneous spatial and temporal activity in the local E/I network?

Canonical parameter slice:

- Excitatory drive: `u_e = 10`.
- Inhibitory drive: `u_i = 0`.
- Time constants: `tau_e = 0.01` and `tau_i = 0.02`.
- Main values: `K = [1, 100, 10000]`.

Planned panels:

| Panel | Content | Purpose | Status |
|---|---|---|---|
| A | Two-sheet local E/I model schematic and normalized 2D E-rate patterns at `K = [1, 100, 10000]` | Define the model and introduce the change in spatial patterns | **Agreed** |
| B | Example E/I and local activity traces at `K = [1, 100, 10000]` | Show the change in spontaneous dynamics at the canonical time constants | **Agreed** |
| C | Normalized PCA variance versus PC rank for `K = [1, 100, 10000]`, with participation-ratio annotations | Show the shape of the dimensional spectrum and estimate the effective dimension | **Agreed** |
| D | Spatial correlation across `K` | Measure spatial organization and correlation length | **Agreed** |
| E | Temporal correlation across `K` | Measure temporal persistence | **Agreed** |
| F | Mean spatial and temporal activity spectra, `P(k)` and `P(f)`, across `K` | Resolve the dominant spatial and temporal scales | **Agreed** |
| G | Balance of activity components across `K` | Show the E/I contributions and their cancellation | **Agreed** |

The pilot `K` by `u_e` scan showed no separate positive-`u_e` regimes. For positive `u_e`, activity amplitude scaled with `u_e`, while normalized pattern structure depended mainly on `K`. Keep `u_e = 10` and `u_i = 0` in the main figures. Do not use a `K` by `u_e` map in panel A.

A pilot compared `tau_i / tau_e = [1, 2, 4]` at fixed `tau_e = 0.01`. Larger `K` increased irregularity and localization along every row. However, slower inhibition moved fluctuation onset to much smaller `K`; at ratio 4, fluctuations were present at `K = 0.1`. Keep this comparison in the supplementary information. The main figure uses ratio 2 because it gives a wider dynamic range across `K` while retaining a quiet low-`K` condition. State clearly that `K` organizes patterns within a fixed time-constant ratio, while the ratio controls the instability boundary.

Spectrum rule: average spectra across time windows and simulation seeds. State the normalization and the averaging order. Plot `P(k)` as a radial spatial spectrum unless anisotropy is a result. Plot `P(f)` from the population or site-averaged activity, with the signal choice stated in the caption.

Panel G shows external, recurrent excitatory, recurrent inhibitory, and net currents into excitatory cells. It also separates current cancellation at active and inactive sites. This separation prevents sparse, inhibition-dominated sites from being interpreted as poor balance at active sites.

Save each panel as PDF and PNG under `output/figures/figure1/`. Keep numerical arrays, metrics, the resolved configuration, and metadata under `output/figures/figure1/data/`. Assemble the full figure only after the separate panels are approved.

Current workflow support:

- `scripts/baseline/run_local.py`: one simulation, example activity, E/I rates, dimension, and spatial coherence.
- `scripts/baseline/scan_spatial_statistics.py`: neighbor correlation, correlation length, and low-wave-number power.
- `scripts/figures/figure1_spontaneous.py`: matched simulations, panels B--G, metrics, averaged spectra, and reproducibility files.

## Figure 2. Functional regimes across connection scale

Main question: How does the connection scale `K` control three forms of computation in the same spatial E/I network?

Shared conditions:

- Excitatory drive: `u_e = 10`.
- Inhibitory drive: `u_i = 0`.
- Scan values: `K = [0.1, 1, 10, 100, 1000, 10000]`.
- Tasks: signal decoding, prediction, and working memory.

Use the same `K` values and the same network parameters when the task design permits this.

Planned panels:

| Panel | Task | Content | Source workflow | Status |
|---|---|---|---|---|
| A | Signal decoding | Small rigid-shift task schematic, example target and readout, and same-time reconstruction performance across `K` | `scripts/driven/train_rigid_reconstruction.py` | **Agreed** |
| B | Prediction | Small moving-dot task schematic, example stimulus and response traces, and cross-correlation performance across `K` | `scripts/driven/scan_driven_dot.py` | **Agreed** |
| C | Working memory | Small cue-readout task schematic, example cue, delay, target, and final cue readout, and final cue-readout performance across `K` | `scripts/working_memory/scan_K.py` | **Agreed** |

Use the same layout in all three panels: task schematic, one representative example, and one summary curve across `K`. Keep each schematic small and show only the information needed to define the task.

For panel C, show only the final cue readout. Do not add a separate memory-state readout to the main panel.

The legacy driven scan uses the complete six-point `K` range. The current refactored tracking and working-memory configurations use shorter subsets and must be aligned before final runs.

## Figure 3. Computational mechanism and controls

Main question: Which task-specific property of the network explains each performance curve in Figure 2?

Use three direct mechanism analyses rather than one common mechanism variable:

| Panel | Figure 2 result | Mechanism analysis | Working hypothesis | Status |
|---|---|---|---|---|
| A | Same-time decoding | Stimulus-locked signal-to-noise ratio across `K` | Small `K` preserves an input-driven, feedforward-like representation with high signal-to-noise ratio | **Agreed** |
| B | Moving-dot prediction | Empirical recurrent contribution across `K` | Intermediate `K` supplies useful spatial recurrent information; large `K` adds excessive fluctuation | **Agreed** |
| C | Final cue readout | Unsupervised metastable-state analysis across `K` | Intermediate `K` creates long-lived recurrent activity states that can support memory | **Agreed** |

Each panel must link its mechanism measure to the matching task performance from Figure 2. Use causal controls when possible. Do not require one mechanism variable to explain all three tasks.

For panel A, repeat an identical stimulus across trials or initial states. Define signal as the temporal variance of the trial-mean activity. Define noise as the time-mean trial variance around that mean. Their ratio is the decoding signal-to-noise ratio.

For panel B, compare intact spatial recurrence with a spatially shuffled recurrent control that preserves E/I strength. Use the change in moving-dot cross-correlation as the prediction-relevant recurrent contribution. Also report the recurrent-to-input field-power ratio across `K` as a direct activity measure.

For panel C, infer metastable states without cue labels. Collect delay-period E/I activity across trials, reduce the state space with a slow-mode method, cluster recurrent states, and estimate transitions between them. Measure empirical dwell-time distributions, self-transition probability, and transition timescales across `K`. Use cue labels only for the separate memory-performance analysis, not for state discovery or metastability measurement.

Call the inferred objects **metastable states**, not stable modes. PCA, DMD, or time-lagged modes can define a reduced coordinate system, but a mode alone is not a persistent state.

## Figure 4. Robustness to non-local disorder

Main question: How does large `K` protect local spatial activity patterns from a low-rank non-local recurrent perturbation?

Fixed scope:

- Use the low-rank non-local term from the local-to-network mode workflow.
- Scan its relative strength `rho_F` and show separate curves for different `K` values.
- Keep `u_e = 10` and `u_i = 0`.
- Compare each condition with `rho_F = 0`.
- Focus the core panels on spatial-pattern robustness. Do not repeat the Figure 2 task scans.

Planned panels:

| Panel | Content | Purpose | Status |
|---|---|---|---|
| A | Local-plus-low-rank connectivity schematic and example activity across `rho_F` at low and high `K` | Define the perturbation and show the robustness effect | **Agreed** |
| B | Local-mode and full-network-mode reconstruction curves | Show how much structure each mode basis captures | **Agreed** |
| C | Full-network mode advantage versus `rho_F`, with one curve per `K` | Show that full-network modes add less explanatory power at large `K` | **Agreed** |
| D | Activity alignment with the non-local subspace versus `rho_F`, with one curve per `K` | Show that non-local modes recruit less activity at large `K` | **Agreed** |
| E or SI | Closed-loop FORCE efficacy across `K` | Test whether a learned low-rank feedback update loses control at large `K` | **Test first** |

The main claim is not that large `K` removes the non-local input. The claim is that local spatial activity structure is less changed by that input at large `K`.

Panels C and D give the mechanism for this robustness. Keep neighbor correlation, correlation length, and low-wave-number power as validation analyses or supplementary results.

A scalar FORCE readout with a fixed feedback map adds an effective rank-1 recurrent term. This gives a testable consequence of panels C and D: closed-loop FORCE feedback can become less effective at large `K`. Compare closed-loop FORCE with the same readout trained without feedback. Match the feedback strength relative to the `sqrt(K)`-scaled recurrent operator so that a large-`K` failure is not a trivial gain mismatch. Keep this result in panel E only if it is clear and robust; otherwise place it in the supplementary information.

## Figure 5. Asymmetric coupling and directed waves

Main question: How do connection scale `K` and asymmetric local coupling `b` control directed wave dynamics?

The asymmetry parameter `b` shifts the excitatory Gaussian kernel. The symmetric network has `b = 0`; positive `b` creates a preferred propagation direction.

Use `K = [1, 10, 100, 1000]` for the first scan. Do not extend the main figure above `K = 1000`.

Planned structure:

| Panel | Content | Purpose | Status |
|---|---|---|---|
| A | Symmetric and shifted excitatory-kernel schematic | Define `b` and the expected direction | **Agreed** |
| B | Quantitative `K` by `b` regime map | Locate stationary, coherent-wave, and irregular regimes | **Agreed; classification metric open** |
| C | Example activity or space-time plots from selected regions | Show the regimes represented in the map | **Agreed** |
| D | Wave robustness to dynamic noise at small and large `K` | Show that noise disrupts waves at small `K` but preserves them at large `K` | **Agreed** |

The current legacy scan uses phase-correlation speed, autocorrelation, and linear dimension. A wave classification must require both nonzero speed and consistent displacement. Do not classify noisy frame shifts as waves. Use speed and directional consistency to construct panel B; place their full curves in the supplementary information. Place temporal coherence and activity dimension in the supplementary information unless they are required to validate the regime labels.

For panel D, add spatially and temporally independent Gaussian noise to the excitatory external drive. Define noise amplitude relative to `u_e` and use the same relative scale across `K`. Compare small and large `K` values that both support coherent waves without noise.

## Supplementary analyses

### Figure 1 supplement: Lyapunov spectra

Compute spontaneous Lyapunov spectra at representative connection scales:

- Small: `K = 1`.
- Medium: `K = 100`.
- Large: `K = 10000`.

Use the same local ReLU E/I model and parameters as Figure 1. Average over initial conditions or seeds and show convergence with simulation time. Report the leading exponent, the positive part of the spectrum, and the Kaplan--Yorke dimension when the computed spectrum is long enough.

The model state has size `2 * N^2`, so a complete spectrum at the main network size can be expensive. Start with a matrix-free tangent-vector method and a partial leading spectrum. If a full spectrum is needed, first validate it at smaller `N`. Label a partial spectrum clearly and do not call it complete.

### Figure 5 supplement: wave measurements

Show the full wave-speed, directional-consistency, temporal-coherence, and activity-dimension measurements across `K` and `b`. These analyses support the main-text regime map and noise-robustness panel.
