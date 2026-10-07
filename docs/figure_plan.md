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

A pilot compared `tau_i / tau_e = [1, 2, 4]` at fixed `tau_e = 0.01`. Larger `K` increased irregularity and localization along every row. However, slower inhibition moved fluctuation onset to much smaller `K`; at ratio 4, fluctuations were present at `K = 0.1`. Keep this comparison in the supplementary information. The main figure uses ratio 2 because it gives a clearer dynamic range across `K`. State clearly that `K` organizes patterns within a fixed time-constant ratio, while the ratio controls the instability boundary.

Spectrum rule: average spectra across time windows and simulation seeds. State the normalization and the averaging order. Plot `P(k)` as a radial spatial spectrum unless anisotropy is a result. Plot `P(f)` from the population or site-averaged activity, with the signal choice stated in the caption.

Panel G shows external, recurrent excitatory, recurrent inhibitory, and net currents into excitatory cells. It also separates current cancellation at active and inactive sites. This separation prevents sparse, inhibition-dominated sites from being interpreted as poor balance at active sites.

Current Figure 1 result: increasing `K` changes activity from a quiet low-`K` regime to structured fluctuations and then sparse, strongly fluctuating activity. The time-constant comparison remains an SI robustness analysis.

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
| A | Signal decoding | Performance across `K` above; rigid-body angle and same-time decoded angle below | `scripts/figures/figure2_tasks.py` | **Corrected first pass generated** |
| B | Prediction | Peak delay across `K` above, with faded peak amplitude; lagged cross-correlation examples below | `scripts/figures/figure2_tasks.py` | **Corrected first pass generated** |
| C | Working memory | Post-cue output-readout `R2` across `K` above; stimulus, delay, common cue, target, and readout trace below | `scripts/figures/figure2_tasks.py` | **Corrected first pass generated** |

Use three compact rows in the assembled figure. Use one row for each task. Put the performance-versus-`K` scan on the left. Put the best and worst examples side by side on the right. Title the scan `Decoding`, `Prediction`, or `Working memory`. Use large legends. Mark the example conditions on the scan and state each `K` value. The examples define the tasks, so a separate task cartoon is not required.

Show two examples below each scan. Use the best-performing and worst-performing `K` values from that task scan.

For panel A, plot the true rigid-body angle and the decoded angle over the same time interval. Treat the angle as a circular variable and avoid artificial jumps at the wrap boundary.

For panel B, use the delay to the cross-correlation peak as the primary performance measure. Plot peak amplitude as a faded curve on a second y-axis. Below the scan, plot cross-correlation as a function of lag and mark the selected peak. Use the dot center of mass and the excitatory-activity center of mass.

For panel C, show the stimulus-specific input, delay, common cue, and response periods in one example trace. Plot the target and output readout. The stimulus-specific pattern occurs first. The same common cue occurs for both choices only after the delay.

For panel C, show only the final cue readout. Do not add a separate memory-state readout to the main panel.

The corrected full pass uses the complete six-point `K` range, three seeds for decoding and memory, and five repetitions for prediction. It uses the legacy task time constants `tau_e = tau_i = 0.01` and a 0.5-second zero-input settling period before each measured trial. A separate `tau_i = 0.02` comparison tests alignment with Figure 1 before it replaces the current output. Panel C uses post-cue output-readout `R2`, as in the initial working-memory scan.

The `tau_i = 0.02` comparison is saved under `output/figures/figure2_tau_i_002/`. It does not preserve the Figure 2 result: decoding shifts toward `K = 1`, prediction correlations weaken, and the positive memory optimum at `K = 10` disappears. Keep `tau_i = 0.01` as the current Figure 2 setting. Do not replace the main output with this comparison.

A calibrated `tau_i = 0.02` first draft is saved under `output/figures/figure2_tau_i_002_memory_calibrated/`. The 200 ms memory trial uses a 20 ms stimulus, 50 ms delay, 20 ms common cue, and a 20 ms scored response after the cue. The initial trigger is tenfold stronger, while the common cue is unchanged. Across three seeds, held-out cue-readout `R2` is approximately `0.33`, `0.88`, and `0.72` at `K = 0.1`, `1`, and `10`, then becomes negative at `K >= 100`. Training `R2` remains high at large `K`, which identifies poor trial generalization rather than failed readout fitting. Panel C displays the full 200 ms readout trajectory and highlights the scored post-cue interval. The assembled first draft uses this output for all three task rows.

Current first-pass results: decoding has a broad optimum at `K = 10` to `100` and fails at large `K`. Prediction delay is smallest at `K = 100`; its peak amplitude is largest at `K = 10` and then decreases. The working-memory output readout has its only positive `R2` maximum at `K = 10`.

## Figure 3. Computational mechanism and controls

Main question: Which task-specific property of the network explains each performance curve in Figure 2?

Use three direct mechanism analyses rather than one common mechanism variable:

| Panel | Figure 2 result | Mechanism analysis | Working hypothesis | Status |
|---|---|---|---|---|
| A | Same-time decoding | Stimulus-locked signal-to-noise ratio across `K` | Small `K` preserves an input-driven, feedforward-like representation with high signal-to-noise ratio | **Placeholder complete** |
| B | Moving-dot prediction | Empirical recurrent contribution across `K` | Intermediate `K` supplies useful spatial recurrent information; large `K` adds excessive fluctuation | **Placeholder complete** |
| C | Final cue readout | Unsupervised metastable-state analysis across `K` | Intermediate `K` creates long-lived recurrent activity states that can support memory | **Placeholder complete** |

Each panel must link its mechanism measure to the matching task performance from Figure 2. Use causal controls when possible. Do not require one mechanism variable to explain all three tasks.

For panel A, repeat an identical stimulus across trials or initial states. Define signal as the temporal variance of the trial-mean activity. Define noise as the time-mean trial variance around that mean. Their ratio is the decoding signal-to-noise ratio.

For panel B, compare intact spatial recurrence with a spatially shuffled recurrent control that preserves E/I strength. Use the change in moving-dot cross-correlation as the prediction-relevant recurrent contribution. Also report the recurrent-to-input field-power ratio across `K` as a direct activity measure.

For panel C, infer metastable states without cue labels. Collect delay-period E/I activity across trials, reduce the state space with a slow-mode method, cluster recurrent states, and estimate transitions between them. Measure empirical dwell-time distributions, self-transition probability, and transition timescales across `K`. Use cue labels only for the separate memory-performance analysis, not for state discovery or metastability measurement.

Call the inferred objects **metastable states**, not stable modes. PCA, DMD, or time-lagged modes can define a reduced coordinate system, but a mode alone is not a persistent state.

The placeholder uses the common six-point `K` axis from Figure 2. Each panel pairs one mechanism measure on the left axis with its matching task-performance measure on the right axis. It contains no artificial data. The files are under `output/figures/figure3/`.

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
| A | Gaussian local E/I coupling on two spatial sheets, plus the nonlocal low-rank loop | Define the local and nonlocal connectivity terms | **Draft complete** |
| B | Instantaneous activity across `rho_F` at `K = 100` and `K = 10000` | Show the pattern effect at moderate and large `K` | **Draft complete** |
| C | Full-network mode advantage versus `rho_F`, with one curve per `K` | Show that full-network modes add less explanatory power at large `K` | **Draft complete** |
| D | Activity alignment with the nonlocal subspace versus `rho_F`, with matched and null controls | Show that nonlocal modes recruit less activity at large `K` | **Draft complete** |
| E | Excitatory and inhibitory active-site fractions versus `rho_F` | Show that large `K` closes the active ReLU gate | **Draft complete** |
| F | Low-rank susceptibility `P_M / P_out` versus `rho_F` | Show that activity response decreases relative to low-rank recurrent current | **Draft complete** |
| SI | Closed-loop FORCE efficacy across `K` | Test whether a learned low-rank feedback update loses control at large `K` | **Test later** |

The main claim is not that large `K` removes the non-local input. The claim is that local spatial activity structure is less changed by that input at large `K`.

Panels C and D establish the mode-based effect. Panels E and F summarize the ReLU-gating mechanism. At small `rho_F`, activity remains mainly local. At intermediate `rho_F`, nonlocal recruitment increases. At large `rho_F`, low-rank recurrent current increases while the active fraction and response per unit current decrease. Keep neighbor correlation, correlation length, low-wave-number power, raw power curves, and the full reconstruction curves as validation analyses or supplementary results.

The current draft uses the saved `tau_i = 0.02` scan for all panels, consistent with Figures 1 and 2. Panel B uses one measured-time snapshot, not a temporal mean. The earlier `tau_i = 0.01` scan and figure draft remain preserved for comparison. Do not mix `tau_i` values across Figure 4 panels.

A scalar FORCE readout with a fixed feedback map adds an effective rank-1 recurrent term. This gives a testable consequence of panels C and D: closed-loop FORCE feedback can become less effective at large `K`. Compare closed-loop FORCE with the same readout trained without feedback. Match the feedback strength relative to the `sqrt(K)`-scaled recurrent operator so that a large-`K` failure is not a trivial gain mismatch. Keep this result in panel E only if it is clear and robust; otherwise place it in the supplementary information.

## Figure 5. Asymmetric coupling and directed waves

Main question: How do connection scale `K` and asymmetric local coupling `b` control directed wave dynamics?

The asymmetry parameter `b` shifts the excitatory Gaussian kernel. The symmetric network has `b = 0`; positive `b` creates a preferred propagation direction.

Use `K = [1, 10, 100, 1000]` for the first scan. Do not extend the main figure above `K = 1000`.

Current structure:

| Panel | Content | Purpose | Status |
|---|---|---|---|
| A | Centered inhibitory and shifted excitatory Gaussian kernels above the spatial sheet | Define `b` and the expected direction | **Draft complete** |
| B | Example activity grid for `K = [1, 10, 100, 1000]` and `b = [0, 2, 4, 8]` | Show how spatial patterns change across the scan | **Draft complete** |
| C | Signed wave speed and directional coherence for `b = 0` and `b = 8` across `K` | Quantify the effect of strong bias | **Draft complete** |
| D | Signed wave speed and directional coherence for noise `0` and `0.5` at `b = 8` across `K` | Test whether dynamic noise improves wave propagation | **Draft complete** |

The draft uses periodic phase correlation with a 20 ms lag. Signed speed measures motion along the expected diagonal direction. Directional coherence is the magnitude of the mean velocity divided by the mean speed. A wave requires both nonzero signed speed and consistent displacement. Panel B remains a pattern grid; panels C and D show the quantitative measurements.

For panel D, add spatially and temporally independent Gaussian noise only to the excitatory external drive, inside the common `sqrt(K)` factor. Use noise amplitudes `0` and `0.5` across all four `K` values. Use five matched seeds. Apply noise during initialization and measurement. The first draft shows a regime-specific effect: noise increases signed speed for `K = 1` to `100`, but directional coherence increases clearly only at `K = 100` and decreases at `K = 1000`.

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
