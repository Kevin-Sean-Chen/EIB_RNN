# Scientific script inventory

This file records the scientific purpose and intended output of each script. Update this file before a script moves to `archive/`.

The paper-level panel plan is in `docs/figure_plan.md`.

Status terms:

- **Current**: supported workflow with configuration and saved output.
- **Migration pending**: scientifically relevant, but still an exploratory script.
- **Archived**: preserved for history. Do not use it as the standard workflow.
- **Intent inferred**: purpose is inferred from code and comments. Confirm it before migration.

## Current workflows

| Script | Purpose | Intended output |
|---|---|---|
| `scripts/baseline/run_local.py` | Run the canonical baseline local spatial E/I network. | Resolved YAML, metadata, NPZ activity and time arrays, CSV metrics, and a summary figure under `output/simulations/local/`. |
| `scripts/disorder/run_disorder.py` | Run one spatial E/I simulation with configurable rank-one or rank-two disorder. | Resolved YAML, metadata, NPZ activity and disorder patterns, CSV metrics, and a summary figure under `output/simulations/disorder/`. |
| `scripts/disorder/scan_disorder_strength.py` | Scan rank-two random disorder across `K` and disorder strength. | Dimension, spatial coherence, latent coherence, example activity, patterns, configuration, metadata, and a summary figure under `output/scans/disorder_strength/`. |
| `scripts/disorder/scan_rank_one.py` | Scan rank-one Gabor disorder across `K` and phase. | Latent coherence, ACF peaks, balance, eigenvector alignment, spectral abscissa, latent traces, configuration, metadata, and a summary figure under `output/scans/rank_one/`. |
| `scripts/driven/run_driven_dot.py` | Run one moving-dot simulation in the driven spatial E/I network. | Resolved YAML, metadata, NPZ activity and tracking arrays, CSV metrics, and a summary figure under `output/simulations/driven_dot/`. |
| `scripts/driven/scan_driven_dot.py` | Measure moving-dot center-of-mass tracking across `K`. | Peak lag, peak cross-correlation, first-repetition traces, activity arrays, configuration, metadata, and a summary figure under `output/tasks/driven_dot_tracking/`. |
| `scripts/baseline/scan_local_network_modes.py` | Compare local geometric modes with full-network modes across non-local strength. | Reconstruction curves, transition index, PCA dimension, example activity, spatial metrics, configuration, metadata, and a summary figure under `output/scans/local_network_modes/`. |
| `scripts/baseline/scan_K_rhoF_modes.py` | Compare the local-to-network mode transition across `K` and `rho_F`. | Per-`K` mode metrics, crossover strength, dimensions, configuration, metadata, and a summary figure under `output/scans/K_rhoF_modes/`. |
| `scripts/baseline/scan_spatial_statistics.py` | Measure direct spatial statistics across non-local strength. | Neighbor correlation, correlation length, low-wave-number power, mode advantage, configuration, metadata, and a summary figure under `output/scans/spatial_statistics/`. |
| `scripts/analyses/run_spectral.py` | Analyze the legacy two-population connectivity spectrum across `K` and Gabor phase. | Complex eigenvalues, spectral abscissa, stability flags, patterns, configuration, metadata, and a summary figure under `output/analyses/spectral/`. |
| `scripts/analyses/run_dmd.py` | Apply dynamic mode decomposition to excitatory activity from the local spatial network. | DMD eigenvalues, spatial modes, rank error, dispersion, growth rates, configuration, metadata, and a summary figure under `output/analyses/dmd/`. |
| `scripts/working_memory/train_force.py` | Train fixed spatial or non-spatial reservoirs with online RLS/FORCE updates. | Resolved YAML, metadata, NPZ training and evaluation arrays, CSV metrics, and a summary figure under `output/working_memory/`. |
| `scripts/working_memory/scan_K.py` | Reproduce the ridge-trained spatial working-memory scan across `K`; allow an RLS comparison. | Post-go output and memory MSE and R2, configuration, metadata, arrays, and a summary figure under `output/scans/working_memory_K/`. |
| `scripts/driven/train_rigid_reconstruction.py` | Reconstruct the same-time rigid-shift angle with spatial E/I, random E/I, or one-population reservoirs and RLS. A separate YAML preserves the Adam readout comparison. | Training error, target and prediction, activity, driven Lyapunov exponent, stimulus, metrics, configuration, metadata, and a summary figure under `output/tasks/`. |
| `scripts/driven/scan_rigid_smoothing.py` | Compare spatial smoothing and fixed pixel permutation effects across spatial and one-population controls. | Reconstruction metrics, stimulus dimension, example frames, configuration, metadata, arrays, and a summary figure under `output/tasks/`. |
| `scripts/figures/figure1_spontaneous.py` | Create matched spontaneous-activity panels A--G at `K = [1, 100, 10000]`. | Separate PDF and PNG panels under `output/figures/figure1/`, with activity, metrics, spectra, configuration, and metadata under `data/`. |
| `scripts/figures/assemble_figure1.py` | Assemble approved Figure 1 panel PNG files without rerunning simulations. | `figure1.png` and `figure1.pdf` under `output/figures/figure1/`. |
| `scripts/figures/figure4_disorder.py` | Create the approved six-panel Figure 4 draft from the saved `tau_i = 0.02` `K_rhoF` scan: schematic, snapshots, mode advantage, nonlocal alignment, active fraction, and low-rank susceptibility. | Separate panel PDF and PNG files plus `figure4.pdf` and `figure4.png` under `output/figures/figure4/`. |

## Figure 1 workflow coverage

| Planned content | Current source | Coverage |
|---|---|---|
| Model schematic and normalized 2D patterns | `scripts/figures/figure1_spontaneous.py` | Two-sheet E/I schematic and matched patterns at the three main `K` values |
| Pilot `K` by `u_e` map, with `u_i = 0` | Throwaway pilot under `output/pilots/figure1_K_ue/` | Completed; positive `u_e` rescales activity, so omit the map from the main figure |
| Example activity and E/I traces | `scripts/figures/figure1_spontaneous.py` | Matched traces at all three main `K` values |
| Activity dimension across `K` | `scripts/figures/figure1_spontaneous.py` | Normalized PCA variance curves with participation-ratio annotations and saved seed values |
| Spatial correlation across `K` | `scripts/figures/figure1_spontaneous.py` | Radial spatial autocorrelation and correlation length |
| Temporal correlation across `K` | `scripts/figures/figure1_spontaneous.py` | Mean sitewise autocorrelation and correlation time |
| Mean `P(k)` and `P(f)` across `K` | `scripts/figures/figure1_spontaneous.py` | Normalized radial and site-averaged spectra |
| E/I activity balance across `K` | `scripts/figures/figure1_spontaneous.py` | Mean current components and active or inactive site cancellation |
| Inhibitory time-constant ratios | Throwaway pilot under `output/pilots/figure1_K_ue/` | Ratios `tau_i / tau_e = [1, 2, 4]` show that slower inhibition shifts fluctuation onset to smaller `K`; use ratio 2 in the main figure and the full comparison in SI |

## Figure 2 workflow coverage

Shared target: `u_e = 10`, `u_i = 0`, and `K = [0.1, 1, 10, 100, 1000, 10000]`.

| Planned task | Current source | Coverage |
|---|---|---|
| Same-time rigid-shift signal decoding | `scripts/figures/figure2_tasks.py` | Six-point scan, best and worst traces, and panel files exist |
| Moving-dot prediction by cross-correlation | `scripts/figures/figure2_tasks.py` | Six-point peak-delay scan, faded peak amplitude, matched lag-window examples, and panel files exist |
| Figure 2 time-constant comparison | `configs/figures/figure2_tasks_tau_i_002.yaml` | Separate `tau_i = 0.02` task panels and metrics under `output/figures/figure2_tau_i_002/`; main output remains unchanged |
| Calibrated Figure 2 memory candidate | `configs/figures/figure2_tasks_tau_i_002_memory_calibrated.yaml` | A 50 ms delay, stronger initial trigger, post-cue response score, full 200 ms trajectories, training `R2`, held-out `R2`, and separate panels under `output/figures/figure2_tau_i_002_memory_calibrated/` |
| Final cue-readout working memory | `scripts/figures/figure2_tasks.py` | Six-point post-cue output-`R2` scan, best and worst task traces, and panel files exist |
| Figure 2 assembly | `scripts/figures/assemble_figure2.py` | Three task rows; each row has the `K` scan on the left and best and worst examples on the right |

The Figure 2 configuration applies the complete six-point `K` range without changing the shorter task-specific source configurations.

## Figure 3 workflow coverage

| Mechanism analysis | Current source | Coverage |
|---|---|---|
| Decoding signal-to-noise ratio | Rigid-reconstruction activity and readout arrays | Add repeated identical stimuli across trials or initial states |
| Recurrent contribution to prediction | Driven activity and input-response cross-correlation | Add a spatially shuffled recurrence control and recurrent-to-input field power |
| Empirical metastability | Working-memory delay activity without cue labels | Add reduced-state clustering, transition estimation, dwell times, and timescales |
| Figure 3 placeholder | `scripts/figures/figure3_placeholder.py` | Three labeled panels and one horizontal assembly; no artificial data |

## Figure 4 workflow coverage

| Planned content | Current source | Coverage |
|---|---|---|
| Low-rank non-local scan across `K` and `rho_F` | `scripts/baseline/scan_K_rhoF_modes.py` | Core scan and saved outputs exist |
| Local and full-network mode reconstruction | `scripts/baseline/scan_K_rhoF_modes.py` | Analysis exists |
| Full-network mode advantage | `scripts/baseline/scan_K_rhoF_modes.py` | Analysis exists |
| Non-local subspace alignment | `scripts/baseline/scan_K_rhoF_modes.py` | Analysis exists |
| Direct spatial structure validation | `scripts/baseline/scan_K_rhoF_modes.py` | Neighbor correlation, correlation length, and low-wave-number fraction exist; keep outside the main four panels |
| Example activity across `rho_F` and `K` | Saved `example_rates` arrays | Figure assembly needed |
| Optional closed-loop FORCE consequence | `archive/legacy_working_memory/WM_force.py` | Legacy feedback stage exists; supported configurations disable feedback, and a normalized feedback scan across `K` is needed |

## Migration-pending ReLU dynamics and analysis

| Script | Intended scientific role | Present output | Planned destination |
|---|---|---|---|
| `relu2D_main.py` | Baseline local two-population ReLU E/I simulation. | Interactive activity figures and animation. | Replaced by `src/local.py` and `scripts/baseline/run_local.py`. Keep it until the legacy comparison and video decision are complete. |
| `scripts/relu2D_disorder.py` | Add low-rank non-local disorder and measure spatial and temporal organization. | Interactive activity, coherence, dimension, autocorrelation, and optional GIF output. | Partly replaced by `src/disorder.py`, `src/metrics.py`, and `scripts/disorder/run_disorder.py`. Keep it until scan and video checks are complete. |
| `scripts/relu2D_dense.py` | Validate a dense-matrix implementation of local and non-local connectivity. | Interactive activity, coherence metrics, and optional GIF output. | Dense reference model or validation tool under `src/`; a small comparison script. |
| `scripts/relu2D_asym.py` | Simulate driven dynamics with an asymmetric excitatory kernel. | Activity arrays and optional GIF output. | Used by the reproducible Figure 5 workflow. Keep it until the simulator moves into `src/`. |
| `scripts/relu2D_DMD.py` | Apply dynamic mode decomposition to spontaneous or driven activity. | DMD eigenvalues, spatial modes, prediction error, dispersion, growth, and dimension figures. | The excitatory-rate workflow is ready in `src/analysis/dmd.py` and `scripts/analyses/run_dmd.py`. Keep this script until the legacy `mue_all` field analysis is reproduced or rejected. |
| `scripts/scan_disorder.py` | Scan `K` and low-rank disorder strength or frequency. | Interactive heatmaps and dimension or coherence summaries. | Strength scan replaced by `scripts/disorder/scan_disorder_strength.py`. Preserve the commented frequency experiment before archive. |
| `scripts/scan_rankone.py` | Scan rank-one strength and phase; inspect coherence and mode alignment. | Interactive heatmaps, spectra, and alignment plots. | Replaced by `scripts/disorder/scan_rank_one.py`. Keep it until the legacy-modulation figure receives visual confirmation. |
| `scripts/scan_asym.py` | Scan asymmetric coupling and estimate wave speed with phase correlation. | Interactive speed, autocorrelation, and activity plots. | YAML asymmetry scan under `scripts/scans/`. |

### Figure 5 coverage

| Planned content | Current source | Coverage |
|---|---|---|
| `K` by `b` activity grid | `scripts/scan_asym.py` | Legacy snapshot grid exists |
| Wave speed | `scripts/scan_asym.py` | Phase-correlation estimate exists |
| Directional consistency | `src/analysis/asymmetric_waves.py` | Periodic phase-correlation displacement consistency |
| Temporal coherence | `scripts/scan_asym.py` | Sampled autocorrelation exists |
| Activity dimension | `scripts/scan_asym.py` | One-dimensional `b` scan exists; extend to `K` by `b` |
| Noise robustness across `K` | `scripts/figures/figure5_asymmetry.py` | Excitatory-only Gaussian noise at `b = 8`, with five matched seeds |
| Reproducible Figure 5 workflow | `scripts/figures/figure5_asymmetry.py` | Separate panels, assembled figure, arrays, configuration, and metadata |

## Migration-pending driven and reservoir tasks

| Script | Intended scientific role | Present output | Planned destination |
|---|---|---|---|
| `scripts/alternative_tasks/relu2D_ds.py` | Train a readout for direction discrimination from a driven spatial reservoir. | Interactive readout, target, and activity figures. | Secondary direction-decoding task. Refactor only if it becomes part of the scientific program. |
| `scripts/alternative_tasks/relu2D_force.py` | Test trained feedback in a driven spatial reservoir. **Intent inferred.** | Interactive target, readout, feedback, and activity figures. | Secondary feedback experiment. Confirm its role before migration. |
| `scripts/alternative_tasks/relu2D_res_local.py` | Test local or masked readout from a spatial reservoir. | Interactive prediction and activity figures. | Secondary local-readout reconstruction task. |
| `scripts/alternative_tasks/scan_res_memory.py` | Measure reservoir reconstruction error across delay. | Interactive memory-error and reconstructed-signal plots. | Secondary delayed-reconstruction task. Working memory is the main memory workflow. |

## Migration-pending working-memory tasks

| Script | Intended scientific role | Present output | Planned destination |
|---|---|---|---|
| `scripts/WM_task.py` | Train a random recurrent network with memory feedback on a working-memory task. | Interactive loss, output, memory, and state figures. | Preserve as an early task implementation; extract the trial definition first. |
| `scripts/WM_task2D.py` | Train a spatial E/I network on the working-memory task with gradient methods. | Interactive loss, output, memory, and spatial-state figures. | Spatial working-memory training script. |

## Current non-core network analysis

| Script | Intended scientific role | Present output |
|---|---|---|
| `network_analysis/DS_rnn.py` | Analyze direction-selective recurrent-network eigenmodes and lags. **Intent inferred.** | Interactive eigenvalue, population, direction, and lag plots. |
| `network_analysis/deg_fluc.py` | Relate graph degree fluctuations to dynamics and size scaling. | Interactive degree, fluctuation, and scaling plots. |
| `network_analysis/elastic_test.py` | Explore elastic or smooth random spatial fields. **Intent inferred.** | Interactive field and correlation figures. |
| `network_analysis/fly_connectome.py` | Load a fly connectome, inspect degree statistics, and test coarse graining. | Interactive degree, adjacency, clustering, and coarse-graining plots. |

These scripts are outside the current core model. Decide later whether they belong in `src/analysis/`, a separate project, or `archive/`.

## Planned supplementary analyses

| Analysis | Current source | Needed workflow |
|---|---|---|
| Figure 1 spontaneous Lyapunov spectra at `K = 1`, `100`, and `10000` | `old_tanh_code/Lyapunov/LE_test.py` and the single driven exponent in `src/learning/rigid_reconstruction.py` | Add a supported matrix-free tangent-vector spectrum for the local ReLU E/I model, convergence checks, and seed aggregation |

## Archived driven experiments

| Script | Preserved scientific content | Replacement |
|---|---|---|
| `archive/legacy_driven/relu2D_driven.py` | Sine, one-dot, and two-dot stimulus experiments; driven E/I dynamics; center-of-mass analysis; optional GIF creation. The script replaced its stimulus several times in one run. | `scripts/driven/run_driven_dot.py`, `src/stimuli.py`, and `src/driven.py`. |
| `archive/legacy_driven/scan_driven.py` | Random-readout exploration and moving-dot center-of-mass tracking across `K`. The random-readout metric used a stale sine target. | `scripts/driven/scan_driven_dot.py` and `src/tasks/driven_dot.py`. |
| `archive/legacy_driven/relu2D_reservoir.py` | Spatial and vanilla reservoir prototypes, Adam readout training, alternative target frequencies, and previously tested parameters. | `scripts/driven/train_rigid_reconstruction.py`, `src/models/rigid_reconstruction.py`, and `src/learning/rigid_reconstruction.py`. |

## Archived working-memory experiments

| Script | Preserved scientific content | Replacement |
|---|---|---|
| `archive/legacy_working_memory/WM_force.py` | Spatial E/I working memory with offline ridge initialization, online FORCE/RLS, fixed output and memory feedback maps, alternative kernels, and optional GIF output. | `scripts/working_memory/train_force.py`, `src/models/working_memory.py`, `src/learning/`, and `src/tasks/working_memory.py`. See `docs/working_memory_legacy_audit.md`. |
| `archive/legacy_working_memory/scan_WM_k.py` | Ridge-trained spatial working-memory MSE and R2 across two alternative `K` lists, with a commented network-size scan. | `scripts/working_memory/scan_K.py` reproduces the ridge protocol and also permits an RLS comparison. The alternative lists remain in `docs/working_memory_legacy_audit.md`. |
| `archive/legacy_working_memory/WM_res.py` | Spatial offline ridge training, time-specific fitting windows, feedback options, and PCA. | `scripts/working_memory/train_force.py`, `scripts/working_memory/scan_K.py`, and the legacy audit. |
| `archive/legacy_working_memory/WM_rnn.py` | Random-RNN ridge comparison, balance alternatives, Dale alternatives, and PCA. | `scripts/working_memory/train_force.py` with the non-spatial RLS configuration and the legacy audit. |
| `archive/legacy_working_memory/relu2D_WM.py` | Earlier Adam readout training, sampled activity traces, and repeated decision plots. | Preserved as a secondary gradient-based experiment; the canonical workflow uses RLS. |

## Archived supporting analyses

| Script | Preserved scientific content | Replacement |
|---|---|---|
| `archive/legacy_analyses/scan_spectral.py` | Spectrum of the legacy repeated-row block operator across `K` and Gabor phase. | `scripts/analyses/run_spectral.py` and `src/analysis/spectral.py`. |

## Older tanh and MATLAB archive

These files predate the current ReLU workflow. Their output is mainly interactive figures unless stated otherwise.

| Script | Preserved scientific intent |
|---|---|
| `old_tanh_code/EI_balanced.py` | Balanced random E/I RNN dynamics, stimulus tuning, and population statistics. |
| `old_tanh_code/balance_2D.py` | Compare weak `1/N` connectivity with strong balanced `1/sqrt(N)` connectivity in 2D. |
| `old_tanh_code/balance_relu.py` | Test the balanced scaling argument with rate iteration and ReLU dynamics. |
| `old_tanh_code/chaotic_2d.py` | Explore chaotic, wave, strip, blinking, and drifting regimes in a 2D tanh network. |
| `old_tanh_code/chaos_2D_EI.m` | MATLAB implementation of 2D E/I chaotic dynamics. |
| `old_tanh_code/chaotic_cij.py` | Test diluted or probabilistic connectivity `C_ij`. |
| `old_tanh_code/adaptive_2D.py` | Add adaptation current; study balance, waves, switching, and parameter scans. |
| `old_tanh_code/analytic_test.py` | Test analytic forms across spatial frequencies. |
| `old_tanh_code/finite_scale.py` | Test finite-size and finite-time scaling in chaotic dynamics. |
| `old_tanh_code/finite_beta.py` | Test long-time variance and ergodic scaling of rate and field variables. |
| `old_tanh_code/init_analysis.py` | Simulate different initial conditions for the 2D convolutional network. |
| `old_tanh_code/init_effects.py` | Analyze saved initial-condition simulations, wave number, and phase. |
| `old_tanh_code/scan_analysis.py` | Analyze saved parameter-scan files and summary measurements. |
| `old_tanh_code/two_dim_scan.py` | Run two-dimensional parameter scans with fixed initial conditions. |
| `old_tanh_code/net_analysis.py` | Analyze low-rank, Gaussian, periodic, block E/I, and motif connectivity matrices. |
| `old_tanh_code/stim_resp.py` | Compare repeated stimulus responses in balanced and unbalanced networks. |
| `old_tanh_code/Lyapunov/LE_test.py` | Test Lyapunov exponent methods, Jacobians, and a possible 2D E/I application. |
| `old_tanh_code/training/back_prop.py` | Train a 2D recurrent model with back-propagation and compare learned spatial structure. |
| `old_tanh_code/training/driven_ds.py` | Test direction selectivity under driven spatiotemporal input; includes GIF output. |
| `old_tanh_code/training/driven_spectra.py` | Study driven temporal spectra with structured spatial input. |
| `old_tanh_code/training/driven_spectra2.py` | Compare temporal and spatial spectra with random spatial input patterns. |
| `old_tanh_code/training/driven_spectra3.py` | Study drifting sine input, spatial spectra, and intrinsic scale matching. |
| `old_tanh_code/training/echo_2d.py` | Train a 2D echo-state style readout on oscillating and abrupt targets. |
| `old_tanh_code/training/echo_Lorenz.py` | Train a 2D recurrent network with Lorenz input or targets. |
| `old_tanh_code/training/echo_Lorenz2.py` | Train and test on separate parts of a long Lorenz trajectory. |
| `old_tanh_code/training/echo_balance.py` | Scan balance conditions during echo-state style training. **Intent inferred.** |
| `old_tanh_code/training/echo_chaos.py` | Use chaotic activity as input, then test training and perturbation response. |
| `old_tanh_code/training/echo_compare.py` | Compare readout choices or spatial averaging for echo-state training. **Intent inferred.** |
| `old_tanh_code/training/echo_rnn.py` | Compare a random chaotic RNN, an E/I RNN, and spatial-network readouts. |
| `old_tanh_code/training/feedback_2d.py` | Test trained output feedback in the 2D recurrent network. |
| `old_tanh_code/training/impulse_resp.py` | Measure transient spatial impulse response and visualize rate and field dynamics. |

## Archive procedure

Before any future archive move:

1. Add or update the script entry in this file.
2. Record its unique scientific question.
3. Record its intended figures, metrics, arrays, and videos.
4. Identify the new script and shared source files that replace it.
5. Run a small numerical or visual comparison when possible.
6. Move the original file without rewriting its scientific code.
