# Scientific script inventory

This file records the scientific purpose and intended output of each script. Update this file before a script moves to `archive/`.

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

## Migration-pending ReLU dynamics and analysis

| Script | Intended scientific role | Present output | Planned destination |
|---|---|---|---|
| `relu2D_main.py` | Baseline local two-population ReLU E/I simulation. | Interactive activity figures and animation. | Replaced by `src/local.py` and `scripts/baseline/run_local.py`. Keep it until the legacy comparison and video decision are complete. |
| `scripts/relu2D_disorder.py` | Add low-rank non-local disorder and measure spatial and temporal organization. | Interactive activity, coherence, dimension, autocorrelation, and optional GIF output. | Partly replaced by `src/disorder.py`, `src/metrics.py`, and `scripts/disorder/run_disorder.py`. Keep it until scan and video checks are complete. |
| `scripts/relu2D_dense.py` | Validate a dense-matrix implementation of local and non-local connectivity. | Interactive activity, coherence metrics, and optional GIF output. | Dense reference model or validation tool under `src/`; a small comparison script. |
| `scripts/relu2D_asym.py` | Simulate driven dynamics with an asymmetric excitatory kernel. | Interactive activity and optional GIF output. | Asymmetric connectivity in `src/`; a wave or direction-selectivity simulation script. |
| `scripts/relu2D_DMD.py` | Apply dynamic mode decomposition to spontaneous or driven activity. | DMD eigenvalues, spatial modes, prediction error, dispersion, growth, and dimension figures. | The excitatory-rate workflow is ready in `src/analysis/dmd.py` and `scripts/analyses/run_dmd.py`. Keep this script until the legacy `mue_all` field analysis is reproduced or rejected. |
| `scripts/scan_disorder.py` | Scan `K` and low-rank disorder strength or frequency. | Interactive heatmaps and dimension or coherence summaries. | Strength scan replaced by `scripts/disorder/scan_disorder_strength.py`. Preserve the commented frequency experiment before archive. |
| `scripts/scan_rankone.py` | Scan rank-one strength and phase; inspect coherence and mode alignment. | Interactive heatmaps, spectra, and alignment plots. | Replaced by `scripts/disorder/scan_rank_one.py`. Keep it until the legacy-modulation figure receives visual confirmation. |
| `scripts/scan_asym.py` | Scan asymmetric coupling and estimate wave speed with phase correlation. | Interactive speed, autocorrelation, and activity plots. | YAML asymmetry scan under `scripts/scans/`. |

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
