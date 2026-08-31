# Working-memory FORCE legacy audit

This audit records the scientific content of `archive/legacy_working_memory/WM_force.py`.

## Supported match

| Item | Legacy active value | New location |
|---|---:|---|
| Spatial size | `N=23` | `model.N` |
| Time steps | `500` | `task.steps` |
| Delay and cue | `250`, `20` | `task.delay_steps`, `task.cue_steps` |
| Network scale | `K=20` | `network.K` |
| Coupling | `[[1,-4],[2,-2]]` | `network.J_*` |
| Time constants | `0.01`, `0.01` | `network.tau_e`, `network.tau_i` |
| Spatial widths | `0.05`, `0.05 sqrt(2)` | `network.sigma_e`, `network.sigma_i` |
| Baseline drive | `10`, `0` | `network.u_e`, `network.u_i` |
| Input gain | `10` | `task.stimulus_gain` |
| RLS trials | `10` | `learning.training_trials` |
| Evaluation trials | `20` | `learning.evaluation_trials` |
| RLS delta | `0.1` | `learning.delta` |
| Forgetting factor | `1.0` | `learning.forgetting_factor` |
| Feedback gain | `0.001` | `learning.feedback_gain` |
| Feedback-map scale | `0.01` | `learning.feedback_scale` |

The new model uses the same normalized periodic Gaussian kernels. It clamps the recurrent fields before it adds feedback, as in the legacy model. A regression test checks this update order.

The new workflow uses direct RLS from zero readout weights. It does not use the legacy offline ridge stage. This difference is intentional. The main scientific program specifies RLS learning.

The canonical configurations disable output and memory feedback. They set `feedback_gain` to zero. This condition tests memory in the fixed reservoir state without learned feedback effects. The legacy feedback values remain recorded below as optional settings.

The canonical non-spatial control enables the row-centering option that was commented in `scripts/WM_rnn.py`. It also applies the same recurrent-field limit as the spatial model. These settings prevent the unbounded ReLU activity seen with the active legacy random matrix at gain `1.5`.

The new workflow uses a fixed seed and balanced choices. The legacy script used unseeded random choices. These changes improve reproducibility and ensure that short runs include both targets.

## Preserved options

- Kernel normalization on or off: `network.normalize_kernel`.
- ReLU or tanh activation: `model.nonlinearity`.
- Fixed or ramped memory target: `task.ramp_memory`.
- Random or Gabor go cue: `task.go_pattern_type`.
- Output feedback on or off: `learning.use_feedback_output`.
- Memory feedback on or off: `learning.use_feedback_memory`.
- One or more dynamics microsteps: `network.microsteps`.
- Output or memory readout training on or off: `learning.train_output` and `learning.train_memory`.
- RLS forgetting-factor and delta changes.
- Feedback-gain changes. The legacy comment suggests values from `0.05` to `0.5` after a stable small-gain run.

## Legacy-only exploratory options

- Offline ridge pretraining used 30 trials. A comment suggests 200 to 1000 trials.
- The ridge penalty was `0.01`. A comment suggests a sweep from `1e-6` to `1`.
- Ridge memory fitting used the delay and later periods.
- Ridge output fitting used the go-cue and later periods.
- A lower excitatory baseline drive was suggested when learning was weak.
- Optional GIF output saved two choice-specific activity movies.

These ridge and GIF options remain documented. They are not part of the supported direct-RLS workflow.

The archived spatial and random ridge scripts also compute PCA cumulative variance from reservoir activity. The earlier Adam script plots sampled-neuron activity and repeated choice-specific decision traces. These analyses remain available in `archive/legacy_working_memory/`; they are not part of the canonical RLS output.

## Legacy K scan

`archive/legacy_working_memory/scan_WM_k.py` used ridge training with 20 trials, a penalty of `0.01`, and no feedback. Its active `K` list was `[1, 10, 100, 1000]`. An earlier list was `[1, 10, 20, 40, 80, 160]`.

The script also contained a commented size scan with `N=[15, 21, 27, 33, 37, 41]`. The supported `K` scan does not run this size experiment. This option remains recorded for a later size-scaling workflow.

The legacy `K` scan used no recurrent-field clamp because it imported the model from `WM_res.py`. The reproduced scan therefore sets `field_clip: null`. It recovers the intermediate performance optimum at `K=10` for the active coarse list.
