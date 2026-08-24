"""Train spatial or non-spatial working memory with FORCE/RLS only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.disorder import gabor_pattern
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz
from src.learning.force import ForceResult, train_force
from src.models.working_memory import (
    NonSpatialWorkingMemoryReservoir,
    SpatialWorkingMemoryReservoir,
)
from src.tasks.working_memory import (
    WORKING_MEMORY_SECTIONS,
    WorkingMemoryConfig,
    make_working_memory_trial,
)


METRIC_FIELDS = ["model_type", "output_mse", "memory_mse", "output_r2", "memory_r2"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/working_memory/force_spatial.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[WorkingMemoryConfig, dict]:
    """Load one FORCE/RLS working-memory configuration."""
    config, document = load_dataclass_sections(
        args.config,
        WorkingMemoryConfig,
        tuple(WORKING_MEMORY_SECTIONS),
    )
    run = document.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("The run section must be a mapping.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    return config, run


def make_patterns(config: WorkingMemoryConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return two trigger patterns and one shared go cue."""
    first = gabor_pattern(
        config.N,
        config.trigger_one_frequency,
        config.trigger_one_angle,
        0.5,
        0.1,
    )
    second = gabor_pattern(
        config.N,
        config.trigger_two_frequency,
        config.trigger_two_angle,
        0.5,
        0.1,
    )
    if config.go_pattern_type == "gabor":
        go_cue = gabor_pattern(
            config.N,
            config.go_frequency,
            config.go_angle,
            0.5,
            0.1,
        )
    else:
        go_cue = np.random.default_rng(config.seed).standard_normal((config.N, config.N))
    return tuple(
        np.asarray(pattern * config.pattern_scale, dtype=np.float32)
        for pattern in (first, second, go_cue)
    )


def make_model(config: WorkingMemoryConfig):
    """Return the selected fixed reservoir model."""
    if config.model_type == "spatial":
        return SpatialWorkingMemoryReservoir(
            N=config.N,
            dt=config.dt,
            tau_e=config.tau_e,
            tau_i=config.tau_i,
            K=config.K,
            coupling=config.coupling,
            drive=config.drive,
            sigma=config.sigma,
            stimulus_gain=config.stimulus_gain,
            feedback_gain=config.feedback_gain,
            feedback_scale=config.feedback_scale,
            use_feedback_output=config.use_feedback_output,
            use_feedback_memory=config.use_feedback_memory,
            init_scale=config.init_scale,
            microsteps=config.microsteps,
            field_clip=config.field_clip,
            seed=config.seed,
            normalize_kernel=config.normalize_kernel,
            nonlinearity=config.nonlinearity,
            device=config.device,
        )
    return NonSpatialWorkingMemoryReservoir(
        unit_count=config.N**2,
        dt=config.dt,
        tau=config.tau_e,
        recurrent_gain=config.recurrent_gain,
        stimulus_gain=config.stimulus_gain,
        feedback_gain=config.feedback_gain,
        feedback_scale=config.feedback_scale,
        init_scale=config.init_scale,
        seed=config.seed,
        nonlinearity=config.nonlinearity,
        device=config.device,
    )


def plot_result(result: ForceResult, config: WorkingMemoryConfig) -> plt.Figure:
    """Plot RLS learning curves, readouts, targets, and example activity."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(result.output_mse_history, label="Output")
    axes[0, 0].plot(result.memory_mse_history, label="Memory")
    axes[0, 0].set(title="RLS learning", xlabel="Training trial", ylabel="MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 1].plot(result.example_output, "--", label="Output")
    axes[0, 1].plot(result.example_output_target, label="Target")
    axes[0, 1].set(title="Decision readout", xlabel="Step", ylabel="Value")
    axes[0, 1].legend()
    axes[1, 0].plot(result.example_memory, "--", label="Memory")
    axes[1, 0].plot(result.example_memory_target, label="Target")
    axes[1, 0].set(title="Memory readout", xlabel="Step", ylabel="Value")
    axes[1, 0].legend()
    activity = result.example_activity
    axes[1, 1].imshow(activity, origin="lower", aspect="auto", cmap="viridis")
    axes[1, 1].set(title="Reservoir activity", xlabel="Step", ylabel="Unit")
    figure.suptitle(f"FORCE/RLS working memory: {config.model_type}")
    return figure


def main() -> None:
    """Train with RLS and save all outputs."""
    args = parse_args()
    config, run = load_config(args)
    patterns = make_patterns(config)
    choice_rng = np.random.default_rng(config.seed)

    def trial_factory(choice: int | None):
        return make_working_memory_trial(
            config.N,
            config.steps,
            config.delay_steps,
            config.cue_steps,
            patterns,
            choice=choice,
            ramp_memory=config.ramp_memory,
            rng=choice_rng,
        )

    model = make_model(config)
    result = train_force(
        model,
        trial_factory,
        config.model_type,
        config.training_trials,
        config.evaluation_trials,
        config.delta,
        config.forgetting_factor,
        config.train_output,
        config.train_memory,
        config.seed,
    )
    output_root = Path(run.get("output_root", "output/working_memory"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_directory = create_run_directory(
        output_root,
        str(run.get("experiment", f"force_{config.model_type}")),
        run.get("run_id"),
    )
    figure = plot_result(result, config)
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, WORKING_MEMORY_SECTIONS))
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", result.__dict__)
    save_csv(
        run_directory / "metrics.csv",
        METRIC_FIELDS,
        [{
            "model_type": config.model_type,
            "output_mse": result.evaluation_output_mse,
            "memory_mse": result.evaluation_memory_mse,
            "output_r2": result.evaluation_output_r2,
            "memory_r2": result.evaluation_memory_r2,
        }],
    )
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved training run to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
