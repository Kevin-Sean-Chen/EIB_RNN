"""Run one spatial E/I simulation with low-rank disorder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.disorder import DISORDER_SECTIONS, DisorderConfig, DisorderResult, simulate_disorder
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz
from src.metrics import latent_coherence, linear_dimension, spatial_coherence


METRIC_FIELDS = ["K", "strength", "rank", "linear_dimension", "spatial_coherence", "latent_coherence_mean"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulations/disorder.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[DisorderConfig, dict]:
    """Load one disorder configuration."""
    config, document = load_dataclass_sections(
        args.config,
        DisorderConfig,
        tuple(DISORDER_SECTIONS),
    )
    if args.seed is not None:
        config.seed = args.seed
    if args.N is not None:
        config.N = args.N
        config.__post_init__()
    run = document.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("The run section must be a mapping.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    return config, run


def calculate_metrics(result: DisorderResult, config: DisorderConfig) -> dict:
    """Return summary metrics for one simulation."""
    activity = result.excitatory.reshape(config.N**2, -1)
    latent_values = [
        latent_coherence(activity, result.left_patterns[:, index])
        for index in range(config.rank)
    ]
    return {
        "K": config.K,
        "strength": config.strength,
        "rank": config.rank,
        "linear_dimension": linear_dimension(result.excitatory),
        "spatial_coherence": spatial_coherence(result.excitatory[:, :, result.excitatory.shape[2] // 2]),
        "latent_coherence_mean": float(np.mean(latent_values)),
    }


def plot_result(result: DisorderResult, config: DisorderConfig) -> plt.Figure:
    """Plot disorder patterns, activity frames, and population means."""
    figure = plt.figure(figsize=(15, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    for index in range(config.rank):
        axis = figure.add_subplot(grid[0, index])
        image = axis.imshow(
            result.left_patterns[:, index].reshape(config.N, config.N),
            origin="lower",
            cmap="bwr",
        )
        axis.set(title=f"Left pattern {index + 1}", xticks=[], yticks=[])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    frame_indices = [0, result.excitatory.shape[2] // 2, -1]
    for column, frame_index in enumerate(frame_indices):
        axis = figure.add_subplot(grid[1, column])
        image = axis.imshow(result.excitatory[:, :, frame_index], origin="lower", cmap="viridis")
        axis.set(title=f"E activity: frame {frame_index}", xticks=[], yticks=[])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    trace_axis = figure.add_subplot(grid[:, 3])
    trace_axis.plot(result.excitatory.mean(axis=(0, 1)), label="E mean")
    trace_axis.plot(result.inhibitory.mean(axis=(0, 1)), label="I mean")
    trace_axis.set(xlabel="Recorded step", ylabel="Mean rate", title="Population activity")
    trace_axis.legend()
    figure.suptitle(
        f"Low-rank disorder, N={config.N}, K={config.K:g}, "
        f"rank={config.rank}, strength={config.strength:g}"
    )
    return figure


def main() -> None:
    """Run the simulation and save its outputs."""
    args = parse_args()
    config, run = load_config(args)
    output_root = Path(run.get("output_root", "output/simulations"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_directory = create_run_directory(
        output_root,
        str(run.get("experiment", "disorder")),
        run.get("run_id"),
    )
    result = simulate_disorder(config)
    metrics = calculate_metrics(result, config)
    figure = plot_result(result, config)

    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, DISORDER_SECTIONS))
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(
        run_directory / "results.npz",
        {
            "excitatory": result.excitatory,
            "inhibitory": result.inhibitory,
            "excitatory_field": result.excitatory_field,
            "inhibitory_field": result.inhibitory_field,
            "left_patterns": result.left_patterns,
            "right_patterns": result.right_patterns,
        },
    )
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, [metrics])
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved run to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
