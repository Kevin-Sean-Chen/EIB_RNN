"""Run one baseline local spatial E/I simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz
from src.local import LOCAL_SECTIONS, LocalConfig, LocalResult, simulate_local
from src.metrics import linear_dimension, spatial_coherence


METRIC_FIELDS = [
    "K",
    "excitatory_mean",
    "inhibitory_mean",
    "excitatory_std",
    "linear_dimension",
    "spatial_coherence",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baseline/local.yaml")
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[LocalConfig, dict]:
    """Load one baseline configuration."""
    config, document = load_dataclass_sections(
        args.config, LocalConfig, tuple(LOCAL_SECTIONS)
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


def calculate_metrics(result: LocalResult, config: LocalConfig) -> dict:
    """Return summary metrics for one baseline simulation."""
    middle = result.excitatory.shape[2] // 2
    return {
        "K": config.K,
        "excitatory_mean": result.excitatory.mean(),
        "inhibitory_mean": result.inhibitory.mean(),
        "excitatory_std": result.excitatory.std(),
        "linear_dimension": linear_dimension(result.excitatory),
        "spatial_coherence": spatial_coherence(result.excitatory[:, :, middle]),
    }


def plot_result(result: LocalResult, config: LocalConfig) -> plt.Figure:
    """Plot population means and three excitatory activity frames."""
    figure = plt.figure(figsize=(15, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    mean_axis = figure.add_subplot(grid[0, :])
    mean_axis.plot(result.time, result.excitatory.mean(axis=(0, 1)), label="E mean")
    mean_axis.plot(result.time, result.inhibitory.mean(axis=(0, 1)), label="I mean")
    mean_axis.set(xlabel="Time", ylabel="Mean rate", title="Population activity")
    mean_axis.legend()

    indices = [0, result.excitatory.shape[2] // 2, -1]
    labels = ["Start", "Middle", "End"]
    for axis_index, (frame_index, label) in enumerate(zip(indices, labels)):
        axis = figure.add_subplot(grid[1, axis_index])
        image = axis.imshow(result.excitatory[:, :, frame_index], origin="lower", cmap="viridis")
        axis.set(title=f"E activity: {label}", xticks=[], yticks=[])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Baseline local E/I network, N={config.N}, K={config.K:g}")
    return figure


def main() -> None:
    """Run the baseline simulation and save all outputs."""
    args = parse_args()
    config, run = load_config(args)
    output_root = Path(run.get("output_root", "output/simulations"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_directory = create_run_directory(
        output_root,
        str(run.get("experiment", "local")),
        run.get("run_id"),
    )
    result = simulate_local(config)
    metrics = calculate_metrics(result, config)
    figure = plot_result(result, config)
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, LOCAL_SECTIONS))
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(
        run_directory / "results.npz",
        {
            "excitatory": result.excitatory,
            "inhibitory": result.inhibitory,
            "time": result.time,
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
