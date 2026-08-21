"""Run one moving-dot simulation in a driven spatial E/I network."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.driven import DRIVEN_DOT_SECTIONS, DrivenDotConfig
from src.io import (
    create_run_directory,
    runtime_metadata,
    save_csv,
    save_json,
    save_npz,
)
from src.tasks.driven_dot import (
    run_tracking_scan,
    scan_metric_rows,
    scan_result_arrays,
)


METRIC_FIELDS = [
    "K",
    "peak_lag_mean",
    "peak_lag_std",
    "peak_height_mean",
    "peak_height_std",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulations/driven_dot.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[DrivenDotConfig, dict, int]:
    """Load one driven-dot configuration."""
    config, document = load_dataclass_sections(
        args.config,
        DrivenDotConfig,
        tuple(DRIVEN_DOT_SECTIONS),
    )
    if args.seed is not None:
        config.seed = args.seed
    if args.N is not None:
        config.N = args.N
        config.__post_init__()
    run = document.get("run", {})
    task = document.get("task", {})
    if not isinstance(run, dict) or not isinstance(task, dict):
        raise ValueError("The run and task sections must be mappings.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    max_lag = int(task.get("max_lag", 20))
    return config, run, max_lag


def plot_result(result, config: DrivenDotConfig) -> plt.Figure:
    """Plot stimulus frames, activity frames, and tracking traces."""
    figure = plt.figure(figsize=(16, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    indices = [0, result.stimulus.shape[2] // 2, -1]
    labels = ["Start", "Middle", "End"]
    activity = result.example_activity[0]

    for column, (index, label) in enumerate(zip(indices, labels)):
        stimulus_axis = figure.add_subplot(grid[0, column])
        stimulus_image = stimulus_axis.imshow(
            result.stimulus[:, :, index], origin="lower", cmap="gray"
        )
        stimulus_axis.set_title(f"Stimulus: {label}")
        stimulus_axis.set(xticks=[], yticks=[])
        figure.colorbar(stimulus_image, ax=stimulus_axis, fraction=0.046, pad=0.04)

        activity_axis = figure.add_subplot(grid[1, column])
        activity_image = activity_axis.imshow(
            activity[:, :, index], origin="lower", cmap="viridis"
        )
        activity_axis.set_title(f"E activity: {label}")
        activity_axis.set(xticks=[], yticks=[])
        figure.colorbar(activity_image, ax=activity_axis, fraction=0.046, pad=0.04)

    tracking_axis = figure.add_subplot(grid[:, 3])
    tracking_axis.plot(result.input_com, "--", color="black", label="Input")
    tracking_axis.plot(result.response_com[0], label="E response")
    tracking_axis.set(
        xlabel="Recorded step",
        ylabel="Normalized center of mass",
        title="Moving-dot tracking",
    )
    tracking_axis.legend()
    figure.suptitle(f"Driven moving dot, N={config.N}, K={config.K:g}")
    return figure


def main() -> None:
    """Run the simulation and save all output files."""
    args = parse_args()
    config, run, max_lag = load_config(args)
    output_root = Path(run.get("output_root", "output/simulations"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    experiment = str(run.get("experiment", "driven_dot"))
    run_directory = create_run_directory(output_root, experiment, run.get("run_id"))
    result = run_tracking_scan(config, [config.K], 1, max_lag)
    figure = plot_result(result, config)

    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, DRIVEN_DOT_SECTIONS))
    resolved["task"] = {"max_lag": max_lag}
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", scan_result_arrays(result))
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, scan_metric_rows(result))
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved run to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
