"""Scan moving-dot tracking across network strength K."""

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
    DrivenDotScanResult,
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
        default=Path("configs/tasks/driven_dot_tracking.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(
    args: argparse.Namespace,
) -> tuple[DrivenDotConfig, dict, list[float], int, int]:
    """Load one driven-dot tracking configuration."""
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
    K_values = [float(value) for value in task.get("K_values", [])]
    repetitions = int(task.get("repetitions", 0))
    max_lag = int(task.get("max_lag", 20))
    return config, run, K_values, repetitions, max_lag


def plot_result(
    result: DrivenDotScanResult,
    config: DrivenDotConfig,
    max_lag: int,
) -> plt.Figure:
    """Plot traces, correlations, peak lags, and peak heights."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    tracking_axis, correlation_axis, lag_axis, height_axis = axes.ravel()

    for index, K in enumerate(result.K_values):
        tracking_axis.plot(result.response_com[index], label=f"K={K:g}")
        correlation_axis.plot(
            result.lags,
            result.cross_correlation[index],
            label=f"K={K:g}",
        )
    tracking_axis.plot(result.input_com, "--", color="black", label="Input")
    tracking_axis.set(
        xlabel="Recorded step",
        ylabel="Normalized center of mass",
        title="Tracking traces, first repetition",
    )
    tracking_axis.legend()
    correlation_axis.set(
        xlabel="Lag",
        ylabel="Overlap-normalized cross-correlation",
        title="Cross-correlation, first repetition",
        xlim=(-max_lag, max_lag),
    )
    correlation_axis.legend()

    lag_mean = result.peak_lags.mean(axis=1)
    lag_std = result.peak_lags.std(axis=1)
    height_mean = result.peak_heights.mean(axis=1)
    height_std = result.peak_heights.std(axis=1)
    lag_axis.errorbar(result.K_values, lag_mean, yerr=lag_std, fmt="o-", capsize=4)
    lag_axis.set_xscale("log")
    lag_axis.set(xlabel="K", ylabel="Peak lag", title="Tracking lag")
    height_axis.errorbar(
        result.K_values,
        height_mean,
        yerr=height_std,
        fmt="o-",
        capsize=4,
    )
    height_axis.set_xscale("log")
    height_axis.set(
        xlabel="K",
        ylabel="Peak cross-correlation",
        title="Tracking accuracy",
    )
    figure.suptitle(f"Driven moving-dot tracking, N={config.N}")
    return figure


def main() -> None:
    """Run the tracking scan and save all output files."""
    args = parse_args()
    config, run, K_values, repetitions, max_lag = load_config(args)
    output_root = Path(run.get("output_root", "output/tasks"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    experiment = str(run.get("experiment", "driven_dot_tracking"))
    run_directory = create_run_directory(output_root, experiment, run.get("run_id"))
    result = run_tracking_scan(config, K_values, repetitions, max_lag)
    figure = plot_result(result, config, max_lag)

    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, DRIVEN_DOT_SECTIONS))
    resolved["task"] = {
        "K_values": K_values,
        "repetitions": repetitions,
        "max_lag": max_lag,
    }
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
