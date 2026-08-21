"""Scan local-to-network mode dominance across K and rho_F."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.analysis.local_network_modes import (
    MODE_SCAN_SECTIONS,
    ModeScanConfig,
    ScanResult,
    run_scan,
    scan_metric_rows,
    scan_result_arrays,
)
from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import (
    create_run_directory,
    runtime_metadata,
    save_csv,
    save_json,
    save_npz,
)


METRIC_FIELDS = [
    "K",
    "strength",
    "relative_strength",
    "spectral_strength",
    "network_advantage",
    "pca_dimension",
    "neighbor_correlation",
    "correlation_length",
    "low_k_fraction",
    "stable",
]


def crossover_strength(
    rho_f: np.ndarray,
    advantage: np.ndarray,
    threshold: float,
) -> float:
    """Return the first interpolated rho_F above a threshold."""
    valid = np.isfinite(rho_f) & np.isfinite(advantage)
    x = rho_f[valid]
    y = advantage[valid]
    if x.size == 0 or np.all(y < threshold):
        return np.nan
    index = int(np.flatnonzero(y >= threshold)[0])
    if index == 0:
        return float(x[0])
    x0, x1 = x[index - 1], x[index]
    y0, y1 = y[index - 1], y[index]
    if np.isclose(y1, y0):
        return float(x1)
    fraction = (threshold - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def scan_all_K(
    config: ModeScanConfig,
    K_values: list[float],
) -> dict[float, ScanResult]:
    """Run matched rho_F scans for all requested K values."""
    results = {}
    for K in K_values:
        print(f"\nStart K={K:g}")
        results[float(K)] = run_scan(replace(config, K=K))
    return results


def plot_results(
    results: dict[float, ScanResult],
    config: ModeScanConfig,
    transition_threshold: float,
) -> plt.Figure:
    """Plot network advantage, dimension, and crossover strength across K."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ax_advantage, ax_dimension, ax_crossover = axes
    crossovers = []

    for K, result in results.items():
        color = ax_advantage._get_lines.get_next_color()
        x = result.relative_strengths
        y = result.transition_index
        ax_advantage.plot(x, y, "o-", color=color, label=f"K={K:g}")
        ax_advantage.fill_between(
            x,
            y - result.transition_std,
            y + result.transition_std,
            color=color,
            alpha=0.15,
        )
        ax_dimension.plot(x, result.dimensions, "o-", color=color, label=f"K={K:g}")
        ax_dimension.fill_between(
            x,
            result.dimensions - result.dimension_std,
            result.dimensions + result.dimension_std,
            color=color,
            alpha=0.15,
        )
        crossovers.append(crossover_strength(x, y, transition_threshold))

    ax_advantage.axhline(transition_threshold, color="black", linestyle=":")
    ax_advantage.axhline(0.0, color="black", linewidth=0.8)
    ax_advantage.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Network advantage",
        title="Local-to-network mode transition",
    )
    ax_advantage.legend()
    ax_dimension.set(
        xlabel="Relative total strength, rho_F",
        ylabel="PCA dimension",
        title="Activity dimension",
    )
    ax_dimension.legend()

    K_array = np.asarray(list(results.keys()))
    ax_crossover.plot(K_array, crossovers, "o-")
    ax_crossover.set_xscale("log")
    ax_crossover.set(
        xlabel="K",
        ylabel="Crossover rho_c",
        title=f"First advantage >= {transition_threshold:g}",
    )
    figure.suptitle(
        f"Mode dominance across K and rho_F, N={config.N}, rank={config.rank}"
    )
    return figure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scans/K_rhoF_modes.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(
    args: argparse.Namespace,
) -> tuple[ModeScanConfig, dict, list[float], float]:
    """Load the base scan and K-scan settings."""
    config, document = load_dataclass_sections(
        args.config,
        ModeScanConfig,
        tuple(MODE_SCAN_SECTIONS),
    )
    if args.seed is not None:
        config.seed = args.seed
    if args.N is not None:
        config.N = args.N
        config.__post_init__()

    run = document.get("run", {})
    k_scan = document.get("k_scan", {})
    if not isinstance(run, dict) or not isinstance(k_scan, dict):
        raise ValueError("The run and k_scan sections must be mappings.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    K_values = [float(value) for value in k_scan.get("K_values", [])]
    if not K_values or any(value <= 0 for value in K_values):
        raise ValueError("k_scan.K_values must contain positive values.")
    transition_threshold = float(k_scan.get("transition_threshold", 0.02))
    return config, run, K_values, transition_threshold


def save_run_files(
    run_directory: Path,
    config: ModeScanConfig,
    results: dict[float, ScanResult],
    K_values: list[float],
    transition_threshold: float,
) -> None:
    """Save one complete K-scan result."""
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, MODE_SCAN_SECTIONS))
    resolved["k_scan"] = {
        "K_values": K_values,
        "transition_threshold": transition_threshold,
    }
    arrays = {}
    rows = []
    for K, result in results.items():
        prefix = f"K_{K:g}_"
        arrays.update(
            {prefix + name: values for name, values in scan_result_arrays(result).items()}
        )
        rows.extend({"K": K, **row} for row in scan_metric_rows(result))
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", arrays)
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, rows)
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))


def main() -> None:
    """Run the scan and save all output files."""
    args = parse_args()
    config, run, K_values, transition_threshold = load_config(args)
    output_root = Path(run.get("output_root", "output/scans"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    experiment = str(run.get("experiment", "K_rhoF_modes"))
    run_directory = create_run_directory(output_root, experiment, run.get("run_id"))
    results = scan_all_K(config, K_values)
    figure = plot_results(results, config, transition_threshold)
    save_run_files(run_directory, config, results, K_values, transition_threshold)
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved run to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
