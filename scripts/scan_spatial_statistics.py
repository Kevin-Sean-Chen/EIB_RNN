"""Plot direct spatial statistics across non-local connection strength."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.scan_local_network_modes import load_config, parse_args, save_run_files
from src.analysis.local_network_modes import run_scan
from src.io import create_run_directory


def plot_spatial_statistics(result, args):
    """Plot mode advantage and three direct spatial measurements."""
    figure, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    x = result.relative_strengths
    panels = (
        (result.transition_index, result.transition_std, "Network advantage"),
        (
            result.neighbor_correlation,
            result.neighbor_correlation_std,
            "Neighbor correlation",
        ),
        (
            result.correlation_length,
            result.correlation_length_std,
            "Correlation length (grid sites)",
        ),
        (
            result.low_k_fraction,
            result.low_k_fraction_std,
            "Low-k power fraction",
        ),
    )
    for axis, (values, spread, label) in zip(axes, panels):
        axis.plot(x, values, marker="o")
        axis.fill_between(x, values - spread, values + spread, alpha=0.2)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set(xlabel="Relative total strength, rho_F", ylabel=label)
    axes[0].axhline(0.02, color="black", linestyle=":", linewidth=1.5)
    axes[2].axhline(
        1.0 - np.exp(-1.0),
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Spatial shuffle null",
    )
    frequency = np.fft.fftfreq(args.N) * args.N
    wave_number = np.sqrt(frequency[:, None] ** 2 + frequency[None, :] ** 2)
    low_k_null = np.count_nonzero((wave_number > 0) & (wave_number <= 2.0)) / (
        args.N**2 - 1
    )
    axes[3].axhline(
        low_k_null,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Spatial shuffle null",
    )
    axes[2].legend()
    axes[3].legend()
    figure.suptitle(
        f"Spatial structure across non-local strength, "
        f"N={args.N}, K={args.K:g}, rank={args.rank}"
    )
    return figure


def main() -> None:
    args = parse_args(Path("configs/scans/spatial_statistics.yaml"))
    config, run = load_config(args)
    output_root = Path(run.get("output_root", "output/scans"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    experiment = str(run.get("experiment", "spatial_statistics"))
    run_directory = create_run_directory(output_root, experiment, run.get("run_id"))
    result = run_scan(config)
    figure = plot_spatial_statistics(result, config)
    save_run_files(run_directory, config, result)
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved run to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
