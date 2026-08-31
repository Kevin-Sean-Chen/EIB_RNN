"""Run DMD on baseline spatial E/I activity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.analysis.dmd import DMDConfig, DMDResult, DMD_SECTIONS, spatial_dmd
from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz
from src.local import LOCAL_SECTIONS, LocalConfig, simulate_local


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/analyses/dmd.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def plot_result(activity: np.ndarray, result: DMDResult, shown_modes: int) -> plt.Figure:
    """Plot activity, DMD spectrum, errors, modes, and dispersion."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].imshow(activity[:, :, activity.shape[-1] // 2], origin="lower", cmap="viridis")
    axes[0, 0].set(title="Example E activity", xticks=[], yticks=[])
    angle = np.linspace(0, 2 * np.pi, 400)
    axes[0, 1].plot(np.cos(angle), np.sin(angle), "k--", alpha=0.4)
    axes[0, 1].scatter(result.eigenvalues.real, result.eigenvalues.imag, s=18)
    axes[0, 1].set(title="DMD eigenvalues", xlabel="Real part", ylabel="Imaginary part", aspect="equal")
    axes[0, 2].plot(result.rank_values, result.rank_errors, "o-")
    axes[0, 2].set(title="One-step prediction", xlabel="Rank", ylabel="Relative error")

    mode_count = min(shown_modes, len(result.modes))
    mode_strip = np.concatenate([result.modes[index].real for index in range(mode_count)], axis=1)
    axes[1, 0].imshow(mode_strip, origin="lower", cmap="coolwarm")
    axes[1, 0].set(title=f"Real parts of first {mode_count} modes", xticks=[], yticks=[])
    scatter = axes[1, 1].scatter(
        result.dominant_wavenumbers, np.abs(result.angular_frequencies),
        c=result.growth_rates, cmap="coolwarm", edgecolor="black",
    )
    axes[1, 1].set(title="DMD dispersion", xlabel="Dominant wave number", ylabel="Angular frequency")
    figure.colorbar(scatter, ax=axes[1, 1], label="Growth rate")
    axes[1, 2].scatter(result.dominant_wavenumbers, result.growth_rates, edgecolor="black")
    axes[1, 2].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1, 2].set(title="Growth versus scale", xlabel="Dominant wave number", ylabel="Growth rate")
    figure.suptitle("Dynamic mode decomposition of spatial E/I activity")
    return figure


def main() -> None:
    """Simulate baseline dynamics, run DMD, and save outputs."""
    args = parse_args()
    local_config, document = load_dataclass_sections(
        args.config, LocalConfig, tuple(LOCAL_SECTIONS),
    )
    dmd_config, _ = load_dataclass_sections(args.config, DMDConfig, tuple(DMD_SECTIONS))
    local_result = simulate_local(local_config)
    sample_dt = local_config.dt * local_config.steps_per_record
    result = spatial_dmd(local_result.excitatory, sample_dt, dmd_config)
    figure = plot_result(local_result.excitatory, result, dmd_config.shown_modes)

    run = document.get("run", {})
    output_root = Path(run.get("output_root", "output/analyses"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    directory = create_run_directory(
        output_root, str(run.get("experiment", "dmd")),
        args.run_id if args.run_id is not None else run.get("run_id"),
    )
    resolved = {"run": {"run_directory": str(directory)}}
    resolved.update(dataclass_to_sections(local_config, LOCAL_SECTIONS))
    resolved.update(dataclass_to_sections(dmd_config, DMD_SECTIONS))
    save_yaml(directory / "config.yaml", resolved)
    save_npz(directory / "results.npz", {
        "excitatory": local_result.excitatory, "inhibitory": local_result.inhibitory,
        "time": local_result.time, "eigenvalues": result.eigenvalues,
        "modes": result.modes, "growth_rates": result.growth_rates,
        "angular_frequencies": result.angular_frequencies,
        "dominant_wavenumbers": result.dominant_wavenumbers,
        "singular_values": result.singular_values,
        "rank_values": result.rank_values, "rank_errors": result.rank_errors,
    })
    save_csv(
        directory / "metrics.csv",
        ["mode", "eigenvalue_real", "eigenvalue_imaginary", "growth_rate", "angular_frequency", "dominant_wavenumber"],
        [
            {
                "mode": index, "eigenvalue_real": value.real,
                "eigenvalue_imaginary": value.imag,
                "growth_rate": result.growth_rates[index],
                "angular_frequency": result.angular_frequencies[index],
                "dominant_wavenumber": result.dominant_wavenumbers[index],
            }
            for index, value in enumerate(result.eigenvalues)
        ],
    )
    save_json(directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(directory / "summary.png", dpi=180)
    print(f"Saved DMD analysis to {directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
