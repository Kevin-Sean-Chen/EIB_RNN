"""Run the supporting connectivity-spectrum analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.analysis.spectral import SPECTRAL_SECTIONS, SpectralConfig, SpectralResult, run_spectral_scan
from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz


METRIC_FIELDS = ["K", "phase", "phase_over_pi", "spectral_abscissa", "stable"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analyses/spectral.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[SpectralConfig, dict]:
    """Load one spectral analysis configuration."""
    config, document = load_dataclass_sections(
        args.config,
        SpectralConfig,
        tuple(SPECTRAL_SECTIONS),
    )
    run = document.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("The run section must be a mapping.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    return config, run


def metric_rows(result: SpectralResult) -> list[dict]:
    """Return one summary row for each scan point."""
    rows = []
    for K_index, K in enumerate(result.K_values):
        for phase_index, phase in enumerate(result.phase_values):
            rows.append(
                {
                    "K": K,
                    "phase": phase,
                    "phase_over_pi": phase / np.pi,
                    "spectral_abscissa": result.spectral_abscissa[K_index, phase_index],
                    "stable": bool(result.stable[K_index, phase_index]),
                }
            )
    return rows


def result_arrays(result: SpectralResult) -> dict[str, np.ndarray]:
    """Return all spectral results as named arrays."""
    return {
        "K_values": result.K_values,
        "phase_values": result.phase_values,
        "eigenvalues": result.eigenvalues,
        "leading_real": result.leading_real,
        "spectral_abscissa": result.spectral_abscissa,
        "stable": result.stable,
        "left_patterns": result.left_patterns,
        "right_patterns": result.right_patterns,
    }


def plot_result(result: SpectralResult) -> plt.Figure:
    """Plot each complex spectrum and the spectral-abscissa summary."""
    rows = len(result.K_values)
    columns = len(result.phase_values) + 1
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 2.7 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for K_index, K in enumerate(result.K_values):
        for phase_index, phase in enumerate(result.phase_values):
            axis = axes[K_index, phase_index]
            values = result.eigenvalues[K_index, phase_index]
            axis.scatter(values.real, values.imag, s=7, alpha=0.65)
            axis.axvline(0, color="black", linewidth=0.7)
            axis.set_title(f"K={K:g}, phase={phase / np.pi:.1f} pi")
            if K_index == rows - 1:
                axis.set_xlabel("Real part")
            if phase_index == 0:
                axis.set_ylabel("Imaginary part")

    summary_axis = axes[:, -1]
    for index, axis in enumerate(summary_axis):
        if index == 0:
            image = axis.imshow(
                result.spectral_abscissa,
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
            )
            axis.set(
                title="Spectral abscissa",
                xlabel="Phase",
                ylabel="K",
                xticks=np.arange(len(result.phase_values)),
                yticks=np.arange(len(result.K_values)),
                xticklabels=[f"{value / np.pi:.1f} pi" for value in result.phase_values],
                yticklabels=[f"{value:g}" for value in result.K_values],
            )
            figure.colorbar(image, ax=axis)
        else:
            axis.remove()
    figure.suptitle("Legacy block-operator spectrum")
    return figure


def main() -> None:
    """Run the analysis and save all outputs."""
    args = parse_args()
    config, run = load_config(args)
    output_root = Path(run.get("output_root", "output/analyses"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_directory = create_run_directory(
        output_root,
        str(run.get("experiment", "spectral")),
        run.get("run_id"),
    )
    result = run_spectral_scan(config)
    figure = plot_result(result)
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, SPECTRAL_SECTIONS))
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", result_arrays(result))
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, metric_rows(result))
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved analysis to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
