"""Scan rank-one Gabor disorder across K and phase."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.analysis.rank_one import RankOneResult, run_rank_one_scan
from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.disorder import DISORDER_SECTIONS, DisorderConfig
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz


METRIC_FIELDS = [
    "K",
    "phase",
    "phase_over_pi",
    "latent_coherence",
    "active_fraction",
    "balance_error",
    "latent_acf_peak",
    "activity_acf_peak",
    "eigenvector_alignment",
    "spectral_abscissa",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/disorder/rank_one.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(
    args: argparse.Namespace,
) -> tuple[DisorderConfig, dict, list[float], list[float], int]:
    """Load one rank-one scan configuration."""
    config, document = load_dataclass_sections(
        args.config,
        DisorderConfig,
        tuple(DISORDER_SECTIONS),
    )
    run = document.get("run", {})
    scan = document.get("scan", {})
    analysis = document.get("analysis", {})
    if not all(isinstance(section, dict) for section in (run, scan, analysis)):
        raise ValueError("The run, scan, and analysis sections must be mappings.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    K_values = [float(value) for value in scan.get("K_values", [])]
    phase_values = [float(value) for value in scan.get("phase_values", [])]
    acf_sample_count = int(analysis.get("acf_sample_count", 100))
    if not K_values or not phase_values or acf_sample_count <= 0:
        raise ValueError("Scan values and acf_sample_count must be valid.")
    return config, run, K_values, phase_values, acf_sample_count


def metric_rows(result: RankOneResult) -> list[dict]:
    """Return one metric row for each scan point."""
    rows = []
    metric_names = [
        "latent_coherence",
        "active_fraction",
        "balance_error",
        "latent_acf_peak",
        "activity_acf_peak",
        "eigenvector_alignment",
        "spectral_abscissa",
    ]
    for K_index, K in enumerate(result.K_values):
        for phase_index, phase in enumerate(result.phase_values):
            row = {"K": K, "phase": phase, "phase_over_pi": phase / np.pi}
            for name in metric_names:
                row[name] = getattr(result, name)[K_index, phase_index]
            rows.append(row)
    return rows


def result_arrays(result: RankOneResult) -> dict[str, np.ndarray]:
    """Return all rank-one results as named arrays."""
    return {
        "K_values": result.K_values,
        "phase_values": result.phase_values,
        "latent_coherence": result.latent_coherence,
        "active_fraction": result.active_fraction,
        "balance_error": result.balance_error,
        "latent_acf_peak": result.latent_acf_peak,
        "activity_acf_peak": result.activity_acf_peak,
        "eigenvector_alignment": result.eigenvector_alignment,
        "spectral_abscissa": result.spectral_abscissa,
        "kappa": result.kappa,
        "example_activity": result.example_activity,
        "left_patterns": result.left_patterns,
        "right_patterns": result.right_patterns,
    }


def plot_result(result: RankOneResult) -> plt.Figure:
    """Plot rank-one activity metrics and latent time courses."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    heatmaps = [
        ("latent_acf_peak", "Latent ACF peak"),
        ("latent_coherence", "Latent coherence"),
        ("eigenvector_alignment", "Eigenvector alignment"),
        ("balance_error", "Balance error"),
        ("spectral_abscissa", "Spectral abscissa"),
    ]
    phase_labels = [f"{value / np.pi:.1f} pi" for value in result.phase_values]
    K_labels = [f"{value:g}" for value in result.K_values]
    for axis, (name, title) in zip(axes.flat[:5], heatmaps):
        image = axis.imshow(getattr(result, name), origin="lower", aspect="auto")
        axis.set(
            title=title,
            xlabel="Phase",
            ylabel="K",
            xticks=np.arange(len(phase_labels)),
            yticks=np.arange(len(K_labels)),
            xticklabels=phase_labels,
            yticklabels=K_labels,
        )
        figure.colorbar(image, ax=axis)
    trace_axis = axes.flat[5]
    for phase_index, label in enumerate(phase_labels):
        trace_axis.plot(result.kappa[-1, phase_index], label=label)
    trace_axis.set(
        title=f"Latent traces at K={result.K_values[-1]:g}",
        xlabel="Recorded step",
        ylabel="Kappa",
    )
    trace_axis.legend()
    figure.suptitle("Rank-one Gabor disorder")
    return figure


def main() -> None:
    """Run the scan and save all outputs."""
    args = parse_args()
    config, run, K_values, phase_values, acf_sample_count = load_config(args)
    output_root = Path(run.get("output_root", "output/scans"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_directory = create_run_directory(
        output_root,
        str(run.get("experiment", "rank_one")),
        run.get("run_id"),
    )
    result = run_rank_one_scan(config, K_values, phase_values, acf_sample_count)
    figure = plot_result(result)
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, DISORDER_SECTIONS))
    resolved["scan"] = {"K_values": K_values, "phase_values": phase_values}
    resolved["analysis"] = {"acf_sample_count": acf_sample_count}
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", result_arrays(result))
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, metric_rows(result))
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved scan to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
