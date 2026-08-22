"""Scan the transition from local spatial modes to non-local network modes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.analysis.local_network_modes import (
    MODE_SCAN_SECTIONS,
    ModeScanConfig,
    plot_result,
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


def parse_args(
    default_config: Path = Path("configs/baseline/local_network_modes.yaml"),
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> tuple[ModeScanConfig, dict]:
    """Load the scan configuration and apply small command-line overrides."""
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
    if not isinstance(run, dict):
        raise ValueError("Configuration section must be a mapping: run")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    return config, run


def save_run_files(run_directory: Path, config: ModeScanConfig, result) -> None:
    """Save the resolved configuration and all numerical results."""
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, MODE_SCAN_SECTIONS))
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", scan_result_arrays(result))
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, scan_metric_rows(result))
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))


def main() -> None:
    """Run the scan and save its summary figure."""
    args = parse_args()
    config, run = load_config(args)
    output_root = Path(run.get("output_root", "output/scans"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    experiment = str(run.get("experiment", "local_network_modes"))
    run_directory = create_run_directory(output_root, experiment, run.get("run_id"))
    result = run_scan(config)
    figure = plot_result(result, config)
    save_run_files(run_directory, config, result)
    figure_path = run_directory / "summary.png"
    figure.savefig(figure_path, dpi=180)
    print(f"Saved run to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
