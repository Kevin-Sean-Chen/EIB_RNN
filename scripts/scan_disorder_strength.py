"""Scan low-rank disorder strength and network coupling strength."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.disorder import DISORDER_SECTIONS, DisorderConfig, simulate_disorder
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz
from src.metrics import latent_coherence, linear_dimension, spatial_coherence


METRIC_FIELDS = [
    "K",
    "strength",
    "linear_dimension",
    "spatial_coherence",
    "latent_coherence_mean",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scans/disorder_strength.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(
    args: argparse.Namespace,
) -> tuple[DisorderConfig, dict, list[float], list[float]]:
    """Load the model and scan configuration."""
    config, document = load_dataclass_sections(
        args.config,
        DisorderConfig,
        tuple(DISORDER_SECTIONS),
    )
    run = document.get("run", {})
    scan = document.get("scan", {})
    if not isinstance(run, dict) or not isinstance(scan, dict):
        raise ValueError("The run and scan sections must be mappings.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    K_values = [float(value) for value in scan.get("K_values", [])]
    strength_values = [float(value) for value in scan.get("strength_values", [])]
    if not K_values or not strength_values:
        raise ValueError("The scan values must not be empty.")
    return config, run, K_values, strength_values


def run_scan(
    base_config: DisorderConfig,
    K_values: list[float],
    strength_values: list[float],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run all scan points and return metrics and example activity."""
    shape = (len(K_values), len(strength_values))
    dimensions = np.empty(shape)
    spatial = np.empty(shape)
    latent = np.empty(shape)
    rows = []
    example_activity = None
    example_patterns = None

    for K_index, K in enumerate(K_values):
        for strength_index, strength in enumerate(strength_values):
            config = replace(base_config, K=K, strength=strength)
            print(f"Run K={K:g}, strength={strength:g}")
            result = simulate_disorder(config)
            activity = result.excitatory.reshape(config.N**2, -1)
            latent_values = [
                latent_coherence(activity, result.left_patterns[:, index])
                for index in range(config.rank)
            ]
            dimension_value = linear_dimension(result.excitatory)
            spatial_value = spatial_coherence(
                result.excitatory[:, :, result.excitatory.shape[2] // 2]
            )
            latent_value = float(np.mean(latent_values))
            dimensions[K_index, strength_index] = dimension_value
            spatial[K_index, strength_index] = spatial_value
            latent[K_index, strength_index] = latent_value
            rows.append(
                {
                    "K": K,
                    "strength": strength,
                    "linear_dimension": dimension_value,
                    "spatial_coherence": spatial_value,
                    "latent_coherence_mean": latent_value,
                }
            )
            if example_activity is None or (
                K_index == len(K_values) - 1
                and strength_index == len(strength_values) - 1
            ):
                example_activity = result.excitatory
                example_patterns = result.left_patterns

    arrays = {
        "K_values": np.asarray(K_values),
        "strength_values": np.asarray(strength_values),
        "linear_dimension": dimensions,
        "spatial_coherence": spatial,
        "latent_coherence_mean": latent,
        "example_activity": example_activity,
        "example_left_patterns": example_patterns,
    }
    return rows, arrays


def plot_scan(arrays: dict[str, np.ndarray]) -> plt.Figure:
    """Plot three disorder scan metrics."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    metrics = [
        ("linear_dimension", "Linear dimension"),
        ("spatial_coherence", "Spatial coherence"),
        ("latent_coherence_mean", "Latent coherence"),
    ]
    strengths = arrays["strength_values"]
    K_values = arrays["K_values"]
    for axis, (key, title) in zip(axes, metrics):
        image = axis.imshow(arrays[key], origin="lower", aspect="auto")
        axis.set(
            title=title,
            xlabel="Disorder strength",
            ylabel="K",
            xticks=np.arange(len(strengths)),
            yticks=np.arange(len(K_values)),
            xticklabels=[f"{value:g}" for value in strengths],
            yticklabels=[f"{value:g}" for value in K_values],
        )
        figure.colorbar(image, ax=axis)
    return figure


def main() -> None:
    """Run the scan and save all outputs."""
    args = parse_args()
    config, run, K_values, strength_values = load_config(args)
    output_root = Path(run.get("output_root", "output/scans"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_directory = create_run_directory(
        output_root,
        str(run.get("experiment", "disorder_strength")),
        run.get("run_id"),
    )
    rows, arrays = run_scan(config, K_values, strength_values)
    figure = plot_scan(arrays)
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, DISORDER_SECTIONS))
    resolved["scan"] = {
        "K_values": K_values,
        "strength_values": strength_values,
    }
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", arrays)
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, rows)
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved scan to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
