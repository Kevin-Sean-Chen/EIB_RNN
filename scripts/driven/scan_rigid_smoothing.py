"""Compare rigid reconstruction across spatial smoothing widths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz
from src.learning.rigid_reconstruction import train_rigid_reconstruction
from src.models.rigid_reconstruction import (
    NonSpatialRigidReconstructionReservoir,
    RigidReconstructionReservoir,
)
from src.stimuli import fixed_spatial_permutation, rigid_shift_movie
from src.tasks.rigid_reconstruction import RIGID_RECONSTRUCTION_SECTIONS, RigidReconstructionConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/driven/rigid_reconstruction_smoothing_scan.yaml"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the smoothing-width comparison and save its outputs."""
    args = parse_args()
    config, document = load_dataclass_sections(
        args.config, RigidReconstructionConfig, tuple(RIGID_RECONSTRUCTION_SECTIONS),
    )
    run = document.get("run", {})
    scan = document.get("scan", {})
    sigma_pixels = [float(value) for value in scan.get("sigma_pixels", [])]
    input_modes = [str(value) for value in scan.get("input_modes", ["raw"])]
    permutation_seed = int(scan.get("permutation_seed", config.seed + 100))
    if not sigma_pixels:
        raise ValueError("The scan must contain at least one sigma value.")
    if not input_modes or any(mode not in ("raw", "shuffled") for mode in input_modes):
        raise ValueError("Input modes must contain raw or shuffled.")

    output_root = Path(run.get("output_root", "output/tasks"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_id = args.run_id if args.run_id is not None else run.get("run_id")
    directory = create_run_directory(
        output_root, str(run.get("experiment", "rigid_smoothing_scan")), run_id,
    )

    model_types = ("spatial", "non_spatial")
    mse = np.zeros((2, len(input_modes), len(sigma_pixels)))
    r2 = np.zeros_like(mse)
    rank_90 = np.zeros(len(sigma_pixels), dtype=int)
    participation = np.zeros(len(sigma_pixels))
    example_frames = np.zeros((len(input_modes), len(sigma_pixels), config.N, config.N))
    rows = []
    for width_index, sigma in enumerate(sigma_pixels):
        config.smoothing_width = sigma / config.N
        stimulus, target = rigid_shift_movie(
            config.N, config.steps, config.dt, config.smoothing_width,
            config.shift_distance, config.seed, config.device,
        )
        movie_matrix = stimulus.cpu().numpy().reshape(config.N**2, config.steps).T
        movie_matrix -= movie_matrix.mean(axis=0, keepdims=True)
        variance = np.linalg.svd(movie_matrix, compute_uv=False) ** 2
        variance_fraction = np.cumsum(variance) / variance.sum()
        rank_90[width_index] = int(np.searchsorted(variance_fraction, 0.9) + 1)
        participation[width_index] = variance.sum() ** 2 / np.sum(variance**2)
        for mode_index, input_mode in enumerate(input_modes):
            model_input = (
                stimulus if input_mode == "raw"
                else fixed_spatial_permutation(stimulus, permutation_seed)
            )
            example_frames[mode_index, width_index] = model_input[:, :, 0].cpu().numpy()
            for model_index, model_type in enumerate(model_types):
                config.model_type = model_type
                model_class = (
                    RigidReconstructionReservoir
                    if model_type == "spatial"
                    else NonSpatialRigidReconstructionReservoir
                )
                result = train_rigid_reconstruction(model_class(config), model_input, target)
                mse[model_index, mode_index, width_index] = result.evaluation_mse
                r2[model_index, mode_index, width_index] = result.evaluation_r2
                rows.append({
                    "model_type": model_type,
                    "input_mode": input_mode,
                    "sigma_pixels": sigma,
                    "stimulus_rank_90": rank_90[width_index],
                    "stimulus_participation": participation[width_index],
                    "mse": result.evaluation_mse,
                    "r2": result.evaluation_r2,
                })
                print(
                    f"{model_type}, {input_mode}: sigma={sigma:g}, "
                    f"R2={result.evaluation_r2:.6g}"
                )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for model_index, model_type in enumerate(model_types):
        for mode_index, input_mode in enumerate(input_modes):
            line_style = "-" if input_mode == "raw" else "--"
            label = f"{model_type}, {input_mode}"
            axes[0, 0].plot(sigma_pixels, r2[model_index, mode_index], "o" + line_style, label=label)
            axes[0, 1].plot(sigma_pixels, mse[model_index, mode_index], "o" + line_style, label=label)
    axes[0, 0].set(xlabel="Smoothing sigma (pixels)", ylabel="R2", title="Reconstruction score")
    axes[0, 1].set(xlabel="Smoothing sigma (pixels)", ylabel="MSE", title="Reconstruction error")
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].imshow(example_frames[0, -1], cmap="gray")
    axes[1, 0].set(title=f"Raw, sigma = {sigma_pixels[-1]:g} pixels", xticks=[], yticks=[])
    last_mode = len(input_modes) - 1
    axes[1, 1].imshow(example_frames[last_mode, -1], cmap="gray")
    axes[1, 1].set(title=f"{input_modes[last_mode].title()}, sigma = {sigma_pixels[-1]:g} pixels", xticks=[], yticks=[])
    figure.suptitle("Rigid reconstruction: effect of spatial smoothing")

    resolved = {"run": {"run_directory": str(directory)}}
    resolved.update(dataclass_to_sections(config, RIGID_RECONSTRUCTION_SECTIONS))
    resolved["scan"] = {
        "sigma_pixels": sigma_pixels, "model_types": list(model_types),
        "input_modes": input_modes, "permutation_seed": permutation_seed,
    }
    save_yaml(directory / "config.yaml", resolved)
    save_csv(
        directory / "metrics.csv",
        ["model_type", "input_mode", "sigma_pixels", "stimulus_rank_90", "stimulus_participation", "mse", "r2"],
        rows,
    )
    save_npz(directory / "results.npz", {
        "sigma_pixels": np.asarray(sigma_pixels), "mse": mse, "r2": r2,
        "rank_90": rank_90, "participation": participation,
        "example_frames": example_frames,
    })
    save_json(directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(directory / "summary.png", dpi=180)
    print(f"Saved smoothing scan to {directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
