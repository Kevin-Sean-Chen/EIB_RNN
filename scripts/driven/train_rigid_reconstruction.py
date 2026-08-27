"""Train same-time rigid-shift reconstruction with RLS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz
from src.learning.rigid_reconstruction import driven_lyapunov_exponent, train_rigid_reconstruction
from src.models.rigid_reconstruction import (
    NonSpatialRigidReconstructionReservoir,
    RandomEIRigidReconstructionReservoir,
    RigidReconstructionReservoir,
)
from src.stimuli import rigid_shift_movie
from src.tasks.rigid_reconstruction import RIGID_RECONSTRUCTION_SECTIONS, RigidReconstructionConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/driven/rigid_reconstruction.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train the readout and save reproducible outputs."""
    args = parse_args()
    config, document = load_dataclass_sections(
        args.config, RigidReconstructionConfig, tuple(RIGID_RECONSTRUCTION_SECTIONS),
    )
    run = document.get("run", {})
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    stimulus, target = rigid_shift_movie(
        config.N, config.steps, config.dt, config.smoothing_width,
        config.shift_distance, config.seed, config.device,
    )
    model_classes = {
        "spatial": RigidReconstructionReservoir,
        "non_spatial": NonSpatialRigidReconstructionReservoir,
        "random_ei": RandomEIRigidReconstructionReservoir,
    }
    model_class = model_classes[config.model_type]
    model = model_class(config)
    lyapunov = driven_lyapunov_exponent(model, stimulus, config.seed)
    result = train_rigid_reconstruction(model, stimulus, target)
    output_root = Path(run.get("output_root", "output/tasks"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    directory = create_run_directory(output_root, str(run.get("experiment", "rigid_reconstruction")), run.get("run_id"))
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(result.training_mse, "o-")
    if result.training_mse.size == 1:
        axes[0, 0].set_xlim(-0.5, 0.5)
    axes[0, 0].set(title=f"{config.learning_method.upper()} training", xlabel="Update", ylabel="MSE")
    axes[0, 1].plot(result.target, label="Target")
    axes[0, 1].plot(result.prediction, "--", label="Readout")
    axes[0, 1].set(title="Same-time reconstruction", xlabel="Step", ylabel="Angle")
    axes[0, 1].legend()
    axes[1, 0].imshow(stimulus[:, :, 0].cpu(), cmap="gray")
    axes[1, 0].set(title="Rigid pattern", xticks=[], yticks=[])
    axes[1, 1].imshow(result.activity, origin="lower", aspect="auto", cmap="viridis")
    axes[1, 1].set(title="Reservoir activity", xlabel="Step", ylabel="Unit")
    figure.suptitle(f"Rigid-shift reconstruction: {config.model_type}")
    resolved = {"run": {"run_directory": str(directory)}}
    resolved.update(dataclass_to_sections(config, RIGID_RECONSTRUCTION_SECTIONS))
    save_yaml(directory / "config.yaml", resolved)
    save_npz(directory / "results.npz", {**result.__dict__, "stimulus": stimulus.cpu().numpy()})
    save_csv(
        directory / "metrics.csv",
        ["model_type", "mse", "r2", "driven_lyapunov_per_second"],
        [{
            "model_type": config.model_type, "mse": result.evaluation_mse,
            "r2": result.evaluation_r2, "driven_lyapunov_per_second": lyapunov,
        }],
    )
    save_json(directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(directory / "summary.png", dpi=180)
    print(f"Saved training run to {directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
