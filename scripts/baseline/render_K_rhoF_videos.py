"""Render matched activity videos from one K and rho_F scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml


def select_condition_indices(
    rho_f: np.ndarray,
    advantage: np.ndarray,
    stable: np.ndarray,
) -> tuple[int, int, int]:
    """Return indices for zero, peak, and largest stable rho_F."""
    valid = stable & np.isfinite(rho_f) & np.isfinite(advantage)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        raise ValueError("The selected K has no stable conditions.")
    zero = indices[np.argmin(np.abs(rho_f[indices]))]
    peak = indices[np.argmax(advantage[indices])]
    high = indices[np.argmax(rho_f[indices])]
    return int(zero), int(peak), int(high)


def select_frame_indices(sample_count: int, frame_count: int) -> np.ndarray:
    """Return uniform frame indices that include both recording endpoints."""
    if sample_count <= 0 or frame_count <= 0:
        raise ValueError("Sample count and frame count must be positive.")
    count = min(sample_count, frame_count)
    return np.linspace(0, sample_count - 1, count, dtype=int)


def activity_frames(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and centered RMS-scaled square activity frames."""
    site_count, _ = activity.shape
    N = int(np.sqrt(site_count))
    if N * N != site_count:
        raise ValueError("The activity site count must be a square number.")
    raw = activity.T.reshape(-1, N, N)
    centered = activity - activity.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2))
    if not np.isfinite(rms) or rms <= np.finfo(float).eps:
        normalized = np.zeros_like(centered)
    else:
        normalized = centered / rms
    return raw, normalized.T.reshape(-1, N, N)


def shared_raw_limit(activities: list[np.ndarray]) -> float:
    """Return one robust upper color limit for all raw activities."""
    pooled = np.concatenate([activity.ravel() for activity in activities])
    limit = float(np.nanquantile(pooled, 0.995))
    return limit if limit > 0.0 else 1.0


def mean_reconstruction_score(
    curve: np.ndarray,
    shell_counts: np.ndarray,
    mode_limit: int,
) -> float:
    """Return mean captured variance at complete shells within a mode limit."""
    selected = shell_counts[shell_counts <= mode_limit]
    if selected.size == 0:
        raise ValueError("The mode limit does not include a complete shell.")
    return float(np.mean(curve[selected - 1]))


def render_condition_video(
    path: Path,
    activity: np.ndarray,
    frame_indices: np.ndarray,
    raw_limit: float,
    K: float,
    rho_f: float,
    sample_interval: float,
    fps: int,
    metrics: dict[str, float],
) -> None:
    """Render one raw and normalized excitatory activity video."""
    raw, normalized = activity_frames(activity)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    raw_axis, normalized_axis = axes
    first = int(frame_indices[0])
    raw_image = raw_axis.imshow(
        raw[first],
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=raw_limit,
        interpolation="nearest",
    )
    normalized_image = normalized_axis.imshow(
        normalized[first],
        origin="lower",
        cmap="RdBu_r",
        vmin=-3.0,
        vmax=3.0,
        interpolation="nearest",
    )
    raw_axis.set(title="Excitatory rate", xlabel="x", ylabel="y")
    normalized_axis.set(
        title="Centered activity in RMS units",
        xlabel="x",
        ylabel="y",
    )
    figure.colorbar(raw_image, ax=raw_axis, fraction=0.046, pad=0.04)
    figure.colorbar(
        normalized_image,
        ax=normalized_axis,
        fraction=0.046,
        pad=0.04,
    )
    metric_label = (
        f"Local={metrics['local']:.3f}, Full={metrics['full']:.3f}, "
        f"Delta={metrics['advantage']:.3f}, "
        f"Non-local={metrics['nonlocal']:.3f}"
    )

    def update(position: int) -> tuple:
        index = int(frame_indices[position])
        raw_image.set_data(raw[index])
        normalized_image.set_data(normalized[index])
        figure.suptitle(
            f"K={K:g}, rho_F={rho_f:g}, t={index * sample_interval:.4f} s\n"
            f"{metric_label}"
        )
        return raw_image, normalized_image

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--K", type=float, default=10000.0)
    parser.add_argument("--frame-count", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    """Render zero, peak, and high rho_F videos from one saved scan."""
    args = parse_args()
    config_path = args.run_directory / "config.yaml"
    results_path = args.run_directory / "results.npz"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    sample_interval = float(config["simulation"]["dt"]) * int(
        config["simulation"]["sample_every"]
    )
    mode_limit = int(config["analysis"]["auc_modes"])
    prefix = f"K_{args.K:g}_"

    with np.load(results_path) as arrays:
        rho_f = arrays[prefix + "relative_strengths"]
        advantage = arrays[prefix + "transition_index"]
        stable = arrays[prefix + "stable"]
        condition_indices = select_condition_indices(rho_f, advantage, stable)
        activities = [
            arrays[prefix + "example_rates"][index] for index in condition_indices
        ]
        local_curves = arrays[prefix + "local_curves"]
        full_curves = arrays[prefix + "network_curves"]
        shell_counts = arrays[prefix + "geometric_shell_counts"]
        nonlocal_fraction = arrays[prefix + "nonlocal_fraction"]

    frame_indices = select_frame_indices(
        sample_count=activities[0].shape[1],
        frame_count=args.frame_count,
    )
    raw_limit = shared_raw_limit(activities)
    video_records = []
    for index, activity in zip(condition_indices, activities):
        value = float(rho_f[index])
        local_score = mean_reconstruction_score(
            local_curves[index], shell_counts, mode_limit
        )
        full_score = mean_reconstruction_score(
            full_curves[index], shell_counts, mode_limit
        )
        path = args.run_directory / f"rhoF_{value:g}.gif"
        metrics = {
            "local": local_score,
            "full": full_score,
            "advantage": full_score - local_score,
            "nonlocal": float(nonlocal_fraction[index]),
        }
        render_condition_video(
            path=path,
            activity=activity,
            frame_indices=frame_indices,
            raw_limit=raw_limit,
            K=args.K,
            rho_f=value,
            sample_interval=sample_interval,
            fps=args.fps,
            metrics=metrics,
        )
        video_records.append(
            {
                "path": str(path),
                "rho_f": value,
                **metrics,
            }
        )
        print(f"Saved video to {path}")

    metadata = {
        "K": args.K,
        "source_results": str(results_path),
        "frame_count": len(frame_indices),
        "fps": args.fps,
        "sample_interval": sample_interval,
        "raw_color_limit": raw_limit,
        "normalized_color_limits": [-3.0, 3.0],
        "videos": video_records,
    }
    with (args.run_directory / "videos.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
