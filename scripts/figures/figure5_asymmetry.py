"""Create Figure 5 for asymmetric local coupling and directed waves."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
from pathlib import Path
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.relu2D_asym import relu2D_bias
from src.analysis.asymmetric_waves import directed_wave_metrics
from src.config import load_yaml, save_yaml
from src.io import runtime_metadata, save_json, save_npz


PANEL_STEMS = (
    "panel_A_bias_schematic",
    "panel_B_K_bias_patterns",
    "panel_C_bias_metrics",
    "panel_D_noise_metrics",
)


@dataclass(frozen=True)
class Figure5Config:
    """Define the asymmetric-wave scan."""

    output_directory: Path
    N: int
    K_values: tuple[float, ...]
    bias_values: tuple[float, ...]
    noise_values: tuple[float, ...]
    seeds: tuple[int, ...]
    dt: float
    tau_e: float
    tau_i: float
    init_steps: int
    record_steps: int
    steps_per_record: int
    u_e: float
    u_i: float
    sigma_e: float
    sigma_i: float
    metric_lag_steps: int
    rate_cap: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/figures/figure5_asymmetry.yaml"),
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> tuple[Figure5Config, dict]:
    """Load the Figure 5 configuration."""
    document = load_yaml(path)
    run = document["run"]
    network = document["network"]
    simulation = document["simulation"]
    scan = document["scan"]
    analysis = document["analysis"]
    config = Figure5Config(
        output_directory=Path(run["output_directory"]),
        N=int(network["N"]),
        K_values=tuple(float(value) for value in scan["K_values"]),
        bias_values=tuple(float(value) for value in scan["bias_values"]),
        noise_values=tuple(float(value) for value in scan["noise_values"]),
        seeds=tuple(int(value) for value in scan["seeds"]),
        dt=float(simulation["dt"]),
        tau_e=float(simulation["tau_e"]),
        tau_i=float(simulation["tau_i"]),
        init_steps=int(simulation["init_steps"]),
        record_steps=int(simulation["record_steps"]),
        steps_per_record=int(simulation["steps_per_record"]),
        u_e=float(network["u_e"]),
        u_i=float(network["u_i"]),
        sigma_e=float(network["sigma_e"]),
        sigma_i=float(network["sigma_i"]),
        metric_lag_steps=int(analysis["metric_lag_steps"]),
        rate_cap=float(simulation["rate_cap"]),
    )
    if config.noise_values != (0.0, 0.5):
        raise ValueError("Figure 5 requires noise values 0 and 0.5.")
    return config, document


def _initial_state(config: Figure5Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one reproducible initial state near the balance solution."""
    rng = np.random.default_rng(seed)
    J0 = np.array([[1.0, -4.0], [2.0, -2.0]])
    baseline = -np.linalg.solve(J0, np.array([config.u_e, config.u_i]))
    re0 = baseline[0] + 0.05 * rng.random((config.N, config.N))
    ri0 = baseline[1] + 0.08 * rng.random((config.N, config.N))
    return re0, ri0


def _simulate_condition(
    config: Figure5Config,
    K: float,
    bias: float,
    noise: float,
    seed: int,
) -> np.ndarray:
    """Run initialization and measurement for one condition."""
    torch.manual_seed(seed)
    re0, ri0 = _initial_state(config, seed)
    frame_count = max(
        config.init_steps // config.steps_per_record,
        config.record_steps // config.steps_per_record,
    )
    input_pattern = torch.zeros((config.N, config.N, frame_count))
    with redirect_stdout(io.StringIO()):
        re_all, _ = relu2D_bias(
            config.N,
            config.dt,
            config.init_steps,
            config.record_steps,
            config.steps_per_record,
            "relu_gaussian",
            K,
            np.array([config.tau_e, config.tau_i]),
            np.array([config.u_e, config.u_i]),
            np.array([[1.0, -4.0], [2.0, -2.0]]),
            np.array([config.sigma_e, config.sigma_i]),
            input_pattern,
            re0,
            ri0,
            bias,
            noise_strength=noise,
            rate_cap=config.rate_cap,
        )
    return re_all


def run_scan(config: Figure5Config) -> dict[str, np.ndarray]:
    """Run the approved bias and noise comparisons."""
    shape = (
        len(config.bias_values),
        len(config.noise_values),
        len(config.K_values),
        len(config.seeds),
    )
    speed = np.full(shape, np.nan)
    coherence = np.full(shape, np.nan)
    patterns = np.full(
        (len(config.bias_values), len(config.K_values), config.N, config.N),
        np.nan,
    )
    expected_direction = (-1.0, -1.0)
    sample_interval = config.dt * config.steps_per_record
    last_bias_index = len(config.bias_values) - 1

    for bias_index, bias in enumerate(config.bias_values):
        noise_indices = (0, 1) if bias_index == last_bias_index else (0,)
        for noise_index in noise_indices:
            noise = config.noise_values[noise_index]
            for K_index, K in enumerate(config.K_values):
                for seed_index, seed in enumerate(config.seeds):
                    print(
                        f"Run b={bias:g}, noise={noise:g}, K={K:g}, seed={seed}",
                        flush=True,
                    )
                    movie = _simulate_condition(config, K, bias, noise, seed)
                    metrics = directed_wave_metrics(
                        movie,
                        sample_interval=sample_interval,
                        lag_steps=config.metric_lag_steps,
                        expected_direction=expected_direction,
                    )
                    speed[bias_index, noise_index, K_index, seed_index] = metrics.signed_speed
                    coherence[bias_index, noise_index, K_index, seed_index] = metrics.directional_coherence
                    if noise_index == 0 and seed_index == 0:
                        patterns[bias_index, K_index] = movie[:, :, movie.shape[2] // 2]

    return {
        "K_values": np.asarray(config.K_values),
        "bias_values": np.asarray(config.bias_values),
        "noise_values": np.asarray(config.noise_values),
        "seeds": np.asarray(config.seeds),
        "signed_speed": speed,
        "directional_coherence": coherence,
        "example_patterns": patterns,
    }


def save_results(
    results: dict[str, np.ndarray],
    config_document: dict,
    output_directory: Path,
) -> Path:
    """Save arrays, configuration, and runtime details."""
    output_directory.mkdir(parents=True, exist_ok=True)
    data_directory = output_directory / "data"
    data_directory.mkdir(exist_ok=True)
    result_path = data_directory / "results.npz"
    save_npz(result_path, results)
    save_yaml(data_directory / "config.yaml", config_document)
    save_json(data_directory / "metadata.json", runtime_metadata(repo_root))
    return result_path


def load_results(output_directory: Path) -> dict[str, np.ndarray]:
    """Load one saved Figure 5 scan."""
    with np.load(output_directory / "data" / "results.npz") as archive:
        return {name: archive[name] for name in archive.files}


def _plot_schematic() -> plt.Figure:
    """Plot centered inhibition and shifted excitation."""
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    axis.axis("off")
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    sheet = np.array([[0.10, 0.18], [0.70, 0.18], [0.88, 0.38], [0.28, 0.38]])
    axis.add_patch(Polygon(sheet, facecolor="#e8f2f5", edgecolor="#7898a3", linewidth=1.4))
    x = np.linspace(0.18, 0.78, 200)
    inhibitory = 0.50 + 0.18 * np.exp(-0.5 * ((x - 0.48) / 0.11) ** 2)
    excitatory = 0.70 + 0.18 * np.exp(-0.5 * ((x - 0.61) / 0.075) ** 2)
    axis.plot(x, inhibitory, color="#3977b7", linewidth=2.5, label=r"$G_I$")
    axis.plot(x, excitatory, color="#d95f4f", linewidth=2.5, label=r"shifted $G_E$")
    axis.annotate("bias, b", xy=(0.61, 0.91), xytext=(0.48, 0.91), ha="center", va="center",
                  arrowprops={"arrowstyle": "->", "color": "0.2", "linewidth": 1.5})
    axis.annotate("expected wave direction", xy=(0.73, 0.29), xytext=(0.42, 0.08), ha="center",
                  arrowprops={"arrowstyle": "->", "color": "#6a3d9a", "linewidth": 2.0})
    axis.legend(frameon=False, loc="upper left", ncol=2)
    axis.set_title("Spatial mismatch breaks translation symmetry", fontsize=13)
    figure.tight_layout()
    return figure


def _plot_pattern_grid(results: dict[str, np.ndarray]) -> plt.Figure:
    """Plot the K-by-bias activity pattern grid."""
    K_values = results["K_values"]
    bias_values = results["bias_values"]
    patterns = results["example_patterns"]
    figure, axes = plt.subplots(
        len(bias_values), len(K_values), figsize=(10.5, 9.5), squeeze=False
    )
    for bias_index, bias in enumerate(bias_values):
        row = patterns[bias_index]
        lower, upper = float(np.nanmin(row)), float(np.nanmax(row))
        for K_index, K in enumerate(K_values):
            axis = axes[bias_index, K_index]
            axis.imshow(row[K_index], cmap="magma", vmin=lower, vmax=upper, interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            if bias_index == 0:
                axis.set_title(rf"$K={K:g}$")
            if K_index == 0:
                axis.set_ylabel(rf"$b={bias:g}$", fontsize=11)
    figure.suptitle("Activity patterns across connection scale and spatial bias", fontsize=14)
    figure.tight_layout()
    return figure


def _mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and standard deviation across seeds."""
    return np.nanmean(values, axis=-1), np.nanstd(values, axis=-1)


def _metric_panel(
    results: dict[str, np.ndarray],
    condition_indices: tuple[tuple[int, int], tuple[int, int]],
    labels: tuple[str, str],
    title: str,
) -> plt.Figure:
    """Plot speed and directional coherence for two conditions."""
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = ("#5b2a86", "#1b9e77")
    K_values = results["K_values"]
    for (bias_index, noise_index), label, color in zip(condition_indices, labels, colors):
        speed_mean, speed_std = _mean_and_std(results["signed_speed"][bias_index, noise_index])
        coherence_mean, coherence_std = _mean_and_std(
            results["directional_coherence"][bias_index, noise_index]
        )
        axes[0].plot(K_values, speed_mean, "o-", color=color, label=label)
        axes[0].fill_between(K_values, speed_mean - speed_std, speed_mean + speed_std, color=color, alpha=0.16)
        axes[1].plot(K_values, coherence_mean, "o-", color=color, label=label)
        axes[1].fill_between(
            K_values,
            np.clip(coherence_mean - coherence_std, 0.0, 1.0),
            np.clip(coherence_mean + coherence_std, 0.0, 1.0),
            color=color,
            alpha=0.16,
        )
    axes[0].set(
        xscale="log",
        xlabel="Connection scale, K",
        ylabel="Signed wave speed (sheet lengths/s)",
    )
    axes[1].set(xscale="log", xlabel="Connection scale, K", ylabel="Directional coherence", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.set_xticks(K_values, labels=[f"{value:g}" for value in K_values])
        axis.legend(frameon=False)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    return figure


def plot_panels(results: dict[str, np.ndarray]) -> dict[str, plt.Figure]:
    """Return the four approved Figure 5 panels."""
    last_bias = len(results["bias_values"]) - 1
    return {
        "panel_A_bias_schematic": _plot_schematic(),
        "panel_B_K_bias_patterns": _plot_pattern_grid(results),
        "panel_C_bias_metrics": _metric_panel(
            results,
            ((0, 0), (last_bias, 0)),
            (r"$b=0$", rf"$b={results['bias_values'][last_bias]:g}$"),
            "Spatial bias creates directed waves",
        ),
        "panel_D_noise_metrics": _metric_panel(
            results,
            ((last_bias, 0), (last_bias, 1)),
            ("Noise = 0", "Noise = 0.5"),
            rf"Noise changes wave propagation at $b={results['bias_values'][last_bias]:g}$",
        ),
    }


def _trim_white(image: np.ndarray, padding: int = 8) -> np.ndarray:
    """Remove white margins from one panel image."""
    content = np.any(image[..., :3] < 0.995, axis=2)
    rows = np.flatnonzero(np.any(content, axis=1))
    columns = np.flatnonzero(np.any(content, axis=0))
    if not rows.size or not columns.size:
        return image
    return image[
        max(0, int(rows[0]) - padding):min(image.shape[0], int(rows[-1]) + padding + 1),
        max(0, int(columns[0]) - padding):min(image.shape[1], int(columns[-1]) + padding + 1),
    ]


def _place_panel(axis: plt.Axes, image: np.ndarray, label: str) -> None:
    """Place one panel image and label."""
    axis.imshow(image)
    axis.axis("off")
    axis.text(-0.01, 0.99, label, transform=axis.transAxes, fontsize=20,
              fontweight="bold", ha="right", va="top")


def save_outputs(panels: dict[str, plt.Figure], output_directory: Path) -> tuple[Path, ...]:
    """Save separate panels and one assembled Figure 5."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    images = {}
    for stem in PANEL_STEMS:
        png_path = output_directory / f"{stem}.png"
        pdf_path = output_directory / f"{stem}.pdf"
        panels[stem].savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
        panels[stem].savefig(pdf_path, bbox_inches="tight", facecolor="white")
        images[stem] = _trim_white(mpimg.imread(png_path))
        paths.extend((png_path, pdf_path))

    figure = plt.figure(figsize=(13.5, 17.0), facecolor="white")
    grid = figure.add_gridspec(3, 2, height_ratios=(0.55, 1.65, 0.8), hspace=0.04, wspace=0.04)
    placements = (
        (PANEL_STEMS[0], "A", grid[0, :]),
        (PANEL_STEMS[1], "B", grid[1, :]),
        (PANEL_STEMS[2], "C", grid[2, 0]),
        (PANEL_STEMS[3], "D", grid[2, 1]),
    )
    for stem, label, slot in placements:
        _place_panel(figure.add_subplot(slot), images[stem], label)
    figure.subplots_adjust(left=0.035, right=0.99, top=0.99, bottom=0.02)
    figure_png = output_directory / "figure5.png"
    figure_pdf = output_directory / "figure5.pdf"
    figure.savefig(figure_png, dpi=180, facecolor="white")
    figure.savefig(figure_pdf, facecolor="white")
    plt.close(figure)
    paths.extend((figure_png, figure_pdf))
    return tuple(paths)


def main() -> None:
    """Run or reload the scan, then save the Figure 5 draft."""
    arguments = parse_args()
    config, document = load_config(arguments.config)
    if arguments.plot_only:
        results = load_results(config.output_directory)
    else:
        results = run_scan(config)
        save_results(results, document, config.output_directory)
    panels = plot_panels(results)
    paths = save_outputs(panels, config.output_directory)
    for figure in panels.values():
        plt.close(figure)
    print("Saved Figure 5 files:")
    for path in paths:
        print(path)
    if arguments.show:
        plt.show()


if __name__ == "__main__":
    main()
