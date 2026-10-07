"""Create separate draft panels B--G for Figure 1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.analysis.spontaneous import (
    current_balance,
    pca_variance_curve,
    participation_dimension,
    radial_spatial_correlation,
    radial_spatial_spectrum,
    temporal_autocorrelation,
    temporal_spectrum,
)
from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import runtime_metadata, save_csv, save_json, save_npz
from src.local import LocalConfig, simulate_local


FIGURE1_SECTIONS = {
    "network": (
        "N", "J_ee", "J_ei", "J_ie", "J_ii", "sigma_e", "sigma_i", "u_e", "u_i",
    ),
    "simulation": (
        "device", "dt", "init_steps", "record_steps", "steps_per_record", "tau_e", "tau_i",
    ),
    "scan": ("K_values", "panel_K_values", "seeds"),
    "analysis": ("max_lag_seconds", "spectrum_window_samples"),
}


@dataclass
class Figure1Config:
    """Define the matched simulations and analyses for panels B--G."""

    N: int = 31
    J_ee: float = 1.0
    J_ei: float = -4.0
    J_ie: float = 2.0
    J_ii: float = -2.0
    sigma_e: float = 0.05
    sigma_i: float = 0.05 * np.sqrt(2)
    u_e: float = 10.0
    u_i: float = 0.0
    device: str = "cpu"
    dt: float = 0.0001
    init_steps: int = 5000
    record_steps: int = 20000
    steps_per_record: int = 5
    tau_e: float = 0.01
    tau_i: float = 0.02
    K_values: list[float] | None = None
    panel_K_values: list[float] | None = None
    seeds: list[int] | None = None
    max_lag_seconds: float = 0.25
    spectrum_window_samples: int = 1000

    def __post_init__(self) -> None:
        if self.K_values is None:
            self.K_values = [1.0, 100.0, 10000.0]
        if self.panel_K_values is None:
            self.panel_K_values = [1.0, 100.0, 10000.0]
        if self.seeds is None:
            self.seeds = [7, 11, 19]
        if not self.K_values or any(value <= 0 for value in self.K_values):
            raise ValueError("K_values must contain positive values.")
        if not self.panel_K_values or any(value <= 0 for value in self.panel_K_values):
            raise ValueError("panel_K_values must contain positive values.")
        if any(value not in self.K_values for value in self.panel_K_values):
            raise ValueError("panel_K_values must be in K_values.")
        if not self.seeds:
            raise ValueError("seeds must not be empty.")

    def local_config(self, K: float, seed: int) -> LocalConfig:
        """Return one local-network configuration."""
        return LocalConfig(
            N=self.N,
            K=K,
            J_ee=self.J_ee,
            J_ei=self.J_ei,
            J_ie=self.J_ie,
            J_ii=self.J_ii,
            sigma_e=self.sigma_e,
            sigma_i=self.sigma_i,
            u_e=self.u_e,
            u_i=self.u_i,
            seed=seed,
            device=self.device,
            dt=self.dt,
            init_steps=self.init_steps,
            record_steps=self.record_steps,
            steps_per_record=self.steps_per_record,
            tau_e=self.tau_e,
            tau_i=self.tau_i,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/figures/figure1_spontaneous.yaml"),
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def save_panel(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
) -> tuple[Path, Path]:
    """Save one panel as vector and raster files."""
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_path = output_directory / f"{stem}.pdf"
    png_path = output_directory / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    return pdf_path, png_path


def _threshold_time(axis: np.ndarray, curve: np.ndarray) -> float:
    """Return the first axis value where a correlation reaches exp(-1)."""
    indices = np.flatnonzero(np.isfinite(curve) & (curve <= np.exp(-1.0)))
    return float(axis[indices[0]]) if indices.size else float(axis[-1])


def panel_indices(K_values: np.ndarray, panel_K_values: np.ndarray) -> np.ndarray:
    """Return scan indices for the selected panel conditions."""
    return np.asarray(
        [int(np.flatnonzero(np.isclose(K_values, value))[0]) for value in panel_K_values]
    )


def run_analysis(config: Figure1Config, data_directory: Path) -> dict[str, np.ndarray]:
    """Run all matched simulations and return panel data."""
    data_directory.mkdir(parents=True, exist_ok=True)
    sample_interval = config.dt * config.steps_per_record
    max_lag_samples = round(config.max_lag_seconds / sample_interval)
    K_count = len(config.K_values)
    seed_count = len(config.seeds)
    dimensions = np.empty((K_count, seed_count))
    spatial_lengths = np.empty((K_count, seed_count))
    temporal_times = np.empty((K_count, seed_count))
    balances = np.empty((K_count, seed_count, 7))
    examples_e = []
    examples_i = []
    spatial_curves = []
    temporal_curves = []
    spatial_powers = []
    temporal_powers = []
    pca_variances = []

    for K_index, K in enumerate(config.K_values):
        seed_spatial = []
        seed_temporal = []
        seed_spatial_power = []
        seed_temporal_power = []
        seed_pca_variance = []
        for seed_index, seed in enumerate(config.seeds):
            print(f"Run K={K:g}, seed={seed}", flush=True)
            local_config = config.local_config(K, seed)
            result = simulate_local(local_config)
            raw_path = data_directory / f"activity_K{K:g}_seed{seed}.npz"
            save_npz(
                raw_path,
                {
                    "excitatory": result.excitatory.astype(np.float32),
                    "inhibitory": result.inhibitory.astype(np.float32),
                    "time": result.time,
                },
            )
            if seed_index == 0:
                examples_e.append(result.excitatory)
                examples_i.append(result.inhibitory)

            dimensions[K_index, seed_index] = participation_dimension(result.excitatory)
            seed_pca_variance.append(pca_variance_curve(result.excitatory))
            distance, spatial_curve = radial_spatial_correlation(result.excitatory)
            lag, temporal_curve = temporal_autocorrelation(
                result.excitatory,
                sample_interval,
                max_lag_samples,
            )
            wave_number, spatial_power = radial_spatial_spectrum(result.excitatory)
            frequency, temporal_power = temporal_spectrum(
                result.excitatory,
                sample_interval,
                config.spectrum_window_samples,
            )
            balance = current_balance(result.excitatory, result.inhibitory, local_config)
            dimensions[K_index, seed_index] /= config.N**2
            spatial_lengths[K_index, seed_index] = _threshold_time(distance, spatial_curve)
            temporal_times[K_index, seed_index] = _threshold_time(lag, temporal_curve)
            balances[K_index, seed_index] = [
                balance["external"],
                balance["excitatory"],
                balance["inhibitory"],
                balance["net"],
                balance["cancellation"],
                balance["active_cancellation"],
                balance["inactive_cancellation"],
            ]
            seed_spatial.append(spatial_curve)
            seed_temporal.append(temporal_curve)
            seed_spatial_power.append(spatial_power)
            seed_temporal_power.append(temporal_power)
        spatial_curves.append(seed_spatial)
        temporal_curves.append(seed_temporal)
        spatial_powers.append(seed_spatial_power)
        temporal_powers.append(seed_temporal_power)
        pca_variances.append(seed_pca_variance)

    return {
        "K_values": np.asarray(config.K_values),
        "panel_K_values": np.asarray(config.panel_K_values),
        "tau_e": np.asarray(config.tau_e),
        "tau_i": np.asarray(config.tau_i),
        "seeds": np.asarray(config.seeds),
        "time": result.time,
        "examples_e": np.asarray(examples_e),
        "examples_i": np.asarray(examples_i),
        "dimension": dimensions,
        "pca_variance": np.asarray(pca_variances),
        "spatial_distance": distance,
        "spatial_correlation": np.asarray(spatial_curves),
        "spatial_correlation_length": spatial_lengths,
        "temporal_lag": lag,
        "temporal_correlation": np.asarray(temporal_curves),
        "temporal_correlation_time": temporal_times,
        "wave_number": wave_number,
        "spatial_power": np.asarray(spatial_powers),
        "frequency": frequency,
        "temporal_power": np.asarray(temporal_powers),
        "balance": balances,
    }


def _mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return seed mean and standard deviation."""
    return np.nanmean(values, axis=1), np.nanstd(values, axis=1)


def _draw_model_schematic(axis: plt.Axes, tau_e: float, tau_i: float) -> None:
    """Draw the two-sheet local E/I model."""
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    excitatory_color = "#d95f4f"
    inhibitory_color = "#3977b7"
    sheet_color = "#dceff5"
    edge_color = "#7ca6b5"
    top = np.array([[0.12, 0.57], [0.72, 0.57], [0.91, 0.75], [0.31, 0.75]])
    bottom = np.array([[0.12, 0.18], [0.72, 0.18], [0.91, 0.36], [0.31, 0.36]])
    axis.add_patch(Polygon(bottom, facecolor=sheet_color, edgecolor=edge_color))
    axis.add_patch(Polygon(top, facecolor=sheet_color, edgecolor=edge_color))
    axis.text(0.05, 0.66, "E", color=excitatory_color, fontsize=15, fontweight="bold")
    axis.text(0.05, 0.27, "I", color=inhibitory_color, fontsize=15, fontweight="bold")
    axis.text(0.17, 0.60, rf"$\tau_E={tau_e:g}$", fontsize=9)
    axis.text(0.17, 0.21, rf"$\tau_I={tau_i:g}$", fontsize=9)

    e_point = (0.52, 0.65)
    i_point = (0.52, 0.26)
    axis.scatter(*e_point, s=40, color=excitatory_color, zorder=5)
    axis.scatter(*i_point, s=40, color=inhibitory_color, zorder=5)
    profile_x = np.linspace(0.31, 0.73, 100)
    e_profile = 0.76 + 0.11 * np.exp(-0.5 * ((profile_x - 0.52) / 0.07) ** 2)
    i_profile = 0.37 + 0.10 * np.exp(-0.5 * ((profile_x - 0.52) / 0.10) ** 2)
    axis.plot(profile_x, e_profile, color=excitatory_color, linewidth=1.8)
    axis.plot(profile_x, i_profile, color=inhibitory_color, linewidth=1.8)
    axis.text(0.73, 0.82, r"$\sigma_E$", color=excitatory_color, fontsize=9)
    axis.text(0.73, 0.40, r"$\sigma_I$", color=inhibitory_color, fontsize=9)

    arrow_style = "Simple,tail_width=0.5,head_width=5,head_length=6"
    axis.add_patch(
        FancyArrowPatch(
            (0.47, 0.62),
            (0.47, 0.30),
            connectionstyle="arc3,rad=0.22",
            arrowstyle=arrow_style,
            color=excitatory_color,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.57, 0.30),
            (0.57, 0.61),
            connectionstyle="arc3,rad=0.22",
            arrowstyle=arrow_style,
            color=inhibitory_color,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.50, 0.68),
            (0.59, 0.67),
            connectionstyle="arc3,rad=-1.2",
            arrowstyle=arrow_style,
            color=excitatory_color,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.50, 0.24),
            (0.59, 0.25),
            connectionstyle="arc3,rad=1.2",
            arrowstyle=arrow_style,
            color=inhibitory_color,
        )
    )
    axis.annotate(
        r"$u_E=10$",
        xy=(0.30, 0.67),
        xytext=(0.02, 0.87),
        arrowprops={"arrowstyle": "->", "color": "0.25"},
        fontsize=9,
    )
    axis.text(0.02, 0.11, r"$u_I=0$", fontsize=9)
    axis.text(0.50, 0.02, r"Local coupling strength $\propto\sqrt{K}$", ha="center", fontsize=9)


def plot_cancellation(
    axis: plt.Axes,
    K_values: np.ndarray,
    mean: np.ndarray,
    spread: np.ndarray,
) -> list[plt.Line2D]:
    """Plot visible all-site, active-site, and inactive-site curves."""
    styles = (
        {
            "label": "All sites",
            "color": "tab:blue",
            "linestyle": "--",
            "markerfacecolor": "white",
            "zorder": 5,
        },
        {
            "label": "Active sites",
            "color": "tab:orange",
            "linestyle": ":",
            "markerfacecolor": "tab:orange",
            "zorder": 4,
        },
        {
            "label": "Inactive sites",
            "color": "tab:green",
            "linestyle": "-.",
            "markerfacecolor": "tab:green",
            "zorder": 3,
        },
    )
    lines = []
    for index, style in enumerate(styles):
        container = axis.errorbar(
            K_values,
            mean[:, index],
            yerr=spread[:, index],
            marker="o",
            markersize=6,
            markeredgewidth=1.5,
            capsize=3,
            linewidth=1.7,
            **style,
        )
        line = container.lines[0]
        line.set_label(style["label"])
        lines.append(line)
    axis.set(
        xscale="log",
        xlabel="K",
        ylabel="Local cancellation ratio",
        ylim=(0.0, 1.0),
    )
    axis.legend(handles=lines, frameon=False, fontsize=8)
    return lines


def plot_panels(arrays: dict[str, np.ndarray]) -> dict[str, plt.Figure]:
    """Return separate draft figures for panels B--G."""
    K_values = arrays["K_values"]
    panel_K_values = arrays.get("panel_K_values", K_values)
    selected = panel_indices(K_values, panel_K_values)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(panel_K_values)))
    figures = {}

    figure = plt.figure(figsize=(12, 3.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 4, width_ratios=[1.45, 1.0, 1.0, 1.0])
    schematic_axis = figure.add_subplot(grid[0, 0])
    _draw_model_schematic(
        schematic_axis,
        float(arrays.get("tau_e", 0.01)),
        float(arrays.get("tau_i", 0.01)),
    )
    image = None
    for panel_index, (scan_index, K, color) in enumerate(zip(selected, panel_K_values, colors)):
        axis = figure.add_subplot(grid[0, panel_index + 1])
        frame = arrays["examples_e"][scan_index, :, :, -1]
        scale = np.percentile(frame, 99.0)
        normalized = np.clip(frame / max(scale, np.finfo(float).eps), 0.0, 1.0)
        image = axis.imshow(normalized, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set(title=f"K={K:g}", xticks=[], yticks=[])
        axis.set_xlabel("Relative E rate")
    if image is not None:
        figure.colorbar(image, ax=figure.axes[1:], fraction=0.025, pad=0.02)
    figures["panel_A_model_patterns"] = figure

    figure, axes = plt.subplots(1, len(panel_K_values), figsize=(12, 3.2), sharex=True)
    center = arrays["examples_e"].shape[1] // 2
    for axis, scan_index, K, color in zip(axes, selected, panel_K_values, colors):
        excitatory = arrays["examples_e"][scan_index]
        inhibitory = arrays["examples_i"][scan_index]
        axis.plot(arrays["time"], excitatory.mean(axis=(0, 1)), color=color, label="E mean")
        axis.plot(arrays["time"], inhibitory.mean(axis=(0, 1)), color="tab:orange", label="I mean")
        axis.plot(
            arrays["time"],
            excitatory[center, center],
            color="0.25",
            alpha=0.55,
            linewidth=0.8,
            label="E site",
        )
        axis.set(title=f"K={K:g}", xlabel="Time (s)")
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figures["panel_B_traces"] = figure

    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    pc_rank = np.arange(1, arrays["pca_variance"].shape[-1] + 1)
    plot_count = min(250, pc_rank.size)
    spatial_size = arrays["examples_e"].shape[1] * arrays["examples_e"].shape[2]
    dimension_mean = np.nanmean(arrays["dimension"], axis=1) * spatial_size
    for scan_index, K, color in zip(selected, panel_K_values, colors):
        variance = np.nanmean(arrays["pca_variance"][scan_index], axis=0)
        axis.plot(
            pc_rank[:plot_count],
            variance[:plot_count],
            color=color,
            label=rf"$K={K:g}$, $D_{{PR}}={dimension_mean[scan_index]:.0f}$",
        )
    axis.set(
        xlabel="PC rank",
        ylabel="Variance fraction",
        yscale="log",
        xlim=(1, plot_count),
    )
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figures["panel_C_dimension"] = figure

    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    for scan_index, K, color in zip(selected, panel_K_values, colors):
        mean, spread = _mean_and_std(arrays["spatial_correlation"][scan_index : scan_index + 1])
        axis.plot(arrays["spatial_distance"], mean[0], color=color, label=f"K={K:g}")
        axis.fill_between(arrays["spatial_distance"], mean[0] - spread[0], mean[0] + spread[0], color=color, alpha=0.18)
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.set(xlabel="Distance", ylabel="Spatial correlation")
    axis.legend(frameon=False)
    figure.tight_layout()
    figures["panel_D_spatial_correlation"] = figure

    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    for scan_index, K, color in zip(selected, panel_K_values, colors):
        mean, spread = _mean_and_std(arrays["temporal_correlation"][scan_index : scan_index + 1])
        axis.plot(arrays["temporal_lag"], mean[0], color=color, label=f"K={K:g}")
        axis.fill_between(arrays["temporal_lag"], mean[0] - spread[0], mean[0] + spread[0], color=color, alpha=0.18)
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.set(xlabel="Lag (s)", ylabel="Temporal correlation")
    axis.legend(frameon=False)
    figure.tight_layout()
    figures["panel_E_temporal_correlation"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for scan_index, K, color in zip(selected, panel_K_values, colors):
        spatial_mean, _ = _mean_and_std(arrays["spatial_power"][scan_index : scan_index + 1])
        temporal_mean, _ = _mean_and_std(arrays["temporal_power"][scan_index : scan_index + 1])
        axes[0].plot(arrays["wave_number"], spatial_mean[0], color=color, label=f"K={K:g}")
        axes[1].plot(arrays["frequency"], temporal_mean[0], color=color, label=f"K={K:g}")
    axes[0].set(xlabel="Wave number k", ylabel="Normalized P(k)", yscale="log")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="Normalized P(f)", xscale="log", yscale="log")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figures["panel_F_spectra"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    balance_mean = np.nanmean(arrays["balance"], axis=1)
    balance_std = np.nanstd(arrays["balance"], axis=1)
    positions = np.arange(len(K_values))
    width = 0.19
    labels = ("External", "Recurrent E", "Recurrent I", "Net")
    for component, label in enumerate(labels):
        axes[0].bar(
            positions + (component - 1.5) * width,
            balance_mean[:, component],
            width,
            yerr=balance_std[:, component],
            label=label,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(xticks=positions, xticklabels=[f"{K:g}" for K in K_values], xlabel="K", ylabel="Mean current / $\\sqrt{K}$")
    axes[0].legend(frameon=False, fontsize=8)
    plot_cancellation(
        axes[1],
        K_values,
        balance_mean[:, 4:7],
        balance_std[:, 4:7],
    )
    figure.tight_layout()
    figures["panel_G_balance"] = figure
    return figures


def metric_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, float | int]]:
    """Return one summary row for each K and seed."""
    rows = []
    for K_index, K in enumerate(arrays["K_values"]):
        for seed_index, seed in enumerate(arrays["seeds"]):
            rows.append(
                {
                    "K": K,
                    "seed": int(seed),
                    "dimension_fraction": arrays["dimension"][K_index, seed_index],
                    "spatial_correlation_length": arrays["spatial_correlation_length"][K_index, seed_index],
                    "temporal_correlation_time": arrays["temporal_correlation_time"][K_index, seed_index],
                    "external_current": arrays["balance"][K_index, seed_index, 0],
                    "excitatory_current": arrays["balance"][K_index, seed_index, 1],
                    "inhibitory_current": arrays["balance"][K_index, seed_index, 2],
                    "net_current": arrays["balance"][K_index, seed_index, 3],
                    "cancellation_ratio": arrays["balance"][K_index, seed_index, 4],
                    "active_cancellation_ratio": arrays["balance"][K_index, seed_index, 5],
                    "inactive_cancellation_ratio": arrays["balance"][K_index, seed_index, 6],
                }
            )
    return rows


def main() -> None:
    """Run simulations, save data, and create separate panel files."""
    args = parse_args()
    config, document = load_dataclass_sections(
        args.config,
        Figure1Config,
        tuple(FIGURE1_SECTIONS),
    )
    output_directory = args.output_directory or Path(
        document.get("run", {}).get("output_directory", "output/figures/figure1")
    )
    if not output_directory.is_absolute():
        output_directory = repo_root / output_directory
    data_directory = output_directory / "data"
    arrays = run_analysis(config, data_directory)
    save_npz(data_directory / "results.npz", arrays)
    save_yaml(
        data_directory / "config.yaml",
        {
            "run": {"output_directory": str(output_directory)},
            **dataclass_to_sections(config, FIGURE1_SECTIONS),
        },
    )
    save_json(data_directory / "metadata.json", runtime_metadata(repo_root))
    rows = metric_rows(arrays)
    save_csv(data_directory / "metrics.csv", list(rows[0]), rows)
    figures = plot_panels(arrays)
    for stem, figure in figures.items():
        save_panel(figure, output_directory, stem)
    print(f"Saved Figure 1 panels to {output_directory}")
    if args.show:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
