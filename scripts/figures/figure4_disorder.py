"""Create Figure 4 from a saved K-rho_F mode scan."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.config import load_yaml


PANEL_STEMS = (
    "panel_A_schematic",
    "panel_B_patterns",
    "panel_C_mode_advantage",
    "panel_D_nonlocal_alignment",
    "panel_E_active_fraction",
    "panel_F_susceptibility",
)
RHO_EXAMPLES = (0.0, 0.5, 8.0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=Path("output/scans/K_rhoF_modes/tau_i_002"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/figures/figure4"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_saved_results(source_directory: Path) -> tuple[dict[float, dict[str, np.ndarray]], int]:
    """Load one saved K-rho_F scan."""
    source_directory = Path(source_directory)
    results_path = source_directory / "results.npz"
    config_path = source_directory / "config.yaml"
    if not results_path.exists():
        raise FileNotFoundError(f"Saved scan does not exist: {results_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Saved configuration does not exist: {config_path}")

    document = load_yaml(config_path)
    N = int(document["network"]["N"])
    results: dict[float, dict[str, np.ndarray]] = {}
    pattern = re.compile(r"^K_([^_]+)_(.+)$")
    with np.load(results_path, allow_pickle=False) as saved:
        for name in saved.files:
            match = pattern.match(name)
            if match is None:
                continue
            K = float(match.group(1))
            results.setdefault(K, {})[match.group(2)] = saved[name]
    if not results:
        raise ValueError("Saved scan does not contain K-indexed arrays.")
    return dict(sorted(results.items())), N


def _nearest_indices(values: np.ndarray, targets: tuple[float, ...]) -> list[int]:
    """Return the nearest array index for each target."""
    values = np.asarray(values, dtype=float)
    return [int(np.nanargmin(np.abs(values - target))) for target in targets]


def _schematic(axis: plt.Axes) -> None:
    """Draw local recurrence and a low-rank loop on a 2D sheet."""
    axis.axis("off")
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.text(0.5, 0.97, r"$J_{eff}=J_{local}+\rho_FUV^T$", ha="center", va="top", fontsize=15)
    excitatory_color = "#d95f4f"
    inhibitory_color = "#3977b7"
    sheet_color = "#e8f2f5"
    edge_color = "#7898a3"
    top = np.array([[0.08, 0.52], [0.51, 0.52], [0.66, 0.66], [0.23, 0.66]])
    bottom = np.array([[0.08, 0.18], [0.51, 0.18], [0.66, 0.32], [0.23, 0.32]])
    axis.add_patch(Polygon(top, facecolor=sheet_color, edgecolor=edge_color, linewidth=1.4))
    axis.add_patch(Polygon(bottom, facecolor=sheet_color, edgecolor=edge_color, linewidth=1.4))
    axis.text(0.04, 0.58, "E", color=excitatory_color, fontsize=14, fontweight="bold")
    axis.text(0.04, 0.24, "I", color=inhibitory_color, fontsize=14, fontweight="bold")

    profile_x = np.linspace(0.18, 0.55, 100)
    e_profile = 0.67 + 0.10 * np.exp(-0.5 * ((profile_x - 0.37) / 0.055) ** 2)
    i_profile = 0.33 + 0.09 * np.exp(-0.5 * ((profile_x - 0.37) / 0.085) ** 2)
    axis.plot(profile_x, e_profile, color=excitatory_color, linewidth=2.0)
    axis.plot(profile_x, i_profile, color=inhibitory_color, linewidth=2.0)
    axis.scatter([0.37], [0.59], color=[excitatory_color], s=30, zorder=5)
    axis.text(0.53, 0.75, r"$G_E(\Delta x)$", color=excitatory_color, fontsize=9)
    axis.text(0.53, 0.39, r"$G_I(\Delta x)$", color=inhibitory_color, fontsize=9)
    axis.annotate(
        "local Gaussian coupling",
        xy=(0.42, 0.71),
        xytext=(0.18, 0.86),
        ha="center",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "0.3"},
    )
    latent_box = dict(boxstyle="round,pad=0.4", facecolor="#f3ecfa", edgecolor="#6a3d9a")
    axis.text(0.82, 0.58, "rank-r\nlatent", ha="center", va="center", fontsize=11, bbox=latent_box)
    axis.annotate(
        r"readout $V^T$",
        xy=(0.76, 0.61),
        xytext=(0.61, 0.70),
        ha="center",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#6a3d9a", "linewidth": 1.5},
    )
    axis.annotate(
        r"feedback $\rho_FU$",
        xy=(0.56, 0.54),
        xytext=(0.77, 0.43),
        ha="center",
        va="center",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#6a3d9a", "linewidth": 1.5},
    )
    axis.text(0.34, 0.08, "2D excitatory and inhibitory sheets", ha="center", fontsize=10)


def _plot_schematic() -> plt.Figure:
    """Plot the local-plus-low-rank network schematic."""
    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    _schematic(axis)
    figure.tight_layout()
    return figure


def _plot_patterns(results: dict[float, dict[str, np.ndarray]], N: int) -> plt.Figure:
    """Plot activity maps for K=100 and K=10000."""
    K_values = sorted(results)
    selected_K = tuple(min(K_values, key=lambda value: abs(np.log(value / target))) for target in (100.0, 10000.0))
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for row, K in enumerate(selected_K):
        result = results[K]
        indices = _nearest_indices(result["relative_strengths"], RHO_EXAMPLES)
        time_index = result["example_rates"].shape[-1] // 2
        maps = [result["example_rates"][index, :, time_index].reshape(N, N) for index in indices]
        lower = min(float(np.nanmin(image)) for image in maps)
        upper = max(float(np.nanmax(image)) for image in maps)
        for column, (index, image) in enumerate(zip(indices, maps)):
            axis = axes[row, column]
            axis.imshow(image, cmap="magma", vmin=lower, vmax=upper, interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            if row == 0:
                rho = float(result["relative_strengths"][index])
                axis.set_title(rf"$\rho_F={rho:g}$", fontsize=11)
            if column == 0:
                axis.set_ylabel(rf"$K={K:g}$", fontsize=12)
    figure.suptitle("Instantaneous activity under low-rank perturbation", fontsize=13)
    figure.tight_layout()
    return figure


def _plot_mode_advantage(results: dict[float, dict[str, np.ndarray]]) -> plt.Figure:
    """Plot full-network mode advantage across K and rho_F."""
    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.90, len(results)))
    for (K, result), color in zip(results.items(), colors):
        x = result["relative_strengths"]
        y = result["transition_index"]
        spread = result["transition_std"]
        axis.plot(x, y, "o-", color=color, label=f"K={K:g}")
        axis.fill_between(x, y - spread, y + spread, color=color, alpha=0.15)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel=r"Relative perturbation strength, $\rho_F$",
        ylabel="Mean gain in captured variance",
        title="Nonlocal modes reshape activity less at large K",
    )
    _format_rho_axis(axis)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    return figure


def _plot_nonlocal_alignment(results: dict[float, dict[str, np.ndarray]]) -> plt.Figure:
    """Plot matched and null nonlocal-subspace alignment."""
    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.90, len(results)))
    for (K, result), color in zip(results.items(), colors):
        x = result["relative_strengths"]
        matched = result["nonlocal_fraction"]
        spread = result["nonlocal_fraction_std"]
        axis.plot(x, matched, "o-", color=color, label=f"Matched, K={K:g}")
        axis.fill_between(x, matched - spread, matched + spread, color=color, alpha=0.15)
        axis.plot(x, result["null_fraction"], "--", color=color, label=f"Null, K={K:g}")
    axis.set(
        xlabel=r"Relative perturbation strength, $\rho_F$",
        ylabel="Variance in rank-r subspace",
        ylim=(0.0, 1.02),
        title="Nonlocal activity recruitment decreases with K",
    )
    _format_rho_axis(axis)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    return figure


def _plot_susceptibility(results: dict[float, dict[str, np.ndarray]]) -> plt.Figure:
    """Plot activity gain per realized low-rank recurrent drive."""
    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.90, len(results)))
    for (K, result), color in zip(results.items(), colors):
        rho = np.asarray(result["relative_strengths"], dtype=float)
        drive = np.asarray(result["lowrank_output_variance"], dtype=float)
        response = np.asarray(result["lowrank_power"], dtype=float)
        gain = np.divide(
            response,
            drive,
            out=np.full_like(response, np.nan),
            where=drive > np.finfo(float).eps,
        )
        use = (rho > 0.0) & np.isfinite(gain) & (gain > 0.0)
        axis.plot(rho[use], gain[use], "o-", color=color, label=f"K={K:g}")
    axis.set(
        xlabel=r"Relative perturbation strength, $\rho_F$",
        ylabel=r"Low-rank susceptibility, $P_M/P_{out}$",
        yscale="log",
        title="ReLU gating reduces low-rank susceptibility",
    )
    _format_rho_axis(axis)
    axis.legend(frameon=False, fontsize=10, ncol=2)
    figure.tight_layout()
    return figure


def _format_rho_axis(axis: plt.Axes) -> None:
    """Show perturbation strength on a nonnegative linear axis."""
    ticks = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0])
    axis.set_xscale("linear")
    axis.set_xlim(0.0, 8.4)
    axis.set_xticks(ticks)


def _plot_active_fraction(results: dict[float, dict[str, np.ndarray]]) -> plt.Figure:
    """Plot excitatory and inhibitory active fractions across K and rho_F."""
    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.90, len(results)))
    for (K, result), color in zip(results.items(), colors):
        rho = result["relative_strengths"]
        axis.plot(rho, result["excitatory_active_fraction"], "o-", color=color, label=f"E, K={K:g}")
        axis.plot(rho, result["inhibitory_active_fraction"], "--", color=color, label=f"I, K={K:g}")
    axis.set(
        xlabel=r"Relative perturbation strength, $\rho_F$",
        ylabel="Active-site fraction",
        ylim=(0.0, 1.02),
        title="Large K closes the active ReLU gate",
    )
    _format_rho_axis(axis)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    return figure


def plot_panels(
    results: dict[float, dict[str, np.ndarray]],
    N: int,
) -> dict[str, plt.Figure]:
    """Return the six approved Figure 4 panels."""
    return {
        "panel_A_schematic": _plot_schematic(),
        "panel_B_patterns": _plot_patterns(results, N),
        "panel_C_mode_advantage": _plot_mode_advantage(results),
        "panel_D_nonlocal_alignment": _plot_nonlocal_alignment(results),
        "panel_E_active_fraction": _plot_active_fraction(results),
        "panel_F_susceptibility": _plot_susceptibility(results),
    }


def _trim_white(image: np.ndarray, padding: int = 8) -> np.ndarray:
    """Remove white margins from one panel image."""
    content = np.any(image[..., :3] < 0.995, axis=2)
    rows = np.flatnonzero(np.any(content, axis=1))
    columns = np.flatnonzero(np.any(content, axis=0))
    if not rows.size or not columns.size:
        return image
    return image[
        max(0, int(rows[0]) - padding) : min(image.shape[0], int(rows[-1]) + padding + 1),
        max(0, int(columns[0]) - padding) : min(image.shape[1], int(columns[-1]) + padding + 1),
    ]


def _place_panel(axis: plt.Axes, image: np.ndarray, label: str) -> None:
    """Place one panel image and its figure letter."""
    axis.imshow(image)
    axis.axis("off")
    axis.text(
        -0.015,
        0.99,
        label,
        transform=axis.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="right",
        va="top",
    )


def save_outputs(panels: dict[str, plt.Figure], output_directory: Path) -> tuple[Path, ...]:
    """Save separate panels and one assembled Figure 4."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    images: dict[str, np.ndarray] = {}
    for stem in PANEL_STEMS:
        figure = panels[stem]
        png_path = output_directory / f"{stem}.png"
        pdf_path = output_directory / f"{stem}.pdf"
        figure.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
        figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        paths.extend((png_path, pdf_path))
        images[stem] = _trim_white(mpimg.imread(png_path))

    figure = plt.figure(figsize=(14, 15.2), facecolor="white")
    grid = figure.add_gridspec(3, 2, height_ratios=(0.95, 1.0, 1.0), hspace=0.05, wspace=0.04)
    placements = tuple(
        (stem, chr(ord("A") + index), grid[index // 2, index % 2])
        for index, stem in enumerate(PANEL_STEMS)
    )
    for stem, label, slot in placements:
        _place_panel(figure.add_subplot(slot), images[stem], label)
    figure.subplots_adjust(left=0.035, right=0.99, top=0.99, bottom=0.02)
    figure_png = output_directory / "figure4.png"
    figure_pdf = output_directory / "figure4.pdf"
    figure.savefig(figure_png, dpi=180, facecolor="white")
    figure.savefig(figure_pdf, dpi=180, facecolor="white")
    plt.close(figure)
    paths.extend((figure_png, figure_pdf))
    return tuple(paths)


def main() -> None:
    """Create Figure 4 from saved arrays."""
    args = parse_args()
    results, N = load_saved_results(args.source_directory)
    panels = plot_panels(results, N)
    paths = save_outputs(panels, args.output_directory)
    for figure in panels.values():
        plt.close(figure)
    print(f"Saved Figure 4 panels and assembly to {args.output_directory}")
    if args.show:
        image = mpimg.imread(args.output_directory / "figure4.png")
        figure, axis = plt.subplots(figsize=(10, 10))
        axis.imshow(image)
        axis.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
