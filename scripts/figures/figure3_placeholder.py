"""Create the approved Figure 3 mechanism placeholder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PANEL_STEMS = (
    "panel_A_decoding_snr",
    "panel_B_prediction_recurrence",
    "panel_C_memory_metastability",
)
K_VALUES = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])


@dataclass(frozen=True)
class PanelSpec:
    """Define one mechanism placeholder panel."""

    stem: str
    title: str
    left_label: str
    right_label: str
    mechanism_label: str
    performance_label: str
    color: str


PANEL_SPECS = (
    PanelSpec(
        "panel_A_decoding_snr",
        "A  Decoding mechanism",
        "Signal-to-noise ratio",
        "Decoding R²",
        "Stimulus SNR",
        "Decoding performance",
        "#377eb8",
    ),
    PanelSpec(
        "panel_B_prediction_recurrence",
        "B  Prediction mechanism",
        "Recurrent contribution (intact - shuffled)",
        "Prediction lead time (s)",
        "Spatial recurrence",
        "Prediction performance",
        "#e6862f",
    ),
    PanelSpec(
        "panel_C_memory_metastability",
        "C  Memory mechanism",
        "Metastable-state dwell time (s)",
        "Memory R²",
        "Median dwell time",
        "Memory performance",
        "#24935c",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/figures/figure3"),
    )
    return parser.parse_args()


def _build_panel(spec: PanelSpec) -> plt.Figure:
    """Create one labeled panel without artificial data."""
    figure, left_axis = plt.subplots(figsize=(6.2, 4.3))
    right_axis = left_axis.twinx()
    left_axis.set(
        xscale="log",
        xlim=(0.07, 15000.0),
        xlabel="Connection scale, K",
        ylabel=spec.left_label,
        title=spec.title,
    )
    right_axis.set_ylabel(spec.right_label)
    left_axis.set_xticks(K_VALUES, labels=[f"{value:g}" for value in K_VALUES])
    left_axis.set_yticks([])
    right_axis.set_yticks([])
    left_axis.grid(axis="x", color="0.90", linewidth=0.8)
    left_axis.text(
        0.5,
        0.48,
        "Analysis pending",
        transform=left_axis.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="0.48",
    )
    handles = (
        Line2D([], [], color=spec.color, marker="o", linewidth=2.2, label=spec.mechanism_label),
        Line2D([], [], color="0.28", linestyle="--", linewidth=2.0, label=spec.performance_label),
    )
    left_axis.legend(handles=handles, frameon=False, loc="upper center", ncol=1)
    figure.tight_layout()
    return figure


def build_panels() -> dict[str, plt.Figure]:
    """Build the three approved Figure 3 placeholders."""
    return {spec.stem: _build_panel(spec) for spec in PANEL_SPECS}


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


def save_outputs(
    panels: dict[str, plt.Figure],
    output_directory: Path,
) -> tuple[Path, ...]:
    """Save separate panels and one horizontal assembly."""
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    images: list[np.ndarray] = []
    for stem in PANEL_STEMS:
        png_path = output_directory / f"{stem}.png"
        pdf_path = output_directory / f"{stem}.pdf"
        panels[stem].savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
        panels[stem].savefig(pdf_path, bbox_inches="tight", facecolor="white")
        images.append(_trim_white(mpimg.imread(png_path)))
        paths.extend((png_path, pdf_path))

    figure, axes = plt.subplots(1, 3, figsize=(19.5, 4.8), facecolor="white")
    for axis, image in zip(axes, images):
        axis.imshow(image)
        axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.11)
    figure_png = output_directory / "figure3.png"
    figure_pdf = output_directory / "figure3.pdf"
    figure.savefig(figure_png, dpi=200, facecolor="white")
    figure.savefig(figure_pdf, facecolor="white")
    plt.close(figure)
    paths.extend((figure_png, figure_pdf))
    return tuple(paths)


def main() -> None:
    """Create and save the Figure 3 placeholder."""
    arguments = parse_args()
    panels = build_panels()
    paths = save_outputs(panels, arguments.output_directory)
    for figure in panels.values():
        plt.close(figure)
    print("Saved Figure 3 placeholder files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
