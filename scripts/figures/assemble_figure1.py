"""Assemble Figure 1 from the approved panel PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


PANEL_STEMS = {
    "A": "panel_A_model_patterns",
    "B": "panel_B_traces",
    "C": "panel_C_dimension",
    "D": "panel_D_spatial_correlation",
    "E": "panel_E_temporal_correlation",
    "F": "panel_F_spectra",
    "G": "panel_G_balance",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-directory",
        type=Path,
        default=Path("output/figures/figure1"),
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _trim_white(image: np.ndarray, padding: int = 8) -> np.ndarray:
    """Remove uniform white margins from one panel image."""
    rgb = image[..., :3]
    content = np.any(rgb < 0.995, axis=2)
    rows = np.flatnonzero(np.any(content, axis=1))
    columns = np.flatnonzero(np.any(content, axis=0))
    if not rows.size or not columns.size:
        return image
    row_start = max(0, int(rows[0]) - padding)
    row_stop = min(image.shape[0], int(rows[-1]) + padding + 1)
    column_start = max(0, int(columns[0]) - padding)
    column_stop = min(image.shape[1], int(columns[-1]) + padding + 1)
    return image[row_start:row_stop, column_start:column_stop]


def _load_panels(panel_directory: Path) -> dict[str, np.ndarray]:
    """Load and trim the seven approved panel images."""
    panels = {}
    for label, stem in PANEL_STEMS.items():
        path = panel_directory / f"{stem}.png"
        if not path.exists():
            raise FileNotFoundError(f"Panel image does not exist: {path}")
        panels[label] = _trim_white(mpimg.imread(path))
    balance = panels["G"]
    panels["G"] = _trim_white(balance[:, int(0.54 * balance.shape[1]) :])
    return panels


def _place_panel(axis: plt.Axes, image: np.ndarray, label: str) -> None:
    """Place one raster panel and add its figure letter."""
    axis.imshow(image)
    axis.axis("off")
    label_y = 0.99 if label == "A" else 1.015
    label_vertical_alignment = "top" if label == "A" else "bottom"
    axis.text(
        -0.015,
        label_y,
        label,
        transform=axis.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="right",
        va=label_vertical_alignment,
        clip_on=False,
    )


def assemble_figure(
    panel_directory: Path,
    output_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Assemble Figure 1 without rerunning simulations."""
    panel_directory = Path(panel_directory)
    output_directory = Path(output_directory or panel_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    panels = _load_panels(panel_directory)

    figure = plt.figure(figsize=(15, 16), facecolor="white")
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=(4.1, 3.9, 4.0, 4.2),
        hspace=0.08,
        wspace=0.05,
    )
    axes = {
        "A": figure.add_subplot(grid[0, :]),
        "B": figure.add_subplot(grid[1, :]),
        "C": figure.add_subplot(grid[2, 0]),
        "D": figure.add_subplot(grid[2, 1]),
        "E": figure.add_subplot(grid[2, 2]),
        "F": figure.add_subplot(grid[3, :2]),
        "G": figure.add_subplot(grid[3, 2]),
    }
    for label, axis in axes.items():
        _place_panel(axis, panels[label], label)

    figure.subplots_adjust(left=0.025, right=0.99, top=0.99, bottom=0.02)
    png_path = output_directory / "figure1.png"
    pdf_path = output_directory / "figure1.pdf"
    figure.savefig(png_path, dpi=180, facecolor="white")
    figure.savefig(pdf_path, dpi=180, facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    """Assemble and save Figure 1."""
    args = parse_args()
    paths = assemble_figure(args.panel_directory, args.output_directory)
    print(f"Saved assembled Figure 1 to {paths[0].parent}")
    if args.show:
        image = mpimg.imread(paths[0])
        figure, axis = plt.subplots(figsize=(10, 11))
        axis.imshow(image)
        axis.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
