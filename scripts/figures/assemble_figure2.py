"""Assemble Figure 2 from saved task arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.figures.figure2_tasks import plot_assembled_figure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-directory",
        type=Path,
        default=Path("output/figures/figure2_tau_i_002_memory_calibrated"),
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def assemble_figure(
    panel_directory: Path,
    output_directory: Path | None = None,
) -> tuple[Path, Path]:
    """Assemble Figure 2 without a simulation run."""
    panel_directory = Path(panel_directory)
    data_path = panel_directory / "data" / "results.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Saved task arrays do not exist: {data_path}")
    with np.load(data_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}

    output_directory = Path(output_directory or panel_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = plot_assembled_figure(arrays)
    png_path = output_directory / "figure2.png"
    pdf_path = output_directory / "figure2.pdf"
    figure.savefig(png_path, dpi=220, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    """Assemble and save Figure 2."""
    args = parse_args()
    paths = assemble_figure(args.panel_directory, args.output_directory)
    print(f"Saved assembled Figure 2 to {paths[0].parent}")
    if args.show:
        image = plt.imread(paths[0])
        figure, axis = plt.subplots(figsize=(12, 9))
        axis.imshow(image)
        axis.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
