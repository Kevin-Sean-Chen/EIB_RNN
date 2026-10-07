"""Tests for Figure 4 low-rank-disorder panels."""

import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np


def synthetic_results() -> dict[float, dict[str, np.ndarray]]:
    """Return a small matched K-rho_F scan."""
    results = {}
    rho = np.array([0.0, 0.5, 8.0])
    mode_counts = np.array([1, 2, 3, 4])
    for offset, K in enumerate((10.0, 100.0, 1000.0, 10000.0)):
        base = 0.05 + 0.01 * offset
        results[K] = {
            "relative_strengths": rho,
            "example_rates": np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5) + offset,
            "local_curves": np.array(
                [[0.1, 0.3, 0.6, 1.0], [0.1, 0.25, 0.55, 1.0], [0.05, 0.2, 0.5, 1.0]]
            ),
            "network_curves": np.array(
                [[0.1, 0.3, 0.6, 1.0], [0.2, 0.5, 0.8, 1.0], [0.3, 0.65, 0.9, 1.0]]
            ),
            "transition_index": np.array([0.0, 0.4 - base, 0.2 - base]),
            "transition_std": np.array([0.0, 0.04, 0.02]),
            "nonlocal_fraction": np.array([0.01, 0.35 - base, 0.15 - base]),
            "nonlocal_fraction_std": np.array([0.002, 0.03, 0.02]),
            "null_fraction": np.array([0.01, 0.012, 0.011]),
            "null_fraction_std": np.array([0.001, 0.001, 0.001]),
            "lowrank_power": np.array([0.0, 0.4 - base, 0.2 - base]),
            "lowrank_output_variance": np.array([0.0, 0.1, 0.1]),
            "excitatory_active_fraction": np.array([0.9, 0.5, 0.2]) / (offset + 1),
            "excitatory_active_fraction_std": np.array([0.02, 0.02, 0.01]),
            "inhibitory_active_fraction": np.array([0.8, 0.4, 0.15]) / (offset + 1),
            "inhibitory_active_fraction_std": np.array([0.02, 0.02, 0.01]),
            "geometric_shell_counts": mode_counts,
        }
    return results


class Figure4DisorderTests(unittest.TestCase):
    """Check the Figure 4 saved-data workflow."""

    def test_default_source_uses_the_tau_i_002_scan(self) -> None:
        module = importlib.import_module("scripts.figures.figure4_disorder")
        with patch.object(sys, "argv", ["figure4_disorder.py"]):
            arguments = module.parse_args()
        self.assertEqual(
            arguments.source_directory,
            Path("output/scans/K_rhoF_modes/tau_i_002"),
        )

    def test_plot_panels_maps_the_six_approved_analyses(self) -> None:
        module_name = "scripts.figures.figure4_disorder"
        self.assertIsNotNone(importlib.util.find_spec(module_name))
        module = importlib.import_module(module_name)

        panels = module.plot_panels(synthetic_results(), N=2)

        self.assertEqual(
            set(panels),
            {
                "panel_A_schematic",
                "panel_B_patterns",
                "panel_C_mode_advantage",
                "panel_D_nonlocal_alignment",
                "panel_E_active_fraction",
                "panel_F_susceptibility",
            },
        )
        schematic_axis = panels["panel_A_schematic"].axes[0]
        self.assertTrue(any(type(patch).__name__ == "Polygon" for patch in schematic_axis.patches))
        gaussian_lines = [line for line in schematic_axis.lines if len(line.get_xdata()) >= 50]
        self.assertGreaterEqual(len(gaussian_lines), 2)
        for line in gaussian_lines[:2]:
            y = np.asarray(line.get_ydata())
            self.assertGreater(np.max(y), max(y[0], y[-1]))
        feedback = next(
            text for text in schematic_axis.texts
            if text.get_text() == r"feedback $\rho_FU$"
        )
        self.assertGreater(feedback.xy[1], 0.5)
        sheet_points = [
            collection.get_offsets()
            for collection in schematic_axis.collections
            if len(collection.get_offsets())
        ]
        self.assertEqual(sum(len(points) for points in sheet_points), 1)
        self.assertGreater(float(sheet_points[0][0, 1]), 0.5)

        pattern_panel = panels["panel_B_patterns"]
        pattern_images = sum(len(axis.images) for axis in pattern_panel.axes)
        self.assertEqual(pattern_images, 6)
        row_labels = {axis.get_ylabel() for axis in pattern_panel.axes}
        self.assertIn("$K=100$", row_labels)
        self.assertIn("$K=10000$", row_labels)
        first_map = next(axis.images[0].get_array() for axis in pattern_panel.axes if axis.images)
        np.testing.assert_array_equal(
            first_map,
            synthetic_results()[100.0]["example_rates"][0, :, 2].reshape(2, 2),
        )
        self.assertEqual(len(panels["panel_C_mode_advantage"].axes[0].lines), 5)
        self.assertEqual(len(panels["panel_D_nonlocal_alignment"].axes[0].lines), 8)
        for stem in (
            "panel_C_mode_advantage",
            "panel_D_nonlocal_alignment",
            "panel_E_active_fraction",
            "panel_F_susceptibility",
        ):
            self.assertEqual(panels[stem].axes[0].get_xscale(), "linear")
            self.assertGreaterEqual(np.min(panels[stem].axes[0].get_xticks()), 0.0)
        active_axis = panels["panel_E_active_fraction"].axes[0]
        self.assertEqual(len(active_axis.lines), 8)
        gain_axis = panels["panel_F_susceptibility"].axes[0]
        self.assertEqual(len(gain_axis.lines), 4)
        self.assertEqual(gain_axis.get_yscale(), "log")
        self.assertTrue(all(np.all(line.get_xdata() > 0) for line in gain_axis.lines))

        for figure in panels.values():
            plt.close(figure)

    def test_top_panel_letter_stays_inside_the_assembled_canvas(self) -> None:
        module = importlib.import_module("scripts.figures.figure4_disorder")
        figure, axis = plt.subplots()

        module._place_panel(axis, np.ones((3, 3, 4)), "A")

        label = axis.texts[-1]
        self.assertLessEqual(label.get_position()[1], 1.0)
        self.assertEqual(label.get_verticalalignment(), "top")
        plt.close(figure)

    def test_save_outputs_writes_panel_files_and_assembled_figure(self) -> None:
        module_name = "scripts.figures.figure4_disorder"
        self.assertIsNotNone(importlib.util.find_spec(module_name))
        module = importlib.import_module(module_name)
        panels = module.plot_panels(synthetic_results(), N=2)

        with tempfile.TemporaryDirectory() as directory:
            paths = module.save_outputs(panels, Path(directory))

            self.assertEqual(len(paths), 14)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))
            self.assertTrue((Path(directory) / "figure4.png").exists())
            self.assertTrue((Path(directory) / "figure4.pdf").exists())
        for figure in panels.values():
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
