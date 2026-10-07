"""Tests for Figure 2 assembly from saved arrays."""

import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


def sample_arrays() -> dict[str, np.ndarray]:
    """Return small saved arrays for an assembly test."""
    return {
        "K_values": np.array([0.1, 1.0, 10.0]),
        "rigid_r2": np.array([[0.2], [0.9], [-0.1]]),
        "rigid_time": np.arange(4) * 0.1,
        "rigid_target": np.tile(np.array([0.0, 0.5, 1.0, 0.5]), (3, 1)),
        "rigid_prediction": np.array(
            [[0.0, 0.4, 0.8, 0.4], [0.0, 0.5, 1.0, 0.5], [0.2, -0.1, 0.1, 0.0]]
        ),
        "dot_peak_height": np.array([[0.2], [0.8], [0.1]]),
        "dot_peak_delay_seconds": np.array([[0.003], [0.001], [0.002]]),
        "dot_lag_seconds": np.array([-0.1, 0.0, 0.1]),
        "dot_max_lag_seconds": np.array(0.1),
        "dot_cross_correlation": np.array(
            [[0.1, 0.2, 0.1], [0.3, 0.8, 0.2], [0.0, 0.1, 0.0]]
        ),
        "memory_r2": np.array([[0.0], [0.8], [-0.5]]),
        "memory_training_r2": np.array([[0.4], [0.95], [0.7]]),
        "memory_time": np.arange(6) * 0.1,
        "memory_output": np.array(
            [[0.0, 0.0, 0.1, 0.2, 0.4, 0.6], [0.0, 0.0, 0.1, 0.5, 0.9, 1.0], [0.0, 0.0, 0.0, -0.1, 0.0, 0.1]]
        ),
        "memory_target": np.tile(
            np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0]), (3, 1)
        ),
        "memory_cue_steps": np.array(1),
        "memory_delay_steps": np.array(2),
        "memory_output_window_steps": np.array(1),
        "memory_output_start_step": np.array(4),
    }


class Figure2AssemblyTests(unittest.TestCase):
    """Check assembly from saved data without simulation."""

    def test_assemble_figure_writes_png_and_pdf_from_saved_arrays(self) -> None:
        module_name = "scripts.figures.assemble_figure2"
        self.assertIsNotNone(importlib.util.find_spec(module_name))
        module = importlib.import_module(module_name)

        with tempfile.TemporaryDirectory() as directory:
            panel_directory = Path(directory)
            data_directory = panel_directory / "data"
            data_directory.mkdir()
            np.savez(data_directory / "results.npz", **sample_arrays())

            paths = module.assemble_figure(panel_directory)

            self.assertEqual({path.name for path in paths}, {"figure2.png", "figure2.pdf"})
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()
