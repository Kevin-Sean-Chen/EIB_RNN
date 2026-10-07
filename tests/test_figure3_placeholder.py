"""Tests for the Figure 3 mechanism placeholder."""

import importlib
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt


class Figure3PlaceholderTests(unittest.TestCase):
    """Check the approved panel labels and saved files."""

    def test_panels_show_approved_axes_legends_and_pending_state(self) -> None:
        module = importlib.import_module("scripts.figures.figure3_placeholder")

        panels = module.build_panels()

        expected = {
            "panel_A_decoding_snr": (
                "A  Decoding mechanism",
                "Signal-to-noise ratio",
                "Decoding R²",
                {"Stimulus SNR", "Decoding performance"},
            ),
            "panel_B_prediction_recurrence": (
                "B  Prediction mechanism",
                "Recurrent contribution (intact - shuffled)",
                "Prediction lead time (s)",
                {"Spatial recurrence", "Prediction performance"},
            ),
            "panel_C_memory_metastability": (
                "C  Memory mechanism",
                "Metastable-state dwell time (s)",
                "Memory R²",
                {"Median dwell time", "Memory performance"},
            ),
        }
        self.assertEqual(set(panels), set(expected))
        for stem, (title, left_label, right_label, legend_labels) in expected.items():
            figure = panels[stem]
            self.assertEqual(len(figure.axes), 2)
            left_axis, right_axis = figure.axes
            self.assertEqual(left_axis.get_title(), title)
            self.assertEqual(left_axis.get_xscale(), "log")
            self.assertEqual(left_axis.get_xlabel(), "Connection scale, K")
            self.assertEqual(left_axis.get_ylabel(), left_label)
            self.assertEqual(right_axis.get_ylabel(), right_label)
            self.assertIn("Analysis pending", [item.get_text() for item in left_axis.texts])
            legend = left_axis.get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual({item.get_text() for item in legend.get_texts()}, legend_labels)

        for figure in panels.values():
            plt.close(figure)

    def test_save_outputs_writes_three_panels_and_one_assembly(self) -> None:
        module = importlib.import_module("scripts.figures.figure3_placeholder")
        panels = module.build_panels()

        with tempfile.TemporaryDirectory() as directory:
            paths = module.save_outputs(panels, Path(directory))

            self.assertEqual(len(paths), 8)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))
        for figure in panels.values():
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
