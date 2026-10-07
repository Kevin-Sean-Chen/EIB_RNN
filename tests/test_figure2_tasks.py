"""Tests for Figure 2 task panels."""

from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures import figure2_tasks
from scripts.figures.figure2_tasks import (
    circular_trace,
    metric_rows,
    plot_panels,
    save_panel,
    select_best_worst,
)


class Figure2TaskPanelTests(unittest.TestCase):
    """Check task selection and panel output."""

    def test_select_best_worst_maximizes_mean_performance(self) -> None:
        performance = np.array(
            [
                [0.1, 0.3],
                [0.8, 0.6],
                [-0.2, 0.0],
            ]
        )

        best, worst = select_best_worst(performance)

        self.assertEqual(best, 1)
        self.assertEqual(worst, 2)

    def test_select_best_worst_minimizes_delay(self) -> None:
        delay = np.array([[0.003], [0.001], [0.002]])

        best, worst = select_best_worst(delay, higher_is_better=False)

        self.assertEqual(best, 1)
        self.assertEqual(worst, 0)

    def test_circular_trace_breaks_wrap_jumps(self) -> None:
        trace = np.array([2.9, 3.2, 3.5, -3.0])

        wrapped = circular_trace(trace)

        finite_pairs = np.isfinite(wrapped[:-1]) & np.isfinite(wrapped[1:])
        self.assertTrue(np.all(np.abs(np.diff(wrapped)[finite_pairs]) <= np.pi))
        self.assertTrue(np.isnan(wrapped).any())

    def test_plot_panels_uses_best_and_worst_examples(self) -> None:
        arrays = {
            "K_values": np.array([0.1, 1.0, 10.0]),
            "rigid_r2": np.array([[0.2], [0.9], [-0.1]]),
            "rigid_time": np.arange(4) * 0.1,
            "rigid_target": np.tile(np.array([0.0, 0.5, 1.0, 0.5]), (3, 1)),
            "rigid_prediction": np.array(
                [[0.0, 0.4, 0.8, 0.4], [0.0, 0.5, 1.0, 0.5], [0.2, -0.1, 0.1, 0.0]]
            ),
            "dot_peak_height": np.array([[0.2, 0.3], [0.8, 0.7], [0.1, 0.0]]),
            "dot_peak_delay_seconds": np.array(
                [[0.003, 0.003], [0.001, 0.002], [0.002, 0.002]]
            ),
            "dot_lag_seconds": np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
            "dot_max_lag_seconds": np.array(0.1),
            "dot_cross_correlation": np.array(
                [
                    [0.9, 0.1, 0.2, 0.1, 0.8],
                    [0.9, 0.3, 0.8, 0.2, 0.8],
                    [0.9, 0.0, 0.1, 0.0, 0.8],
                ]
            ),
            "memory_r2": np.array([[0.0], [0.8], [-0.5]]),
            "memory_training_r2": np.array([[0.4], [0.95], [0.7]]),
            "memory_time": np.arange(6) * 0.1,
            "memory_output": np.array(
                [[0.0, 0.0, 0.1, 0.2, 0.4, 0.6], [0.0, 0.0, 0.1, 0.5, 0.9, 1.0], [0.0, 0.0, 0.0, -0.1, 0.0, 0.1]]
            ),
            "memory_target": np.tile(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0]), (3, 1)),
            "memory_cue_steps": np.array(1),
            "memory_delay_steps": np.array(2),
            "memory_output_window_steps": np.array(1),
            "memory_output_start_step": np.array(4),
        }

        figures = plot_panels(arrays)

        self.assertEqual(set(figures), {"panel_A_decoding", "panel_B_prediction", "panel_C_memory"})
        expected_titles = {
            "panel_A_decoding": {"Best: K=1", "Worst: K=10"},
            "panel_B_prediction": {"Best: K=1", "Worst: K=0.1"},
            "panel_C_memory": {"Best: K=1", "Worst: K=10"},
        }
        for name, figure in figures.items():
            titles = {axis.get_title() for axis in figure.axes}
            self.assertTrue(expected_titles[name].issubset(titles))
            plt.close(figure)
        decoding_scan = figures["panel_A_decoding"].axes[0]
        self.assertEqual(decoding_scan.get_ylim(), (-1.0, 1.0))
        memory_scan = figures["panel_C_memory"].axes[0]
        self.assertEqual(memory_scan.get_ylim(), (-1.0, 1.0))
        prediction_figure = figures["panel_B_prediction"]
        prediction_scan = prediction_figure.axes[0]
        np.testing.assert_allclose(
            prediction_scan.lines[0].get_ydata(),
            np.array([3.0, 1.5, 2.0]),
        )
        amplitude_axes = [axis for axis in prediction_figure.axes if axis.get_ylabel() == "Peak amplitude"]
        self.assertEqual(len(amplitude_axes), 1)
        self.assertLess(amplitude_axes[0].lines[0].get_alpha(), 1.0)
        prediction_axes = [
            axis for axis in prediction_figure.axes
            if axis.get_title().startswith(("Best", "Worst"))
        ]
        self.assertTrue(
            all(np.allclose(axis.get_xlim(), (-0.1, 0.1)) for axis in prediction_axes)
        )
        self.assertTrue(
            all(
                np.max(np.abs(axis.lines[0].get_xdata())) <= 0.1
                for axis in prediction_axes
            )
        )
        memory_axes = [
            axis for axis in figures["panel_C_memory"].axes
            if axis.get_title().startswith(("Best", "Worst"))
        ]
        for axis in memory_axes:
            labels = axis.get_legend_handles_labels()[1]
            self.assertIn("Stimulus", labels)
            self.assertIn("Delay", labels)
            self.assertIn("Common cue", labels)
            self.assertIn("Scored response", labels)
            readout = next(line for line in axis.lines if line.get_label() == "Readout")
            self.assertEqual(np.isfinite(readout.get_ydata()).sum(), 6)
        rows = metric_rows(arrays)
        self.assertEqual(rows[0]["prediction_delay_ms_mean"], 3.0)
        self.assertAlmostEqual(rows[0]["prediction_peak_amplitude_mean"], 0.25)
        self.assertEqual(rows[1]["memory_training_r2_mean"], 0.95)

    def test_save_panel_writes_vector_and_raster_files(self) -> None:
        figure, axis = plt.subplots()
        axis.plot([0.0, 1.0], [0.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            paths = save_panel(figure, Path(directory), "panel_A_decoding")

            self.assertEqual({path.suffix for path in paths}, {".pdf", ".png"})
            self.assertTrue(all(path.exists() for path in paths))
        plt.close(figure)

    def test_assembled_figure_places_each_scan_left_of_two_examples(self) -> None:
        arrays = {
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
        plot_assembled = getattr(figure2_tasks, "plot_assembled_figure", None)

        self.assertTrue(callable(plot_assembled))
        figure = plot_assembled(arrays)

        titled_axes = [
            axis
            for axis in figure.axes
            if axis.get_title().startswith(("Best", "Worst"))
        ]
        self.assertEqual(len(titled_axes), 6)
        scan_axes = [axis for axis in figure.axes if axis.get_xlabel() == "K"]
        self.assertEqual(len(scan_axes), 3)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in scan_axes],
            ["Decoding", "Prediction", "Working memory"],
        )
        for row, scan_axis in enumerate(scan_axes):
            row_examples = [
                axis
                for axis in titled_axes
                if abs(axis.get_position().y0 - scan_axis.get_position().y0) < 0.02
            ]
            self.assertEqual(len(row_examples), 2, msg=f"row {row}")
            self.assertTrue(
                all(scan_axis.get_position().x1 < axis.get_position().x0 for axis in row_examples)
            )
        legend_text = [
            text
            for axis in figure.axes
            if axis.get_legend() is not None
            for text in axis.get_legend().get_texts()
        ]
        self.assertTrue(legend_text)
        self.assertTrue(all(text.get_fontsize() >= 10 for text in legend_text))
        self.assertEqual([text.get_text() for text in figure.texts], ["A", "B", "C"])
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
