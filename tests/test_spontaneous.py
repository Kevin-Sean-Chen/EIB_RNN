"""Tests for the Figure 1 spontaneous-activity analysis."""

from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.spontaneous import (
    current_balance,
    pca_variance_curve,
    participation_dimension,
    radial_spatial_correlation,
    radial_spatial_spectrum,
    temporal_autocorrelation,
    temporal_spectrum,
)
from src.local import LocalConfig
from scripts.figures.figure1_spontaneous import plot_cancellation, save_panel
from scripts.figures.assemble_figure1 import assemble_figure


class SpontaneousAnalysisTests(unittest.TestCase):
    """Check the metrics used in Figure 1 panels C--G."""

    def test_participation_dimension_detects_one_component(self) -> None:
        pattern = np.arange(9, dtype=float).reshape(3, 3)
        time_course = np.array([-2.0, -1.0, 1.0, 2.0])
        activity = pattern[:, :, None] * time_course[None, None, :]

        self.assertAlmostEqual(participation_dimension(activity), 1.0)

    def test_pca_variance_curve_matches_participation_dimension(self) -> None:
        first = np.array([1.0, -1.0, 1.0, -1.0])
        second = np.array([1.0, 1.0, -1.0, -1.0])
        time = np.arange(8, dtype=float)
        activity = (
            first[:, None] * np.sin(2.0 * np.pi * time[None, :] / 8.0)
            + second[:, None] * np.cos(2.0 * np.pi * time[None, :] / 8.0)
        ).reshape(2, 2, 8)

        variance = pca_variance_curve(activity)

        self.assertAlmostEqual(variance.sum(), 1.0)
        self.assertAlmostEqual(1.0 / np.sum(variance**2), 2.0)
        self.assertAlmostEqual(
            1.0 / np.sum(variance**2),
            participation_dimension(activity),
        )

    def test_spatial_correlation_starts_at_one(self) -> None:
        axis = np.arange(9)
        pattern = np.tile(
            np.cos(2.0 * np.pi * axis[:, None] / 9.0),
            (1, 9),
        )
        activity = np.repeat(pattern[:, :, None], 5, axis=2)

        distance, correlation = radial_spatial_correlation(activity)

        self.assertEqual(distance[0], 0.0)
        self.assertAlmostEqual(correlation[0], 1.0)
        self.assertEqual(distance.shape, correlation.shape)

    def test_temporal_autocorrelation_finds_period(self) -> None:
        time = np.arange(80)
        series = np.sin(2.0 * np.pi * time / 10.0)
        activity = np.tile(series, (3, 3, 1))

        lag, correlation = temporal_autocorrelation(
            activity,
            sample_interval=0.5,
            max_lag_samples=20,
        )

        self.assertAlmostEqual(correlation[0], 1.0)
        self.assertGreater(correlation[10], 0.9)
        self.assertAlmostEqual(lag[10], 5.0)

    def test_spatial_spectrum_finds_cosine_wave_number(self) -> None:
        size = 15
        wave_number = 3
        axis = np.arange(size)
        pattern = np.tile(
            np.cos(2.0 * np.pi * wave_number * axis[:, None] / size),
            (1, size),
        )
        activity = np.repeat(pattern[:, :, None], 4, axis=2)

        radius, power = radial_spatial_spectrum(activity)

        self.assertEqual(radius[np.argmax(power)], wave_number)
        self.assertAlmostEqual(power.sum(), 1.0)

    def test_spatial_spectrum_includes_diagonal_wave_numbers(self) -> None:
        size = 15
        axis = np.arange(size)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        pattern = np.cos(2.0 * np.pi * 7.0 * (x + y) / size)
        activity = np.repeat(pattern[:, :, None], 4, axis=2)

        radius, power = radial_spatial_spectrum(activity)

        self.assertEqual(radius[np.argmax(power)], 10.0)
        self.assertAlmostEqual(power.sum(), 1.0)

    def test_temporal_spectrum_finds_frequency(self) -> None:
        sample_interval = 0.01
        time = np.arange(1000) * sample_interval
        signal = np.sin(2.0 * np.pi * 5.0 * time)
        activity = np.tile(signal, (2, 2, 1))

        frequency, power = temporal_spectrum(
            activity,
            sample_interval=sample_interval,
            window_samples=500,
        )

        self.assertAlmostEqual(frequency[np.argmax(power)], 5.0)
        self.assertAlmostEqual(power.sum(), 1.0)

    def test_current_balance_returns_named_components(self) -> None:
        config = LocalConfig(N=5, K=4.0, u_e=10.0)
        excitatory = np.full((5, 5, 3), 2.0)
        inhibitory = np.full((5, 5, 3), 1.0)

        balance = current_balance(excitatory, inhibitory, config)

        self.assertEqual(
            set(balance),
            {
                "external",
                "excitatory",
                "inhibitory",
                "net",
                "cancellation",
                "active_cancellation",
                "inactive_cancellation",
            },
        )
        self.assertAlmostEqual(balance["external"], 10.0)
        self.assertAlmostEqual(
            balance["net"],
            balance["external"]
            + balance["excitatory"]
            + balance["inhibitory"],
        )
        self.assertGreaterEqual(balance["cancellation"], 0.0)
        self.assertLessEqual(balance["cancellation"], 1.0)

    def test_save_panel_writes_pdf_and_png(self) -> None:
        figure, axis = plt.subplots()
        axis.plot([0.0, 1.0], [0.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            paths = save_panel(figure, Path(directory), "panel_B_traces")
            self.assertEqual({path.suffix for path in paths}, {".pdf", ".png"})
            self.assertTrue(all(path.exists() for path in paths))
        plt.close(figure)

    def test_cancellation_curves_have_distinct_visible_styles(self) -> None:
        figure, axis = plt.subplots()
        K_values = np.array([1.0, 100.0, 10000.0])
        mean = np.array(
            [
                [0.02, 0.03, 0.02],
                [0.15, 0.04, 0.16],
                [0.52, 0.07, 0.52],
            ]
        )
        spread = np.zeros_like(mean)

        lines = plot_cancellation(axis, K_values, mean, spread)

        self.assertEqual([line.get_label() for line in lines], ["All sites", "Active sites", "Inactive sites"])
        self.assertEqual(len({line.get_linestyle() for line in lines}), 3)
        self.assertTrue(all(line.get_marker() == "o" for line in lines))
        self.assertEqual(lines[0].get_markerfacecolor(), "white")
        self.assertGreater(lines[0].get_zorder(), lines[2].get_zorder())
        plt.close(figure)

    def test_assemble_figure_writes_png_and_pdf(self) -> None:
        panel_names = (
            "panel_A_model_patterns",
            "panel_B_traces",
            "panel_C_dimension",
            "panel_D_spatial_correlation",
            "panel_E_temporal_correlation",
            "panel_F_spectra",
            "panel_G_balance",
        )
        with tempfile.TemporaryDirectory() as directory:
            panel_directory = Path(directory)
            for index, name in enumerate(panel_names):
                image = np.full((20, 40, 3), index / len(panel_names))
                plt.imsave(panel_directory / f"{name}.png", image)

            paths = assemble_figure(panel_directory)

            self.assertEqual({path.name for path in paths}, {"figure1.pdf", "figure1.png"})
            self.assertTrue(all(path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
