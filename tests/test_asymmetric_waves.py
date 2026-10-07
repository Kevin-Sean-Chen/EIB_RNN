"""Tests for directed-wave measurements in the asymmetric network."""

import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np


def translated_movie(size: int, frame_count: int) -> np.ndarray:
    """Return one pattern translated by one diagonal pixel per frame."""
    rng = np.random.default_rng(4)
    pattern = rng.standard_normal((size, size))
    return np.stack(
        [np.roll(pattern, shift=(index, index), axis=(0, 1)) for index in range(frame_count)],
        axis=2,
    )


def synthetic_results() -> dict[str, np.ndarray]:
    """Return compact Figure 5 scan arrays."""
    K_values = np.array([1.0, 10.0, 100.0, 1000.0])
    bias_values = np.array([0.0, 2.0, 4.0, 8.0])
    seeds = np.array([1, 2, 3])
    shape = (len(bias_values), 2, len(K_values), len(seeds))
    speed = np.zeros(shape)
    coherence = np.zeros(shape)
    for bias_index in range(len(bias_values)):
        speed[bias_index, 0] = bias_index + np.log10(K_values)[None, :].T
        coherence[bias_index, 0] = np.clip(0.1 * bias_index + 0.1, 0.0, 1.0)
    speed[-1, 1] = speed[-1, 0] + 0.3
    coherence[-1, 1] = np.clip(coherence[-1, 0] + 0.2, 0.0, 1.0)
    patterns = np.arange(4 * 4 * 3 * 3, dtype=float).reshape(4, 4, 3, 3)
    return {
        "K_values": K_values,
        "bias_values": bias_values,
        "noise_values": np.array([0.0, 0.5]),
        "seeds": seeds,
        "signed_speed": speed,
        "directional_coherence": coherence,
        "example_patterns": patterns,
    }


class AsymmetricWaveTests(unittest.TestCase):
    """Check wave metrics and Figure 5 panel construction."""

    def test_directed_translation_has_expected_speed_and_unit_coherence(self) -> None:
        module_name = "src.analysis.asymmetric_waves"
        self.assertIsNotNone(importlib.util.find_spec(module_name))
        module = importlib.import_module(module_name)
        size = 8
        dt = 0.1

        metrics = module.directed_wave_metrics(
            translated_movie(size, 6),
            sample_interval=dt,
            lag_steps=1,
            expected_direction=(1.0, 1.0),
        )

        self.assertAlmostEqual(metrics.signed_speed, np.sqrt(2.0) / (size * dt))
        self.assertAlmostEqual(metrics.directional_coherence, 1.0)

    def test_stationary_pattern_has_zero_speed_and_coherence(self) -> None:
        module = importlib.import_module("src.analysis.asymmetric_waves")
        pattern = translated_movie(8, 1)
        movie = np.repeat(pattern, 5, axis=2)

        metrics = module.directed_wave_metrics(
            movie,
            sample_interval=0.1,
            lag_steps=1,
            expected_direction=(1.0, 1.0),
        )

        self.assertEqual(metrics.signed_speed, 0.0)
        self.assertEqual(metrics.directional_coherence, 0.0)

    def test_plot_panels_create_the_approved_four_panel_story(self) -> None:
        module_name = "scripts.figures.figure5_asymmetry"
        self.assertIsNotNone(importlib.util.find_spec(module_name))
        module = importlib.import_module(module_name)

        panels = module.plot_panels(synthetic_results())

        self.assertEqual(
            set(panels),
            {
                "panel_A_bias_schematic",
                "panel_B_K_bias_patterns",
                "panel_C_bias_metrics",
                "panel_D_noise_metrics",
            },
        )
        self.assertEqual(sum(len(axis.images) for axis in panels["panel_B_K_bias_patterns"].axes), 16)
        self.assertEqual(len(panels["panel_C_bias_metrics"].axes), 2)
        self.assertEqual(len(panels["panel_D_noise_metrics"].axes), 2)
        for stem in ("panel_C_bias_metrics", "panel_D_noise_metrics"):
            for axis in panels[stem].axes:
                self.assertEqual(axis.get_xscale(), "log")
                self.assertEqual(len(axis.lines), 2)

        for figure in panels.values():
            plt.close(figure)

    def test_save_outputs_writes_panels_and_assembly(self) -> None:
        module = importlib.import_module("scripts.figures.figure5_asymmetry")
        panels = module.plot_panels(synthetic_results())

        with tempfile.TemporaryDirectory() as directory:
            paths = module.save_outputs(panels, Path(directory))

            self.assertEqual(len(paths), 10)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))
        for figure in panels.values():
            plt.close(figure)

    def test_run_scan_returns_matched_condition_arrays(self) -> None:
        module = importlib.import_module("scripts.figures.figure5_asymmetry")
        config = module.Figure5Config(
            output_directory=Path("unused"),
            N=5,
            K_values=(1.0,),
            bias_values=(0.0, 1.0),
            noise_values=(0.0, 0.5),
            seeds=(3,),
            dt=0.0001,
            tau_e=0.01,
            tau_i=0.02,
            init_steps=2,
            record_steps=4,
            steps_per_record=1,
            u_e=10.0,
            u_i=0.0,
            sigma_e=0.05,
            sigma_i=0.05 * np.sqrt(2.0),
            metric_lag_steps=1,
            rate_cap=3000.0,
        )

        results = module.run_scan(config)

        self.assertEqual(results["signed_speed"].shape, (2, 2, 1, 1))
        self.assertEqual(results["directional_coherence"].shape, (2, 2, 1, 1))
        self.assertEqual(results["example_patterns"].shape, (2, 1, 5, 5))
        self.assertTrue(np.isnan(results["signed_speed"][0, 1]).all())
        self.assertTrue(np.isfinite(results["signed_speed"][-1, 1]).all())


if __name__ == "__main__":
    unittest.main()
