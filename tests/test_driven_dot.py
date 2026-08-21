"""Tests for driven moving-dot simulations and tracking."""

import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "Torch is not installed.")
class DrivenDotTests(unittest.TestCase):
    """Check stimulus, tracking metric, and deterministic simulation rules."""

    def test_moving_dot_has_expected_shape(self) -> None:
        from src.stimuli import moving_dot

        stimulus = moving_dot(5, 4, 0.2, 1.0)

        self.assertEqual(tuple(stimulus.shape), (5, 5, 4))
        self.assertTrue(torch.isfinite(stimulus).all())

    def test_identical_traces_have_zero_peak_lag(self) -> None:
        from src.tasks.driven_dot import overlap_cross_correlation

        trace = np.array([-1.0, 0.0, 1.0, 0.0])
        correlation, lags = overlap_cross_correlation(trace, trace)

        self.assertEqual(int(lags[np.argmax(correlation)]), 0)

    def test_small_tracking_scan_is_deterministic(self) -> None:
        from src.driven import DrivenDotConfig
        from src.tasks.driven_dot import run_tracking_scan

        config = DrivenDotConfig(
            N=5,
            K=10.0,
            seed=3,
            init_steps=2,
            record_steps=6,
            dot_size=0.2,
            drift_rate=1.0,
        )

        first = run_tracking_scan(config, [10.0], 1, 2)
        second = run_tracking_scan(config, [10.0], 1, 2)

        np.testing.assert_array_equal(first.stimulus, second.stimulus)
        np.testing.assert_array_equal(first.peak_lags, second.peak_lags)
        np.testing.assert_allclose(first.peak_heights, second.peak_heights)


if __name__ == "__main__":
    unittest.main()
