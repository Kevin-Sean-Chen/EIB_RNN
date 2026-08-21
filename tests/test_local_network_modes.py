"""Tests for local and non-local mode analysis."""

import unittest

import numpy as np

from src.analysis.local_network_modes import (
    ModeScanConfig,
    periodic_gaussian_kernel,
    reconstruction_curve,
    run_scan,
)


class LocalNetworkModeTests(unittest.TestCase):
    """Check the main numerical rules for mode analysis."""

    def test_periodic_kernel_is_normalized(self) -> None:
        kernel = periodic_gaussian_kernel(5, 0.1)

        self.assertEqual(kernel.shape, (5, 5))
        self.assertAlmostEqual(float(kernel.sum()), 1.0)
        np.testing.assert_allclose(kernel, kernel.T)

    def test_identity_basis_recovers_all_variance(self) -> None:
        activity = np.array(
            [
                [0.0, 1.0, 2.0],
                [2.0, 0.0, 1.0],
            ]
        )

        curve = reconstruction_curve(activity, np.eye(2))

        self.assertAlmostEqual(float(curve[-1]), 1.0)
        self.assertTrue(np.all(np.diff(curve) >= 0.0))

    def test_small_scan_is_deterministic(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0, 0.5],
            rank=2,
            seed=3,
            n_seeds=1,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )

        first = run_scan(config)
        second = run_scan(config)

        np.testing.assert_array_equal(first.stable, second.stable)
        np.testing.assert_allclose(
            first.transition_index,
            second.transition_index,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            first.dimensions,
            second.dimensions,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
