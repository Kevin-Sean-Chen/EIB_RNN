import unittest

import numpy as np

from src.analysis.spectral import (
    SpectralConfig,
    circular_convolution_matrix,
    legacy_block_operator,
    legacy_gaussian_kernel,
    run_spectral_scan,
)


class SpectralAnalysisTest(unittest.TestCase):
    def test_convolution_matrix_centers_an_impulse(self):
        kernel = legacy_gaussian_kernel(5, 0.1)
        operator = circular_convolution_matrix(kernel)
        impulse = np.zeros(25)
        impulse[12] = 1
        np.testing.assert_allclose((operator @ impulse).reshape(5, 5), kernel)

    def test_legacy_block_has_repeated_population_rows_before_decay(self):
        local_e = np.eye(9)
        local_i = 2 * np.eye(9)
        pattern = np.ones(9)
        operator = legacy_block_operator(local_e, local_i, pattern, pattern, K=1, strength=0)
        np.testing.assert_allclose(operator[:9, :9] + np.eye(9), operator[9:, :9])
        np.testing.assert_allclose(operator[:9, 9:], operator[9:, 9:] + np.eye(9))

    def test_small_scan_returns_finite_eigenvalues(self):
        config = SpectralConfig(
            N=5,
            K_values=[10],
            phase_values=[0.0],
            leading_eigenvalues=4,
        )
        result = run_spectral_scan(config)
        self.assertEqual(result.eigenvalues.shape, (1, 1, 4))
        self.assertTrue(np.isfinite(result.eigenvalues).all())


if __name__ == "__main__":
    unittest.main()
