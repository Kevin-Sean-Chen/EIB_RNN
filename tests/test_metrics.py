import unittest

import numpy as np

from src.metrics import latent_coherence, linear_dimension, second_acf_peak


class MetricsTest(unittest.TestCase):
    def test_latent_coherence_is_one_for_matching_rank_one_activity(self):
        pattern = np.array([1.0, -1.0, 1.0, -1.0])
        time_course = np.array([1.0, 2.0, -1.0])
        activity = pattern[:, None] * time_course[None, :]
        self.assertAlmostEqual(latent_coherence(activity, pattern), 1.0)

    def test_linear_dimension_detects_one_component(self):
        weights = np.arange(9, dtype=float).reshape(3, 3)
        activity = weights[:, :, None] * np.arange(4, dtype=float)
        self.assertEqual(linear_dimension(activity), 1)

    def test_second_acf_peak_finds_periodic_recurrence(self):
        peak, acf = second_acf_peak(np.tile([1.0, -1.0], 10))
        self.assertGreater(peak, 0.5)
        self.assertEqual(acf[0], 1.0)


if __name__ == "__main__":
    unittest.main()
