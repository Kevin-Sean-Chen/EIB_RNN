import unittest

import numpy as np

from src.analysis.dmd import DMDConfig, spatial_dmd


class DMDTest(unittest.TestCase):
    def test_traveling_wave_has_finite_modes_and_decreasing_error(self):
        size = 7
        steps = 60
        x = np.arange(size)[:, None, None]
        time = np.arange(steps)[None, None, :]
        activity = np.cos(2 * np.pi * x / size - 0.2 * time)
        activity = np.broadcast_to(activity, (size, size, steps)).copy()
        result = spatial_dmd(activity, 0.1, DMDConfig(rank=4, shown_modes=2))
        self.assertEqual(result.modes.shape[1:], (size, size))
        self.assertTrue(np.isfinite(result.growth_rates).all())
        self.assertTrue(np.all(np.diff(result.rank_errors) <= 1e-10))

    def test_constant_activity_has_no_dynamic_modes(self):
        with self.assertRaises(ValueError):
            spatial_dmd(np.ones((5, 5, 10)), 0.1, DMDConfig(rank=3))


if __name__ == "__main__":
    unittest.main()
