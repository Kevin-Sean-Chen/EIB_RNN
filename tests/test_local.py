import unittest

import numpy as np

from src.local import LocalConfig, simulate_local


class LocalSimulationTest(unittest.TestCase):
    def test_small_simulation_is_reproducible(self):
        config = LocalConfig(
            N=5,
            seed=3,
            init_steps=10,
            record_steps=15,
            steps_per_record=5,
        )
        first = simulate_local(config)
        second = simulate_local(config)
        np.testing.assert_array_equal(first.excitatory, second.excitatory)
        self.assertEqual(first.excitatory.shape, (5, 5, 3))
        self.assertTrue(np.isfinite(first.excitatory).all())


if __name__ == "__main__":
    unittest.main()
