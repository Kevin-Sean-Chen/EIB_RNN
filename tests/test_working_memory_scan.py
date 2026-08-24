import unittest

import numpy as np

from src.analysis.working_memory import run_working_memory_K_scan
from src.tasks.working_memory import WorkingMemoryConfig


class WorkingMemoryScanTest(unittest.TestCase):
    def test_small_scan_is_reproducible(self):
        config = WorkingMemoryConfig(
            N=5, steps=24, delay_steps=8, cue_steps=4,
            training_trials=2, evaluation_trials=2, seed=3,
        )
        first = run_working_memory_K_scan(config, [10.0, 20.0])
        second = run_working_memory_K_scan(config, [10.0, 20.0])
        np.testing.assert_allclose(first.output_mse, second.output_mse)
        np.testing.assert_allclose(first.memory_mse, second.memory_mse)
        self.assertTrue(np.isfinite(first.output_r2).all())
        self.assertTrue(np.isfinite(first.memory_r2).all())


if __name__ == "__main__":
    unittest.main()
