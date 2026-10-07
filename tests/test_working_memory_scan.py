import unittest

import numpy as np

from src.analysis.working_memory import output_readout_window, run_working_memory_K_scan
from src.tasks.working_memory import WorkingMemoryConfig


class WorkingMemoryScanTest(unittest.TestCase):
    def test_output_window_can_start_after_common_cue(self):
        config = WorkingMemoryConfig(
            N=5,
            steps=20,
            delay_steps=4,
            cue_steps=2,
            response_after_cue=True,
            output_window_steps=3,
        )

        self.assertEqual(output_readout_window(config), (8, 11))

    def test_small_scan_is_reproducible(self):
        config = WorkingMemoryConfig(
            N=5, steps=24, delay_steps=8, cue_steps=4,
            training_trials=2, evaluation_trials=2, seed=3,
            learning_method="ridge",
        )
        first = run_working_memory_K_scan(config, [10.0, 20.0])
        second = run_working_memory_K_scan(config, [10.0, 20.0])
        np.testing.assert_allclose(first.output_mse, second.output_mse)
        np.testing.assert_allclose(first.memory_mse, second.memory_mse)
        self.assertTrue(np.isfinite(first.output_r2).all())
        self.assertTrue(np.isfinite(first.training_output_r2).all())
        self.assertTrue(np.isfinite(first.memory_r2).all())
        self.assertEqual(first.final_output_accuracy.shape, (2,))
        self.assertTrue(np.all((first.final_output_accuracy >= 0.0) & (first.final_output_accuracy <= 1.0)))
        self.assertEqual(first.example_output.shape, (2, config.steps))
        self.assertEqual(first.example_output_target.shape, (2, config.steps))

    def test_scan_settles_each_trial_before_measurement(self):
        config = WorkingMemoryConfig(
            N=3, steps=12, delay_steps=4, cue_steps=2,
            training_trials=1, evaluation_trials=1, seed=2,
            init_steps=3,
        )

        result = run_working_memory_K_scan(config, [10.0])

        self.assertEqual(result.example_output.shape, (1, 12))
        self.assertTrue(np.isfinite(result.example_output).all())
        self.assertGreaterEqual(result.final_output_accuracy[0], 0.0)
        self.assertLessEqual(result.final_output_accuracy[0], 1.0)


if __name__ == "__main__":
    unittest.main()
