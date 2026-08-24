import unittest

import numpy as np
import torch

from src.learning.rls import initialize_inverse_correlation, update_rls
from src.tasks.working_memory import make_working_memory_trial


class RlsTest(unittest.TestCase):
    def test_rls_learns_one_linear_mapping(self):
        weights = torch.zeros(2, 1)
        inverse = initialize_inverse_correlation(2, delta=1.0)
        target_weights = torch.tensor([[2.0], [-1.0]])
        features = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 1.0]),
        ]
        initial_error = None
        final_error = None
        for _ in range(20):
            for feature in features:
                target = feature @ target_weights
                error = feature @ weights - target
                if initial_error is None:
                    initial_error = abs(float(error))
                weights, inverse = update_rls(weights, inverse, feature, error)
                final_error = abs(float(feature @ weights - target))
        self.assertLess(final_error, initial_error)
        torch.testing.assert_close(weights, target_weights, atol=0.05, rtol=0.05)


class WorkingMemoryTrialTest(unittest.TestCase):
    def test_trial_has_trigger_delay_and_go_cue(self):
        patterns = (
            np.ones((3, 3), dtype=np.float32),
            -np.ones((3, 3), dtype=np.float32),
            np.full((3, 3), 2.0, dtype=np.float32),
        )
        output, memory, stimulus, choice = make_working_memory_trial(
            N=3,
            steps=10,
            delay_steps=4,
            cue_steps=2,
            input_patterns=patterns,
            choice=1,
        )
        self.assertEqual(choice, 1)
        np.testing.assert_array_equal(stimulus[:, :, 0], patterns[1])
        np.testing.assert_array_equal(stimulus[:, :, 3], 0)
        np.testing.assert_array_equal(stimulus[:, :, 6], patterns[2])
        self.assertEqual(float(memory[3]), -1.0)
        self.assertEqual(float(output[-1]), -1.0)


if __name__ == "__main__":
    unittest.main()
