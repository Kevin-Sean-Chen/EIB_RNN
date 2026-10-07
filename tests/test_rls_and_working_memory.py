import unittest

import numpy as np
import torch

from src.learning.rls import initialize_inverse_correlation, update_rls
from src.analysis.working_memory import make_patterns
from src.learning.force import run_trial
from src.learning.ridge import collect_readout_data
from src.tasks.working_memory import WorkingMemoryConfig, make_working_memory_trial


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
    def test_pattern_scales_separate_trigger_and_go_cue(self):
        base = make_patterns(WorkingMemoryConfig(N=5, seed=3))
        scaled = make_patterns(
            WorkingMemoryConfig(
                N=5,
                seed=3,
                trigger_scale=4.0,
                go_scale=0.5,
            )
        )

        np.testing.assert_allclose(scaled[0], 4.0 * base[0])
        np.testing.assert_allclose(scaled[1], 4.0 * base[1])
        np.testing.assert_allclose(scaled[2], 0.5 * base[2])

    def test_collect_readout_data_respects_stop_step(self):
        class CountingModel:
            def initial_state(self, seed):
                return torch.tensor(0.0)

            def step(self, state, stimulus):
                return state + 1.0

            def features(self, state):
                return state.reshape(1)

            def predict(self, features):
                return features[0], features[0]

        trial = (
            torch.arange(6, dtype=torch.float32),
            torch.zeros(6),
            torch.zeros(1, 1, 6),
            0,
        )

        features, targets = collect_readout_data(
            CountingModel(),
            lambda _: trial,
            trial_count=2,
            start_step=2,
            target_index=3,
            seed=1,
            stop_step=5,
        )

        self.assertEqual(features.shape, (6, 2))
        np.testing.assert_array_equal(targets[:, 0], np.array([2, 3, 4, 2, 3, 4]))

    def test_run_trial_settles_before_recording(self):
        class CountingModel:
            def initial_state(self, seed):
                return torch.tensor(0.0)

            def step(self, state, stimulus):
                return state + 1.0

            def features(self, state):
                return state.reshape(1)

            def predict(self, features):
                return features[0], features[0]

        trial = (
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(1, 1, 2),
            0,
        )

        result = run_trial(CountingModel(), trial, "spatial", 3, settle_steps=3)

        self.assertEqual(float(result[0][0]), 4.0)

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

    def test_go_cue_is_common_to_both_choices(self):
        patterns = (
            np.ones((3, 3), dtype=np.float32),
            -np.ones((3, 3), dtype=np.float32),
            np.full((3, 3), 2.0, dtype=np.float32),
        )

        left = make_working_memory_trial(3, 10, 4, 2, patterns, choice=0)
        right = make_working_memory_trial(3, 10, 4, 2, patterns, choice=1)

        np.testing.assert_array_equal(left[2][:, :, 0], patterns[0])
        np.testing.assert_array_equal(right[2][:, :, 0], patterns[1])
        np.testing.assert_array_equal(left[2][:, :, 3], 0.0)
        np.testing.assert_array_equal(right[2][:, :, 3], 0.0)
        np.testing.assert_array_equal(left[2][:, :, 6], patterns[2])
        np.testing.assert_array_equal(right[2][:, :, 6], patterns[2])


if __name__ == "__main__":
    unittest.main()
