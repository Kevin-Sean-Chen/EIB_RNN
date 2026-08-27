import unittest
import numpy as np
import torch

from src.learning.rigid_reconstruction import driven_lyapunov_exponent, train_rigid_reconstruction
from src.models.rigid_reconstruction import (
    NonSpatialRigidReconstructionReservoir,
    RandomEIRigidReconstructionReservoir,
    RigidReconstructionReservoir,
)
from src.stimuli import fixed_spatial_permutation, rigid_shift_movie
from src.tasks.rigid_reconstruction import RigidReconstructionConfig


class RigidReconstructionTest(unittest.TestCase):
    def test_spatial_permutation_is_fixed_across_time(self):
        frame = torch.arange(25, dtype=torch.float32).reshape(5, 5)
        movie = torch.stack((frame, frame + 100), dim=-1)
        shuffled = fixed_spatial_permutation(movie, seed=11)
        torch.testing.assert_close(shuffled[:, :, 1] - shuffled[:, :, 0], torch.full((5, 5), 100.0))
        torch.testing.assert_close(torch.sort(shuffled[:, :, 0].reshape(-1)).values, frame.reshape(-1))

    def test_small_run_is_finite(self):
        config = RigidReconstructionConfig(
            N=5, steps=20, smoothing_width=0.2, shift_distance=2,
            training_trials=2, evaluation_trials=2, epochs=2, init_steps=2,
        )
        stimulus, target = rigid_shift_movie(5, 20, config.dt, 0.2, 2, 3)
        result = train_rigid_reconstruction(RigidReconstructionReservoir(config), stimulus, target)
        self.assertTrue(np.isfinite(result.evaluation_mse))
        self.assertEqual(result.prediction.shape, target.shape)

    def test_same_size_non_spatial_control_is_finite(self):
        config = RigidReconstructionConfig(
            model_type="non_spatial", N=5, steps=20, smoothing_width=0.2,
            shift_distance=2, training_trials=1, evaluation_trials=1,
            init_steps=2,
        )
        stimulus, target = rigid_shift_movie(5, 20, config.dt, 0.2, 2, 3)
        model = NonSpatialRigidReconstructionReservoir(config)
        self.assertEqual(model.recurrent.shape, (25, 25))
        torch.testing.assert_close(
            model.recurrent.sum(dim=1), torch.zeros(25), atol=1e-6, rtol=0,
        )
        spatial_model = RigidReconstructionReservoir(config)
        self.assertEqual(model.feature_count, 26)
        self.assertEqual(model.feature_count, spatial_model.feature_count)
        result = train_rigid_reconstruction(model, stimulus, target)
        self.assertTrue(np.isfinite(result.evaluation_r2))

    def test_external_drive_matches_spatial_model(self):
        config = RigidReconstructionConfig(
            N=5, K=100.0, u_e=2.0, stimulus_gain=3.0,
            baseline_mode="add", init_steps=0,
        )
        spatial_model = RigidReconstructionReservoir(config)
        random_model = NonSpatialRigidReconstructionReservoir(config)
        stimulus = torch.ones((5, 5))
        zero_spatial = (torch.zeros((5, 5)), torch.zeros((5, 5)))
        zero_random = (torch.zeros(25),)
        spatial_e, _ = spatial_model.step(zero_spatial, stimulus)
        random_rate, = random_model.step(zero_random, stimulus)
        torch.testing.assert_close(random_rate, spatial_e.reshape(-1))

    def test_random_ei_control_matches_population_and_readout_sizes(self):
        config = RigidReconstructionConfig(model_type="random_ei", N=5, K=3, init_steps=0)
        model = RandomEIRigidReconstructionReservoir(config)
        self.assertEqual(model.connectivity_ee.shape, (25, 25))
        torch.testing.assert_close(model.connectivity_ee.sum(dim=1), torch.ones(25))
        self.assertEqual(model.feature_count, 26)
        state = model.initial_state(2)
        self.assertEqual(len(state), 2)
        self.assertEqual(state[0].shape, (5, 5))

    def test_driven_lyapunov_measure_is_finite(self):
        config = RigidReconstructionConfig(N=5, K=3, steps=10, init_steps=2)
        model = RandomEIRigidReconstructionReservoir(config)
        stimulus, _ = rigid_shift_movie(5, 10, config.dt, 0.2, 2, config.seed)
        exponent = driven_lyapunov_exponent(model, stimulus, config.seed, discard_steps=2)
        self.assertTrue(np.isfinite(exponent))


if __name__ == "__main__":
    unittest.main()
