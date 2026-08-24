import unittest
import numpy as np

from src.learning.rigid_reconstruction import train_rigid_reconstruction
from src.models.rigid_reconstruction import RigidReconstructionReservoir
from src.stimuli import rigid_shift_movie
from src.tasks.rigid_reconstruction import RigidReconstructionConfig


class RigidReconstructionTest(unittest.TestCase):
    def test_small_run_is_finite(self):
        config = RigidReconstructionConfig(
            N=5, steps=20, smoothing_width=0.2, shift_distance=2,
            training_trials=2, evaluation_trials=2, epochs=2,
        )
        stimulus, target = rigid_shift_movie(5, 20, config.dt, 0.2, 2, 3)
        result = train_rigid_reconstruction(RigidReconstructionReservoir(config), stimulus, target)
        self.assertTrue(np.isfinite(result.evaluation_mse))
        self.assertEqual(result.prediction.shape, target.shape)


if __name__ == "__main__":
    unittest.main()
