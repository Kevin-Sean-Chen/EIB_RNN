import unittest

import numpy as np
import torch

from src.disorder import DisorderConfig, make_disorder_patterns, simulate_disorder


class DisorderTest(unittest.TestCase):
    def test_random_patterns_are_reproducible(self):
        config = DisorderConfig(N=5, seed=11, rank=2)
        first = make_disorder_patterns(config)
        second = make_disorder_patterns(config)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_small_simulation_has_finite_output(self):
        config = DisorderConfig(
            N=5,
            init_steps=2,
            record_steps=3,
            strength=0.1,
            seed=2,
        )
        result = simulate_disorder(config)
        self.assertEqual(result.excitatory.shape, (5, 5, 3))
        self.assertTrue(np.isfinite(result.excitatory).all())


if __name__ == "__main__":
    unittest.main()
