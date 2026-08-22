import unittest

import numpy as np

from src.analysis.rank_one import run_rank_one_scan
from src.disorder import DisorderConfig


class RankOneAnalysisTest(unittest.TestCase):
    def test_small_scan_is_reproducible(self):
        config = DisorderConfig(
            N=5,
            seed=3,
            init_steps=2,
            record_steps=8,
            rank=1,
            pattern_type="gabor",
        )
        first = run_rank_one_scan(config, [10], [0.0], 4)
        second = run_rank_one_scan(config, [10], [0.0], 4)
        np.testing.assert_allclose(first.latent_coherence, second.latent_coherence)
        np.testing.assert_allclose(first.kappa, second.kappa)
        self.assertEqual(first.kappa.shape, (1, 1, 8))


if __name__ == "__main__":
    unittest.main()
