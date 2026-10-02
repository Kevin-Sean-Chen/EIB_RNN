import unittest

import numpy as np
import torch

from scripts.relu2D_asym import relu2D_driven_step


class AsymmetricRankOneTest(unittest.TestCase):
    def test_zero_strength_preserves_default_output(self):
        arguments = (
            np.ones((3, 3)),
            np.zeros((3, 3)),
            3,
            0.001,
            1,
            "relu_gaussian",
            1.0,
            np.ones(2),
            np.zeros(2),
            np.eye(2),
            np.array([0.05, 0.05]),
            np.zeros((3, 3)),
            0.0,
        )

        default_result = relu2D_driven_step(*arguments)
        zero_result = relu2D_driven_step(
            *arguments,
            g=0.0,
            m=torch.arange(9),
            n=torch.arange(9),
        )

        for default_value, zero_value in zip(default_result, zero_result):
            np.testing.assert_array_equal(
                np.asarray(default_value),
                np.asarray(zero_value),
            )

    def test_rank_one_vectors_must_match_network_size(self):
        try:
            relu2D_driven_step(
                np.ones((3, 3)),
                np.zeros((3, 3)),
                3,
                0.001,
                1,
                "relu_gaussian",
                1.0,
                np.ones(2),
                np.zeros(2),
                np.eye(2),
                np.array([0.05, 0.05]),
                np.zeros((3, 3)),
                0.0,
                g=1.0,
                m=torch.ones(8),
                n=torch.ones(9),
            )
        except ValueError as error:
            self.assertIn("N**2", str(error))
        except Exception as error:
            self.fail(f"The function raised an unclear error: {error}")
        else:
            self.fail("The function accepted a vector with the wrong size")

    def test_nonzero_strength_requires_both_vectors(self):
        try:
            relu2D_driven_step(
                np.ones((3, 3)),
                np.zeros((3, 3)),
                3,
                0.001,
                1,
                "relu_gaussian",
                1.0,
                np.ones(2),
                np.zeros(2),
                np.eye(2),
                np.array([0.05, 0.05]),
                np.zeros((3, 3)),
                0.0,
                g=1.0,
            )
        except ValueError as error:
            self.assertIn("m and n", str(error))
        except Exception as error:
            self.fail(f"The function raised an unclear error: {error}")
        else:
            self.fail("The function accepted missing rank-one vectors")

    def test_rank_one_term_changes_excitatory_input(self):
        N = 3
        state = np.ones((N, N))
        zeros = np.zeros((N, N))
        vector = torch.ones(N**2)

        try:
            _, _, baseline_mu, _ = relu2D_driven_step(
                state,
                zeros,
                N,
                0.001,
                1,
                "relu_gaussian",
                1.0,
                np.ones(2),
                np.zeros(2),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([0.05, 0.05]),
                zeros,
                0.0,
            )
            _, _, disorder_mu, _ = relu2D_driven_step(
                state,
                zeros,
                N,
                0.001,
                1,
                "relu_gaussian",
                1.0,
                np.ones(2),
                np.zeros(2),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([0.05, 0.05]),
                zeros,
                0.0,
                g=1.0,
                m=vector,
                n=vector,
            )
        except TypeError as error:
            self.fail(f"The rank-one parameters are not available: {error}")

        np.testing.assert_allclose(
            (disorder_mu - baseline_mu).cpu().numpy(),
            np.full((1, 1, N, N), 3.0),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
