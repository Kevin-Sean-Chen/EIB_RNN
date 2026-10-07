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

    def test_noise_is_scaled_by_sqrt_K_and_only_changes_excitatory_input(self):
        N = 3
        zeros = np.zeros((N, N))
        input_pattern = torch.zeros((N, N))

        torch.manual_seed(1)
        _, _, unit_mue, unit_mui = relu2D_driven_step(
            zeros,
            zeros,
            N,
            0.001,
            1,
            "relu_gaussian",
            1.0,
            np.ones(2),
            np.zeros(2),
            np.zeros((2, 2)),
            np.array([0.05, 0.05]),
            input_pattern,
            0.0,
            noise_strength=1.0,
        )
        torch.manual_seed(1)
        _, _, scaled_mue, scaled_mui = relu2D_driven_step(
            zeros,
            zeros,
            N,
            0.001,
            1,
            "relu_gaussian",
            4.0,
            np.ones(2),
            np.zeros(2),
            np.zeros((2, 2)),
            np.array([0.05, 0.05]),
            input_pattern,
            0.0,
            noise_strength=1.0,
        )

        self.assertFalse(np.allclose(unit_mue.cpu().numpy(), 0.0))
        np.testing.assert_allclose(
            scaled_mue.cpu().numpy(),
            2.0 * unit_mue.cpu().numpy(),
        )
        np.testing.assert_array_equal(unit_mui.cpu().numpy(), 0.0)
        np.testing.assert_array_equal(scaled_mui.cpu().numpy(), 0.0)

    def test_noise_is_redrawn_for_each_integration_substep(self):
        N = 3
        zeros = np.zeros((N, N))
        input_pattern = torch.zeros((N, N))
        arguments = (
            zeros,
            zeros,
            N,
            0.001,
        )
        trailing_arguments = (
            "relu_gaussian",
            1.0,
            np.ones(2),
            np.zeros(2),
            np.zeros((2, 2)),
            np.array([0.05, 0.05]),
            input_pattern,
            0.0,
        )

        torch.manual_seed(2)
        _, _, first_mue, first_mui = relu2D_driven_step(
            *arguments,
            1,
            *trailing_arguments,
            noise_strength=1.0,
        )
        torch.manual_seed(2)
        _, _, second_mue, second_mui = relu2D_driven_step(
            *arguments,
            2,
            *trailing_arguments,
            noise_strength=1.0,
        )

        self.assertFalse(
            np.array_equal(first_mue.cpu().numpy(), second_mue.cpu().numpy())
        )
        np.testing.assert_array_equal(first_mui.cpu().numpy(), 0.0)
        np.testing.assert_array_equal(second_mui.cpu().numpy(), 0.0)

    def test_rate_cap_limits_both_populations_and_warns(self):
        N = 3
        zeros = np.zeros((N, N))

        try:
            with self.assertWarnsRegex(RuntimeWarning, "rate cap"):
                re, ri, _, _ = relu2D_driven_step(
                    zeros,
                    zeros,
                    N,
                    1.0,
                    1,
                    "relu_gaussian",
                    1.0,
                    np.ones(2),
                    np.array([4000.0, 5000.0]),
                    np.zeros((2, 2)),
                    np.array([0.05, 0.05]),
                    torch.zeros((N, N)),
                    0.0,
                    rate_cap=3000.0,
                )
        except TypeError as error:
            self.fail(f"The rate cap is not available: {error}")

        np.testing.assert_array_equal(re, np.full((N, N), 3000.0))
        np.testing.assert_array_equal(ri, np.full((N, N), 3000.0))


if __name__ == "__main__":
    unittest.main()
