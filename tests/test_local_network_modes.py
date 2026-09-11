"""Tests for local and non-local mode analysis."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

import src.analysis.local_network_modes as local_network_modes
from scripts.baseline.scan_K_rhoF_modes import (
    plot_balance,
    plot_diagnostics,
    plot_ei_cancellation,
    plot_results,
)
from src.analysis.local_network_modes import (
    ModeScanConfig,
    periodic_gaussian_kernel,
    reconstruction_curve,
    run_scan,
)


class LocalNetworkModeTests(unittest.TestCase):
    """Check the main numerical rules for mode analysis."""

    def test_periodic_kernel_is_normalized(self) -> None:
        kernel = periodic_gaussian_kernel(5, 0.1)

        self.assertEqual(kernel.shape, (5, 5))
        self.assertAlmostEqual(float(kernel.sum()), 1.0)
        np.testing.assert_allclose(kernel, kernel.T)

    def test_identity_basis_recovers_all_variance(self) -> None:
        activity = np.array(
            [
                [0.0, 1.0, 2.0],
                [2.0, 0.0, 1.0],
            ]
        )

        curve = reconstruction_curve(activity, np.eye(2))

        self.assertAlmostEqual(float(curve[-1]), 1.0)
        self.assertTrue(np.all(np.diff(curve) >= 0.0))

    def test_activity_diagnostics_separate_projected_and_total_power(self) -> None:
        activity = np.array(
            [
                [1.0, 3.0],
                [2.0, 4.0],
            ]
        )
        left = np.array([[1.0], [0.0]])
        right = np.array([[1.0], [1.0]])

        result = local_network_modes.activity_diagnostics(
            activity,
            left,
            right,
            strength=2.0,
            spatial_width=2,
        )

        self.assertAlmostEqual(result.lowrank_power, 1.0)
        self.assertAlmostEqual(result.total_variance, 2.0)
        self.assertAlmostEqual(result.lowrank_fraction, 0.5)
        self.assertAlmostEqual(result.lowrank_input_variance, 1.0)
        self.assertAlmostEqual(result.lowrank_output_variance, 4.0)

    def test_zero_strength_has_zero_lowrank_output_variance(self) -> None:
        activity = np.array([[1.0, 3.0], [2.0, 4.0]])
        left = np.array([[1.0], [0.0]])
        right = np.array([[1.0], [1.0]])

        result = local_network_modes.activity_diagnostics(
            activity,
            left,
            right,
            strength=0.0,
            spatial_width=2,
        )

        self.assertEqual(result.lowrank_output_variance, 0.0)

    def test_activity_diagnostics_ignore_rank_deficient_directions(self) -> None:
        activity = np.array([[0.0, 0.0], [1.0, -1.0]])
        left = np.array([[1.0, 1.0], [0.0, 0.0]])
        right = np.eye(2)

        result = local_network_modes.activity_diagnostics(
            activity,
            left,
            right,
            strength=1.0,
            spatial_width=2,
        )

        self.assertAlmostEqual(result.lowrank_power, 0.0)
        self.assertAlmostEqual(result.lowrank_fraction, 0.0)

    def test_current_diagnostics_detect_exact_ei_cancellation(self) -> None:
        excitatory = np.array([[1.0, 3.0], [2.0, 4.0]])
        inhibitory = -excitatory

        result = local_network_modes.current_diagnostics(excitatory, inhibitory)

        self.assertAlmostEqual(result.excitatory_power, 2.0)
        self.assertAlmostEqual(result.inhibitory_power, 2.0)
        self.assertAlmostEqual(result.net_power, 0.0)
        self.assertAlmostEqual(result.cancellation_ratio, 0.0)
        self.assertAlmostEqual(result.ei_correlation, -1.0)

    def test_current_diagnostics_preserve_small_net_power(self) -> None:
        excitatory = np.array([[1e8, -1e8]])
        inhibitory = np.array([[-1e8 + 1.0, 1e8 - 1.0]])

        result = local_network_modes.current_diagnostics(excitatory, inhibitory)

        self.assertAlmostEqual(result.net_power, 1.0)

    def test_current_diagnostics_are_scale_invariant(self) -> None:
        excitatory = np.array([[1e-10, -1e-10]])
        inhibitory = excitatory.copy()

        result = local_network_modes.current_diagnostics(excitatory, inhibitory)

        self.assertAlmostEqual(result.cancellation_ratio, 2.0)
        self.assertAlmostEqual(result.ei_correlation, 1.0)

    def test_constant_currents_have_undefined_cancellation_metrics(self) -> None:
        excitatory = np.ones((2, 3))
        inhibitory = -np.ones((2, 3))

        result = local_network_modes.current_diagnostics(excitatory, inhibitory)

        self.assertTrue(np.isnan(result.cancellation_ratio))
        self.assertTrue(np.isnan(result.ei_correlation))

    def test_population_balance_measures_global_current_cancellation(self) -> None:
        external = np.full((1, 2), 2.0)
        excitatory = np.full((1, 2), 3.0)
        inhibitory = np.full((1, 2), -4.0)
        active = np.ones((1, 2), dtype=bool)

        result = local_network_modes.population_balance_diagnostics(
            external, excitatory, inhibitory, active
        )

        self.assertAlmostEqual(result.mean_balance, 1.0 / 9.0)
        self.assertAlmostEqual(result.active_local_balance, 1.0 / 9.0)
        self.assertTrue(np.isnan(result.inactive_local_balance))
        self.assertAlmostEqual(result.mean_net_current, 1.0)

    def test_population_balance_separates_active_and_inactive_sites(self) -> None:
        external = np.zeros((1, 2))
        excitatory = np.full((1, 2), 2.0)
        inhibitory = np.array([[-1.0, -3.0]])
        active = np.array([[True, False]])

        result = local_network_modes.population_balance_diagnostics(
            external, excitatory, inhibitory, active
        )

        self.assertAlmostEqual(result.active_local_balance, 1.0 / 3.0)
        self.assertAlmostEqual(result.inactive_local_balance, 1.0 / 5.0)

    def test_small_scan_is_deterministic(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0, 0.5],
            rank=2,
            seed=3,
            n_seeds=1,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )

        first = run_scan(config)
        second = run_scan(config)

        np.testing.assert_array_equal(first.stable, second.stable)
        np.testing.assert_allclose(
            first.transition_index,
            second.transition_index,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            first.dimensions,
            second.dimensions,
            equal_nan=True,
        )

    def test_zero_strength_uses_same_local_and_full_basis(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0],
            rank=2,
            seed=3,
            n_seeds=1,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )

        result = run_scan(config)

        np.testing.assert_allclose(result.local_curves[0], result.network_curves[0])

    def test_scan_returns_nonlocal_alignment_diagnostics(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0, 0.5],
            rank=2,
            seed=3,
            n_seeds=2,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )

        result = run_scan(config)

        for values in (
            result.nonlocal_fraction,
            result.nonlocal_fraction_std,
            result.null_fraction,
            result.null_fraction_std,
            result.full_nonlocal_overlap,
            result.full_nonlocal_overlap_std,
            result.excitatory_active_fraction,
            result.inhibitory_active_fraction,
        ):
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.all((0.0 <= values) & (values <= 1.0)))

        self.assertGreater(
            result.full_nonlocal_overlap[-1],
            result.full_nonlocal_overlap[0],
        )

        for values in (
            result.lowrank_power,
            result.lowrank_power_std,
            result.total_variance,
            result.total_variance_std,
            result.lowrank_input_variance,
            result.lowrank_input_variance_std,
            result.lowrank_output_variance,
            result.lowrank_output_variance_std,
        ):
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.all(values >= 0.0))

    def test_k_scan_summary_has_four_diagnostic_panels(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0, 0.5],
            rank=2,
            seed=3,
            n_seeds=1,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )
        result = run_scan(config)

        figure = plot_results({10.0: result}, config, 0.02)

        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(
            [axis.get_title() for axis in figure.axes],
            [
                "Local and full reconstruction",
                "Full-mode advantage",
                "Non-local activity alignment",
                "Full-to-non-local overlap",
            ],
        )
        plt.close(figure)

    def test_diagnostic_figure_has_five_mechanism_panels(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0, 0.5],
            rank=2,
            seed=3,
            n_seeds=1,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )
        result = run_scan(config)

        figure = plot_diagnostics({10.0: result})

        self.assertEqual(
            [axis.get_title() for axis in figure.axes],
            [
                "Low-rank activity power",
                "Total activity variance",
                "Active fractions",
                "Low-rank input variance",
                "Low-rank output variance",
            ],
        )
        plt.close(figure)

    def test_ei_figure_has_four_cancellation_panels(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0, 0.5],
            rank=2,
            seed=3,
            n_seeds=1,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )
        result = run_scan(config)

        figure = plot_ei_cancellation({10.0: result})

        self.assertEqual(
            [axis.get_title() for axis in figure.axes[:4]],
            [
                "E, I, and net current power",
                "E/I cancellation ratio",
                "E/I current correlation",
                "Gating and low-rank susceptibility",
            ],
        )
        plt.close(figure)

    def test_balance_figure_has_four_panels(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0, 0.5],
            rank=2,
            seed=3,
            n_seeds=1,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )
        result = run_scan(config)

        figure = plot_balance({10.0: result}, config)

        self.assertEqual(
            [axis.get_title() for axis in figure.axes],
            [
                "Global mean balance",
                "Active-site local balance",
                "Inactive-site local balance",
                "Signed mean currents at K=10",
            ],
        )
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
