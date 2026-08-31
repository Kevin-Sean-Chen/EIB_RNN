import unittest

import numpy as np
import torch

from src.models.working_memory import (
    NonSpatialWorkingMemoryReservoir,
    SpatialWorkingMemoryReservoir,
)


class WorkingMemoryModelTest(unittest.TestCase):
    @staticmethod
    def make_spatial_model() -> SpatialWorkingMemoryReservoir:
        """Return a small legacy-matching spatial reservoir."""
        return SpatialWorkingMemoryReservoir(
            N=5,
            dt=0.001,
            tau_e=0.01,
            tau_i=0.01,
            K=20,
            coupling=np.array([[1, -4], [2, -2]]),
            drive=np.array([10, 0]),
            sigma=np.array([0.05, 0.05 * np.sqrt(2)]),
            stimulus_gain=10,
            feedback_gain=0.001,
            feedback_scale=0.01,
            use_feedback_output=True,
            use_feedback_memory=True,
            init_scale=0.1,
            microsteps=1,
            field_clip=100.0,
            seed=3,
        )

    def test_spatial_model_has_no_trainable_parameters(self):
        model = self.make_spatial_model()
        self.assertEqual(list(model.parameters()), [])
        state = model.initial_state(4)
        next_state = model.step(state, torch.zeros(5, 5))
        self.assertTrue(torch.isfinite(next_state[0]).all())

    def test_spatial_step_uses_legacy_feedback_order(self):
        model = self.make_spatial_model()
        state = model.initial_state(4)
        stimulus = torch.zeros(5, 5)
        re, ri = state
        pad = model.kernel_e.shape[-1] // 2
        local_e = torch.nn.functional.conv2d(
            torch.nn.functional.pad(re[None, None], (pad, pad, pad, pad), mode="circular"),
            model.kernel_e,
        ).squeeze()
        local_i = torch.nn.functional.conv2d(
            torch.nn.functional.pad(ri[None, None], (pad, pad, pad, pad), mode="circular"),
            model.kernel_i,
        ).squeeze()
        output, memory = model.predict(model.activation(re).reshape(-1))
        feedback = (
            model.feedback_output * output.squeeze()
            + model.feedback_memory * memory.squeeze()
        )
        field_e = model.sqrt_K * (
            model.drive[0]
            + model.coupling[0, 0] * local_e
            + model.coupling[0, 1] * local_i
            + model.stimulus_gain * stimulus
        )
        field_i = model.sqrt_K * (
            model.drive[1]
            + model.coupling[1, 0] * local_e
            + model.coupling[1, 1] * local_i
        )
        field_e = torch.clamp(field_e, -100, 100) + model.feedback_gain * feedback
        field_i = torch.clamp(field_i, -100, 100)
        expected_re = re + (model.dt / model.tau_e) * (-re + model.activation(field_e))
        expected_ri = ri + (model.dt / model.tau_i) * (-ri + model.activation(field_i))
        actual_re, actual_ri = model.step(state, stimulus)
        torch.testing.assert_close(actual_re, expected_re)
        torch.testing.assert_close(actual_ri, expected_ri)

    def test_non_spatial_model_has_no_trainable_parameters(self):
        model = NonSpatialWorkingMemoryReservoir(
            unit_count=25,
            dt=0.001,
            tau=0.005,
            recurrent_gain=1.5,
            stimulus_gain=1.0,
            feedback_gain=0.001,
            feedback_scale=0.1,
            init_scale=0.1,
            seed=3,
        )
        self.assertEqual(list(model.parameters()), [])
        state = model.initial_state(4)
        next_state = model.step(state, torch.zeros(25))
        self.assertTrue(torch.isfinite(next_state).all())

    def test_balanced_non_spatial_rows_and_field_limit(self):
        model = NonSpatialWorkingMemoryReservoir(
            unit_count=25, dt=0.001, tau=0.005, recurrent_gain=1.5,
            stimulus_gain=1.0, feedback_gain=0.0, feedback_scale=0.0,
            init_scale=0.1, seed=3, field_clip=10.0, balance_rows=True,
        )
        torch.testing.assert_close(
            model.recurrent.sum(dim=1), torch.zeros(25), atol=1e-6, rtol=0,
        )
        state = torch.full((25,), 1e6)
        next_state = model.step(state, torch.zeros(25))
        self.assertTrue(torch.isfinite(next_state).all())


if __name__ == "__main__":
    unittest.main()
