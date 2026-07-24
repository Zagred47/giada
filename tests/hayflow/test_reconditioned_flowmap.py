import unittest

import numpy as np

from src.hayflow_data import (
    ReconditionedAuxiliaryNormalizer,
    ReconditionedStateNormalizer,
    ReconditioningConfig,
)
from src.hayflow_eval import binary_event_metric_rows
from src.hayflow_model import (
    ReconditionedExperimentConfig,
    ReconditionedRunSpec,
)


class FakeLayout:
    def __init__(self):
        self.state_width = 4
        self.category_slices = {
            "voltage": slice(0, 1),
            "mechanism_states": slice(1, 2),
            "calcium_ions": slice(2, 3),
            "synapse_states": slice(3, 4),
        }
        self.core_records = [
            self._record("voltage", "neuron", "v", 0),
            self._record("mechanism_states", "NaTa_t", "m", 0),
            self._record("calcium_ions", "ca_ion", "cai", 0),
            self._record("synapse_states", "ProbAMPANMDA2", "A_AMPA", 7),
        ]

    @staticmethod
    def _record(category, mechanism, variable, owner_id):
        return {
            "category": category,
            "scope": "segment" if category != "synapse_states" else "synapse",
            "owner_id": owner_id,
            "mechanism": mechanism,
            "variable": variable,
            "kind": "state",
        }


class ReconditionedFlowmapTest(unittest.TestCase):
    def _states(self):
        rows = 12
        state_t = np.zeros((rows, 4), dtype=np.float64)
        state_t[:, 0] = np.linspace(-76.0, -70.0, rows)
        state_t[:, 1] = np.linspace(0.2, 0.7, rows)
        state_t[:, 2] = 1e-4
        state_t[:, 3] = 0.2
        state_t1 = state_t.copy()
        state_t1[:, 0] += np.linspace(0.05, 0.2, rows)
        state_t1[:, 1] += 0.01
        state_t1[-2:, 2] += (2e-5, 4e-5)
        state_t1[-1, 3] += 5.0
        return state_t, state_t1

    def test_zero_inflated_scale_is_fit_on_active_values(self):
        layout = FakeLayout()
        state_t, state_t1 = self._states()
        normalizer = ReconditionedStateNormalizer(
            layout,
            ReconditioningConfig(
                sparse_update_fraction=0.25,
                minimum_scale=1e-4,
            ),
        ).fit(state_t, state_t1)
        normalized, active = normalizer.delta_and_activity(state_t, state_t1)
        self.assertTrue(normalizer.sparse_mask[3])
        self.assertEqual(int(active[:, 3].sum()), 1)
        np.testing.assert_allclose(normalized[~active[:, 3], 3], 0.0)
        self.assertGreaterEqual(normalizer.delta_scale[3], 1e-4)
        self.assertLess(float(np.max(np.abs(normalized[active]))), 20.0)
        synapse_row = normalizer.audit_rows[3]
        self.assertEqual(synapse_row["nonzero_count"], 1)
        self.assertAlmostEqual(synapse_row["zero_fraction"], 11.0 / 12.0)

    def test_hurdle_and_s0_preserve_inactive_synapse_state(self):
        layout = FakeLayout()
        state_t, state_t1 = self._states()
        normalizer = ReconditionedStateNormalizer(
            layout, ReconditioningConfig(sparse_update_fraction=0.25)
        ).fit(state_t, state_t1)
        normalized, _ = normalizer.delta_and_activity(state_t, state_t1)
        inactive_probability = np.zeros_like(normalized)
        hurdle = normalizer.reconstruct(
            state_t,
            normalized,
            activity_probability=inactive_probability,
            synapse_mode="hurdle",
        )
        excluded = normalizer.reconstruct(
            state_t,
            normalized,
            activity_probability=np.ones_like(normalized),
            synapse_mode="exclude",
        )
        np.testing.assert_allclose(hurdle[:, 3], state_t[:, 3])
        np.testing.assert_allclose(excluded[:, 3], state_t[:, 3])

    def test_privileged_normalization_ignores_non_applicable_values(self):
        values = np.asarray(
            [[np.nan, 100.0], [10.0, 102.0], [12.0, 104.0]],
            dtype=np.float64,
        )
        layout = {
            "currents_conductances_t_plus_1": slice(0, 1),
            "dense_auxiliary": slice(1, 2),
        }
        records = [
            {
                "mechanism": "Ca_HVA",
                "variable": "ica",
                "kind": "current",
            }
        ]
        normalizer = ReconditionedAuxiliaryNormalizer(1e-4).fit(
            values, layout, privileged_records=records
        )
        self.assertEqual(normalizer.center[0], 11.0)
        transformed, mask = normalizer.transform(values)
        self.assertFalse(mask[0, 0])
        self.assertEqual(transformed[0, 0], 0.0)
        self.assertTrue(np.isfinite(transformed).all())

    def test_event_metric_accepts_validation_threshold_per_class(self):
        probability = np.zeros((1, 6), dtype=np.float64)
        target = np.zeros_like(probability)
        probability[0, :2] = (0.6, 0.6)
        target[0, :2] = 1.0
        rows = binary_event_metric_rows(
            probability,
            target,
            timing_prediction=None,
            timing_target=None,
            timing_mask=None,
            model_name="test",
            split="validation",
            threshold=np.asarray([0.5, 0.7, 0.5, 0.5, 0.5, 0.5]),
        )
        self.assertEqual(rows[0]["true_positive"], 1)
        self.assertEqual(rows[1]["false_negative"], 1)

    def test_smoke_profile_and_run_identifiers_are_explicit(self):
        effective = ReconditionedExperimentConfig(profile="smoke").effective()
        self.assertEqual(effective.maximum_epochs, 2)
        self.assertEqual(effective.initialization_seeds, (17,))
        self.assertEqual(effective.rollout_horizons_ms, (2,))
        spec = ReconditionedRunSpec("U2", "hurdle", "P1b")
        self.assertIn("U2-hurdle-P1b", spec.identifier(17))
        with self.assertRaisesRegex(ValueError, "input encoding"):
            ReconditionedRunSpec("future", "hurdle", "P0").validate()


if __name__ == "__main__":
    unittest.main()
