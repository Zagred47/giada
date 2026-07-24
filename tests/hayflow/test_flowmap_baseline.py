import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.hayflow_data import (
    FlowmapContractError,
    StateNormalizer,
    batch_iterator,
    prepare_flowmap_bundle,
)
from src.hayflow_eval import (
    binary_event_metric_rows,
    decide_go_no_go,
    rollout_metric_row,
    state_metric_rows,
)
from src.hayflow_model import (
    DualRidgeBaseline,
    FlowmapExperimentConfig,
    ridge_design_matrix,
    structured_arrays,
)


class FakeLayout:
    def __init__(self):
        self.np = np
        self.state_width = 4
        self.core_records = [
            {
                "category": "voltage",
                "mechanism": "neuron",
                "variable": "v",
            },
            {
                "category": "mechanism_states",
                "mechanism": "Na",
                "variable": "m",
            },
            {
                "category": "calcium_ions",
                "mechanism": "ion",
                "variable": "cai",
            },
            {
                "category": "synapse_states",
                "mechanism": "NetCon",
                "variable": "u",
            },
        ]


class FlowmapBaselineTest(unittest.TestCase):
    def _bundle_root(self, root: Path, schema: str) -> Path:
        dataset = root / "diagnostic_dataset_v1"
        dataset.mkdir()
        manifest = {
            "schema_version": schema,
            "teacher_commit": "074c4666300a8ad246601dab179a97a6942f0f29",
            "transition_count": 1224,
            "transition_store": "transition_dataset.h5",
        }
        state_schema = {
            "schema_version": schema,
            "core_state_width": 17220,
            "privileged_state_width": 9182,
        }
        validation = {
            "schema_version": schema,
            "valid": True,
            "blockers": [],
        }
        for name, payload in (
            ("dataset_manifest.json", manifest),
            ("state_schema.json", state_schema),
            ("validation_report.json", validation),
            ("manifest.json", {"segments": [], "synapses": []}),
            ("artifact_index.json", {"artifacts": []}),
        ):
            (dataset / name).write_text(json.dumps(payload), encoding="utf-8")
        (dataset / "transition_dataset.h5").write_bytes(b"placeholder")
        return dataset

    def test_bundle_rejects_schema_1_0_0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._bundle_root(Path(directory), "1.0.0")
            with self.assertRaisesRegex(FlowmapContractError, "accepts only"):
                prepare_flowmap_bundle(root)

    def test_bundle_accepts_only_green_1_0_1_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._bundle_root(Path(directory), "1.0.1")
            bundle = prepare_flowmap_bundle(root)
            self.assertEqual(bundle.schema_version, "1.0.1")
            self.assertTrue(bundle.artifact_validation["valid"])

    def test_semantic_normalizer_round_trip(self):
        layout = FakeLayout()
        state_t = np.asarray(
            [[-70.0, 0.2, 1e-4, 0.3], [-65.0, 0.4, 2e-4, 0.5]]
        )
        state_t1 = np.asarray(
            [[-69.0, 0.25, 1.1e-4, 0.35], [-63.0, 0.45, 2.2e-4, 0.55]]
        )
        normalizer = StateNormalizer(layout).fit(state_t, state_t1)
        delta = normalizer.normalize_delta(state_t, state_t1)
        reconstructed = normalizer.reconstruct(state_t, delta)
        np.testing.assert_allclose(reconstructed, state_t1, atol=1e-10)
        self.assertEqual(normalizer.transform_codes.tolist(), [0, 2, 1, 2])

    def test_batch_iterator_is_seeded_and_complete(self):
        first = list(
            batch_iterator(range(10), batch_size=3, seed=17, shuffle=True)
        )
        second = list(
            batch_iterator(range(10), batch_size=3, seed=17, shuffle=True)
        )
        self.assertEqual([row.tolist() for row in first], [row.tolist() for row in second])
        self.assertEqual(sorted(np.concatenate(first).tolist()), list(range(10)))

    def test_dual_ridge_fits_multioutput_delta(self):
        x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        y = np.concatenate([2.0 * x + 1.0, -x], axis=1)
        model = DualRidgeBaseline(alpha=1e-8).fit(x, y)
        np.testing.assert_allclose(model.predict(x), y, atol=1e-5)

    def test_ridge_design_distinguishes_u1_and_u2(self):
        batch = {
            "state_t": np.zeros((2, 4), dtype=np.float32),
            "u1": np.ones((2, 3, 2), dtype=np.float32),
            "u2_features": np.ones((2, 2, 5), dtype=np.float32),
            "u2_segment_ids": np.asarray([[1, 2], [0, 0]]),
            "u2_mask": np.asarray([[True, True], [False, False]]),
        }
        u1 = ridge_design_matrix(
            batch,
            voltage_width=2,
            state_mode="voltage_only",
            input_encoding="U1",
        )
        u2 = ridge_design_matrix(
            batch,
            voltage_width=2,
            state_mode="voltage_only",
            input_encoding="U2",
            maximum_events=3,
        )
        self.assertEqual(u1.shape, (2, 8))
        self.assertEqual(u2.shape, (2, 23))

    def test_smoke_profile_is_not_silently_full(self):
        config = FlowmapExperimentConfig(profile="smoke")
        effective = config.effective()
        self.assertEqual(effective.initialization_seeds, (17,))
        self.assertEqual(effective.maximum_epochs, 2)
        self.assertEqual(effective.rollout_horizons_ms, (2,))

    def test_go_no_go_is_conservative(self):
        no_go = decide_go_no_go(
            {
                "b3_test_voltage_rmse": 2.0,
                "persistence_test_voltage_rmse": 1.0,
                "affine_test_voltage_rmse": 1.5,
                "b3_rollout_16ms_bounded": True,
                "b3_macro_event_f1": 0.9,
                "full_state_gain_fraction": 0.2,
                "privileged_gain_fraction": 0.1,
            }
        )
        self.assertEqual(no_go["decision"], "NO_GO")

    def test_event_metrics_report_masked_region_accuracy(self):
        probabilities = np.zeros((2, 6), dtype=np.float32)
        targets = np.zeros_like(probabilities)
        probabilities[:, 0] = (0.9, 0.1)
        targets[0, 0] = 1.0
        region_prediction = np.zeros((2, 6), dtype=np.int64)
        region_target = np.zeros_like(region_prediction)
        region_target[0, 0] = 2
        region_prediction[0, 0] = 2
        region_mask = np.zeros_like(region_prediction, dtype=bool)
        region_mask[0, 0] = True
        rows = binary_event_metric_rows(
            probabilities,
            targets,
            timing_prediction=None,
            timing_target=None,
            timing_mask=None,
            region_prediction=region_prediction,
            region_target=region_target,
            region_mask=region_mask,
            model_name="test",
            split="validation",
        )
        self.assertEqual(rows[0]["region_support"], 1)
        self.assertEqual(rows[0]["region_accuracy"], 1.0)

    def test_rollout_metrics_report_physical_domain_violations(self):
        layout = FakeLayout()
        prediction = np.asarray([[-70.0, 1.2, -0.1, -0.2]])
        target = np.asarray([[-70.0, 0.8, 0.1, 0.2]])
        row = rollout_metric_row(
            prediction,
            target,
            model_name="test",
            split="test",
            horizon_ms=2,
            voltage_width=1,
            layout=layout,
        )
        self.assertGreater(row["outside_domain_fraction"], 0.0)

    def test_structured_child_padding_preserves_leaf_identity(self):
        layout = type(
            "Layout",
            (),
            {
                "children": [[1], []],
                "segment_count": 2,
                "core_segment_ids": np.asarray([0, 1]),
                "core_category_ids": np.asarray([0, 0]),
                "core_mechanism_ids": np.asarray([0, 0]),
                "core_variable_ids": np.asarray([0, 0]),
                "core_kind_ids": np.asarray([0, 0]),
                "privileged_segment_ids": np.asarray([0]),
                "privileged_mechanism_ids": np.asarray([0]),
                "privileged_variable_ids": np.asarray([0]),
                "privileged_kind_ids": np.asarray([0]),
                "segment_region_ids": np.asarray([0, 0]),
                "segment_static": np.zeros((2, 7), dtype=np.float32),
                "parent_ids": np.asarray([0, 0]),
            },
        )()
        arrays = structured_arrays(layout)
        self.assertEqual(arrays["child_ids"].tolist(), [[1], [1]])
        self.assertEqual(arrays["child_mask"].tolist(), [[1.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
