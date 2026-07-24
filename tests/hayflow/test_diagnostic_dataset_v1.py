import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.hayflow_data import ProtocolTrajectory
from src.hayflow_teacher import (
    DiagnosticDatasetV1Session,
    actions_from_selected_protocol,
    canonical_json_sha256,
    filter_synaptic_actions,
    validate_calibration_artifacts,
)
from src.hayflow_teacher.audit import sha256_file
from src.hayflow_teacher.diagnostic_dataset_v1 import (
    CALCIUM_PROTOCOL_ID,
    PLATEAU_PROTOCOL_ID,
    _summarize_negative_control_outcomes,
)


class DiagnosticDatasetV1ContractTest(unittest.TestCase):
    def test_every_declared_negative_must_suppress_its_target_event(self):
        outcomes = [
            {
                "control_of": PLATEAU_PROTOCOL_ID,
                "target_event_present": False,
            },
            {
                "control_of": PLATEAU_PROTOCOL_ID,
                "target_event_present": True,
            },
            {
                "control_of": CALCIUM_PROTOCOL_ID,
                "target_event_present": False,
            },
        ]
        summary = _summarize_negative_control_outcomes(outcomes)
        self.assertTrue(all(summary["family_coverage"].values()))
        self.assertFalse(summary["all_declared_suppress_target_event"])
        self.assertFalse(summary["valid"])

        corrected = _summarize_negative_control_outcomes(
            [row for row in outcomes if not row["target_event_present"]]
        )
        self.assertTrue(corrected["all_declared_suppress_target_event"])
        self.assertTrue(corrected["valid"])

    def test_prefix_comparison_keeps_event_probe_distinct_from_tuft_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            np.savez_compressed(
                reference,
                time_ms=np.asarray([100.0, 100.025]),
                voltage_tuft_cluster_center_mv=np.asarray([-70.0, -69.0]),
                voltage_event_probe_mv=np.asarray([-60.0, -58.0]),
            )
            session = object.__new__(DiagnosticDatasetV1Session)
            session.np = np
            session.output_dir = root
            session.calibration_root = root
            session.reference_trace_by_protocol_seed = {
                ("calcium", 1): reference
            }
            session.audit = type(
                "Audit",
                (),
                {"representatives": {"tuft_cluster_center": 460}},
            )()
            trajectory = ProtocolTrajectory(
                trajectory_id="calcium-seed1",
                category="dendritic_events",
                protocol="calcium",
                protocol_id="calcium",
                seed=1,
                duration_ms=1,
                split="train",
            )
            comparison = session._compare_reference_prefix(
                trajectory,
                [100.0, 100.025],
                {
                    "tuft_cluster_center": [-70.0, -69.0],
                    "voltage_event_probe_mv": [-60.0, -58.0],
                },
                0.025,
                reference_path=reference,
                reference_kind="test",
            )
            self.assertTrue(comparison["valid"])
            self.assertEqual(comparison["maximum_absolute_error"], 0.0)

    def test_generation_requires_a_green_preflight(self):
        session = object.__new__(DiagnosticDatasetV1Session)
        session.preflight_report = {}
        protocol = ProtocolTrajectory(
            trajectory_id="blocked",
            category="rest_subthreshold",
            protocol="rest",
            seed=1,
            duration_ms=1,
            split="train",
        )
        with self.assertRaisesRegex(RuntimeError, "run_v1_preflight"):
            session.generate_dataset([protocol])

    def test_preflight_hash_binds_the_exact_protocol_plan(self):
        first = ProtocolTrajectory(
            trajectory_id="first",
            category="rest_subthreshold",
            protocol="rest",
            seed=1,
            duration_ms=1,
            split="train",
        )
        second = ProtocolTrajectory(
            trajectory_id="second",
            category="rest_subthreshold",
            protocol="rest",
            seed=2,
            duration_ms=2,
            split="validation",
        )
        digest = DiagnosticDatasetV1Session._protocol_plan_sha256(
            [first, second]
        )
        self.assertEqual(
            digest,
            DiagnosticDatasetV1Session._protocol_plan_sha256([second, first]),
        )
        changed = ProtocolTrajectory(
            trajectory_id="second",
            category="rest_subthreshold",
            protocol="rest",
            seed=2,
            duration_ms=3,
            split="validation",
        )
        self.assertNotEqual(
            digest,
            DiagnosticDatasetV1Session._protocol_plan_sha256([first, changed]),
        )
        session = object.__new__(DiagnosticDatasetV1Session)
        session.preflight_report = {
            "valid": True,
            "protocol_plan_sha256": digest,
        }
        with self.assertRaisesRegex(RuntimeError, "protocol plan changed"):
            session.generate_dataset([first, changed])

    def test_extended_splits_keep_whole_trajectory_contract(self):
        for split in (
            "deterministic_test",
            "event_boundary_test",
            "branching_test",
        ):
            ProtocolTrajectory(
                trajectory_id=f"trajectory-{split}",
                category="branching" if split == "branching_test" else "dendritic_events",
                protocol="protocol",
                seed=1,
                duration_ms=80,
                split=split,
                stimulus_onset_step=3,
            ).validate()

    def test_selected_schedule_is_reconstructed_without_weight_changes(self):
        protocol = {
            "input_schedule_template": {
                "3": [
                    {
                        "kind": "synaptic_event",
                        "offset_ms": 0.25,
                        "synapse_id": 10,
                        "weight_multiplier": 1.0,
                        "metadata": "",
                    },
                    {
                        "kind": "synaptic_event",
                        "offset_ms": 0.75,
                        "synapse_id": 12,
                        "weight_multiplier": 1.0,
                    },
                ]
            }
        }
        actions = actions_from_selected_protocol(protocol)
        self.assertEqual([row.synapse_id for row in actions[3]], [10, 12])
        self.assertTrue(all(row.weight_multiplier == 1.0 for row in actions[3]))

        filtered = filter_synaptic_actions(actions, [12])
        self.assertEqual([row.synapse_id for row in filtered[3]], [12])
        self.assertEqual(filtered[3][0].offset_ms, 0.75)

    def test_configuration_hash_is_order_independent(self):
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": [2, 3]}),
            canonical_json_sha256({"b": [2, 3], "a": 1}),
        )

    def test_calibration_artifact_index_checks_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "report.json"
            artifact.write_text('{"valid": true}\n', encoding="utf-8")
            index = {
                "schema_version": "0.3.0",
                "artifacts": [
                    {
                        "path": artifact.name,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                ],
            }
            (root / "artifact_index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            valid = validate_calibration_artifacts(root)
            self.assertTrue(valid["valid"])
            self.assertFalse(valid["artifact_count_matches_reference"])

            artifact.write_text("changed\n", encoding="utf-8")
            changed = validate_calibration_artifacts(root)
            self.assertFalse(changed["valid"])


if __name__ == "__main__":
    unittest.main()
