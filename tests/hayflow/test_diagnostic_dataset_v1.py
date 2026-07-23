import json
import tempfile
import unittest
from pathlib import Path

from src.hayflow_data import ProtocolTrajectory
from src.hayflow_teacher import (
    actions_from_selected_protocol,
    canonical_json_sha256,
    filter_synaptic_actions,
    validate_calibration_artifacts,
)
from src.hayflow_teacher.audit import sha256_file


class DiagnosticDatasetV1ContractTest(unittest.TestCase):
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
