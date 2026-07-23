import unittest

from src.hayflow_teacher.dendritic_calibration import (
    DendriticCandidate,
    DendriticProtocolCalibrator,
    build_candidate_actions,
    candidate_from_mapping,
    evenly_spaced_offsets,
)


class DendriticCalibrationContractTest(unittest.TestCase):
    def test_offsets_are_deterministic_and_strictly_inside_step(self):
        offsets = evenly_spaced_offsets(8)
        self.assertEqual(offsets, evenly_spaced_offsets(8))
        self.assertEqual(offsets, sorted(offsets))
        self.assertTrue(all(0.0 <= value < 1.0 for value in offsets))
        with self.assertRaisesRegex(ValueError, "positive"):
            evenly_spaced_offsets(0)
        narrow = evenly_spaced_offsets(8, 0.2)
        self.assertLess(max(narrow) - min(narrow), max(offsets) - min(offsets))
        self.assertTrue(all(0.4 < value < 0.6 for value in narrow))
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            evenly_spaced_offsets(8, 1.0)

    def test_candidate_schedule_keeps_canonical_weights(self):
        candidate = DendriticCandidate(
            "nexus_nmda",
            "nexus",
            ("nmda_spike",),
            synapse_count=3,
            burst_count=2,
            events_per_synapse_per_burst=2,
            burst_start_ms=3,
        )
        actions = build_candidate_actions(
            candidate, [10, 12, 14], duration_ms=20
        )
        self.assertEqual(sorted(actions), [3, 4])
        self.assertTrue(all(len(rows) == 6 for rows in actions.values()))
        self.assertTrue(
            all(
                action.weight_multiplier == 1.0
                for rows in actions.values()
                for action in rows
            )
        )
        self.assertTrue(
            all(
                [action.offset_ms for action in rows]
                == sorted(action.offset_ms for action in rows)
                for rows in actions.values()
            )
        )

    def test_paired_schedule_has_one_current_per_macro_step(self):
        candidate = DendriticCandidate(
            "paired_ca",
            "hot_zone",
            ("calcium_spike",),
            synapse_count=2,
            burst_count=2,
            pair_with_somatic_spike=True,
        )
        actions = build_candidate_actions(
            candidate,
            [2, 4],
            duration_ms=20,
            somatic_current_na=3.0,
        )
        for step in (2, 3):
            currents = [
                action
                for action in actions[step]
                if action.kind == "somatic_current"
            ]
            self.assertEqual(len(currents), 1)
            self.assertEqual(currents[0].amplitude_na, 3.0)

    def test_tree_distance_uses_lowest_common_ancestor(self):
        parents = {0: None, 1: 0, 2: 1, 3: 1, 4: 3}
        distances = {0: 0.0, 1: 10.0, 2: 20.0, 3: 25.0, 4: 40.0}
        self.assertEqual(
            DendriticProtocolCalibrator.tree_distance_um(
                2, 4, parents, distances
            ),
            40.0,
        )
        self.assertEqual(
            DendriticProtocolCalibrator.tree_distance_um(
                3, 4, parents, distances
            ),
            15.0,
        )

    def test_compact_cluster_does_not_mix_distant_branches(self):
        parents = {0: None, 1: 0, 2: 1, 3: 2, 4: 1, 5: 4, 6: 5}
        distances = {
            0: 0.0,
            1: 10.0,
            2: 20.0,
            3: 30.0,
            4: 30.0,
            5: 50.0,
            6: 70.0,
        }
        candidates = [(2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]
        center, selected = DendriticProtocolCalibrator.compact_synapse_cluster(
            candidates,
            3,
            parents,
            distances,
            target_segment_id=1,
            maximum_center_distance_um=45.0,
        )
        self.assertIn(center, {4, 5, 6})
        self.assertEqual({row[2] for row in selected}, {40, 50, 60})
        self.assertLessEqual(max(row[0] for row in selected), 40.0)

    def test_mapping_builds_versioned_candidate(self):
        family = {
            "target": "tuft",
            "required_event_kinds": ["nmda_plateau"],
            "burst_start_ms": 4,
            "burst_interval_ms": 2,
            "pair_with_somatic_spike": False,
            "maximum_tree_distance_um": 250.0,
            "selection_mode": "branch_cluster",
            "event_probe_mode": "cluster_center",
            "event_probe_kinds": ["nmda_spike", "nmda_plateau"],
        }
        level = {
            "synapse_count": 32,
            "burst_count": 4,
            "events_per_synapse_per_burst": 2,
            "event_window_ms": 0.25,
        }
        candidate = candidate_from_mapping("tuft_plateau", family, level)
        self.assertEqual(candidate.target, "tuft")
        self.assertEqual(candidate.event_cost, 256)
        self.assertEqual(candidate.maximum_tree_distance_um, 250.0)
        self.assertEqual(candidate.selection_mode, "branch_cluster")
        self.assertEqual(candidate.event_probe_mode, "cluster_center")
        self.assertEqual(candidate.event_window_ms, 0.25)
        self.assertIn("n32-b4-r2", candidate.candidate_id)
        self.assertIn("branch-w250", candidate.candidate_id)

    def test_required_and_forbidden_events_cannot_overlap(self):
        candidate = DendriticCandidate(
            "invalid",
            "nexus",
            ("nmda_spike",),
            synapse_count=8,
            burst_count=1,
            forbidden_event_kinds=("nmda_spike",),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            candidate.validate(20)


if __name__ == "__main__":
    unittest.main()
