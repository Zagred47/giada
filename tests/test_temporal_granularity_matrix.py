import unittest

import numpy as np

from src.giada_teacher.temporal_granularity_matrix import (
    EVENT_SLOTS_PER_BIN, FEATURE_WIDTH, SUBSTEPS, _binary_f1, _substep_input,
    realized_schedule, select_rows,
    trajectory_role,
)


class TemporalGranularityMatrixTests(unittest.TestCase):
    def test_frozen_selection_deduplicates_sites(self):
        rows = [{"transition_index": 8, "trajectory_id": "a"},
                {"transition_index": 3, "trajectory_id": "b"},
                {"transition_index": 8, "trajectory_id": "a"}]
        self.assertEqual([r["transition_index"] for r in select_rows(rows)], [3, 8])
        self.assertEqual(trajectory_role("a"), trajectory_role("a"))

    def test_realized_schedule_keeps_temporal_location(self):
        rows = [{"kind": "synaptic_event", "offset_ms": .125,
                 "synapse_id": 4, "released_quantity": 2.0},
                {"kind": "somatic_current", "offset_ms": .25,
                 "duration_ms": .25, "amplitude_na": .1}]
        schedule = realized_schedule(rows, {4: 10})
        self.assertEqual(schedule.shape, (8, FEATURE_WIDTH))
        self.assertAlmostEqual(schedule[1, 0], 11 / 643)
        self.assertAlmostEqual(schedule[1, 1], .125)
        self.assertAlmostEqual(schedule[1, 2], 2)
        self.assertAlmostEqual(schedule[1, 3], 5 / 10000)
        self.assertAlmostEqual(schedule[2, -4], .1)
        self.assertAlmostEqual(schedule[3, -4], .1)
        for n in SUBSTEPS:
            width = 8 // n
            reconstructed = sum((_substep_input(schedule, i * width, width)
                                 for i in range(n)), np.zeros(schedule.size, dtype=np.float32))
            np.testing.assert_array_equal(reconstructed, schedule.reshape(-1))

    def test_event_overflow_fails_instead_of_dropping_timestamps(self):
        rows = [{"kind": "synaptic_event", "offset_ms": .1,
                 "synapse_id": 4, "released_quantity": 1.0}
                for _ in range(EVENT_SLOTS_PER_BIN + 1)]
        with self.assertRaisesRegex(RuntimeError, "event-token capacity"):
            realized_schedule(rows, {4: 10})

    def test_boundary_proxy_f1(self):
        self.assertAlmostEqual(_binary_f1([1, 0, 1], [1, 1, 0]), .5)


if __name__ == "__main__":
    unittest.main()
