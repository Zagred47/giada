"""Regression tests for distinguishing release decisions from analog state."""

import copy
import unittest

from src.giada_teacher.roadmap_task17db_solver_release_confirmation import (
    _release_audit, _release_pair,
)


class ReleaseDecisionTests(unittest.TestCase):
    def trial(self):
        outcome = {
            "transition_id": 170029000, "event_index": 0, "synapse_id": 5,
            "scheduled_time_ms": 900.1, "random123_seed": 170029,
            "random123_stream_id": 5, "random123_global_index": 0,
            "rng_sequence_before": 2.0, "rng_sequence_after": 3.0,
            "rng_preview_value": 0.3, "release_success": True,
            "release_probability": 0.5, "released_quantity": 0.8,
            "pre_synapse_state": {"A": 0.1},
        }
        return {"release_rows": [
            {"step": i, "verification": {"valid": True},
             "outcomes": [copy.deepcopy(outcome)] if i == 0 else []}
            for i in range(60)]}

    def test_continuous_synapse_change_does_not_fake_release_flip(self):
        first = self.trial()
        second = copy.deepcopy(first)
        second["release_rows"][0]["outcomes"][0]["pre_synapse_state"]["A"] = 0.2
        second["release_rows"][0]["outcomes"][0]["release_probability"] = 0.51
        pair = _release_pair(_release_audit(first), _release_audit(second))
        self.assertTrue(pair["discrete_match"])
        self.assertAlmostEqual(pair["max_release_probability_difference"], 0.01)

    def test_release_flip_is_detected(self):
        first = self.trial()
        second = copy.deepcopy(first)
        second["release_rows"][0]["outcomes"][0]["release_success"] = False
        pair = _release_pair(_release_audit(first), _release_audit(second))
        self.assertFalse(pair["discrete_match"])
        self.assertEqual(pair["mismatch_counts"]["release_success"], 1)


if __name__ == "__main__":
    unittest.main()
