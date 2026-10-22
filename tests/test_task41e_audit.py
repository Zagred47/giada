"""Checks on immutable 41e audit outputs; no scientific rerun."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class Task41eAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=ROOT/'experiments/results/task41e_audited_v1'
        cls.probe=json.loads((folder/'residual_probe.json').read_text(encoding='utf-8'))
        cls.synthesis=json.loads((folder/'synthesis.json').read_text(encoding='utf-8'))

    def test_no_promotion_or_threshold_change(self):
        self.assertEqual(self.synthesis['final_decision'],'TRANSFER_DIAGNOSIS_NO_GO')
        self.assertFalse(self.synthesis['thresholds_changed'])
        self.assertFalse(self.synthesis['next_experiment']['launched'])
        self.assertFalse(self.synthesis['next_experiment']['task42_authorized'])
        self.assertTrue(self.probe['original_freeze_unchanged'])

    def test_boundary_reranking_is_not_original_result(self):
        audit=self.probe['diagnostic_reranking']
        self.assertEqual(audit['changed_members'],5)
        self.assertEqual(audit['original_steps'][12:15],audit['evaluator_consistent_steps'][12:15])
        for row in self.probe['selection_predicate_disagreements']:
            for member in row['rows']:
                self.assertGreater(member['ranked_target_failures'],member['official_target_failures'])
                for event in member['boundary_events']:
                    self.assertTrue(event['official_passed'])
                    self.assertAlmostEqual(event['event']['offset_error_ms'],.3)

    def test_probe_reference_change_explains_counts(self):
        for row in self.probe['probe_reproduction']:
            passes=lambda key:sum(r['passed'] for r in row[key])
            self.assertEqual(passes('probe_saved_with_FP64_reference'),6)
            if row['precision']=='FP32':
                self.assertEqual(passes('original_full_FP32_reference'),passes('probe_rescored_with_original_full_FP32_reference'))
                self.assertLess(passes('probe_rescored_with_original_full_FP32_reference'),6)
            else:
                self.assertEqual(passes('probe_rescored_with_original_full_FP32_reference'),6)

    def test_reported_remaining_cases(self):
        s=self.synthesis['summaries']
        self.assertEqual(s['Q0']['panels']['41c']['autonomous']['responses'],334)
        self.assertEqual(s['H10']['panels']['41c']['autonomous']['responses'],335)
        self.assertEqual(s['H10']['stress']['41c']['passed'],33)
        self.assertEqual(s['H10']['development']['passed'],35)
        self.assertTrue(self.synthesis['backup']['valid'])
        self.assertEqual(self.synthesis['backup']['final_members_verified'],206)


if __name__=='__main__':unittest.main()
