"""Immutable evidence and non-promotion checks for the complete41f audit."""
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class Task41fAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=json.loads((ROOT/'artifacts/task41f_reserve_external/final_detailed_audit.json').read_text())
        cls.s=json.loads((ROOT/'experiments/results/task41f_audited_v1/synthesis.json').read_text(encoding='utf-8'))
    def test_complete_backup(self):
        b=self.a['backup']
        self.assertTrue(b['valid'] and b['runtime_independent'] and b['byte_identical_to_remote'] and b['crc_all_members_valid'])
        self.assertEqual(b['bytes'],1928258234)
        self.assertEqual(b['sha256'],'4383d52aad6106e10bba17e0d91e146237a590ffed9e0987637b6923c85fc82a')
        self.assertTrue(all(b['parameters_finite'].values()))
        self.assertEqual(b['all_training_updates_completed'],60000)
    def test_no_promotion(self):
        self.assertEqual(self.s['final_decision'],'PRECISION_RELAXATION_NO_GO')
        self.assertFalse(self.s['thresholds_changed'])
        self.assertFalse(self.s['next_experiment']['launched'])
        self.assertFalse(self.s['next_experiment']['task42_authorized'])
        self.assertEqual(self.a['nominee'],'UZ64_broad_point')
    def test_precision_vs_same_official_reference(self):
        self.assertTrue(self.a['reproduction']['valid'])
        for row in self.a['precision']:
            n=row['counts']['official_FP32']['passed']
            self.assertEqual(row['counts']['official_FP32']['total'],6)
            if row['policy']=='FP32':self.assertIn(n,(4,5))
            else:self.assertEqual(n,6)
    def test_equilibrium_not_relaxation_oracle(self):
        rows={(r['model'],r['oracle']):r for r in self.a['coefficient_oracles']}
        for name in ('P110','E_Q0','E_H10'):
            self.assertEqual(rows[name,'u_m']['passed'],18)
            self.assertEqual(rows[name,'z_m']['passed'],rows[name,'learned']['passed'])
            self.assertLess(rows[name,'learned']['passed'],18)
    def test_selection_and_contrasts(self):
        self.assertTrue(self.a['selection_predicates_consistent'])
        self.assertEqual(self.a['selection']['chosen_steps'].count(0),3)
        self.assertEqual(len(self.a['contrasts']),20)
        self.assertEqual(sum(r['registered_contrast_passed'] for r in self.a['contrasts']),4)
    def test_regressions_are_not_hidden_by_counts(self):
        q=self.s['summaries']['E_Q0'];f=self.s['summaries']['UZ64_broad_point']
        self.assertEqual(q['stress']['41']['passed'],36)
        self.assertEqual(f['stress']['41']['passed'],35)
        self.assertEqual(q['stress']['41c']['passed'],33)
        self.assertEqual(f['stress']['41c']['passed'],32)
        for baseline in ('P110','E_Q0','E_H10'):
            self.assertIn('UZ64_broad_point',self.a['regressions'][baseline])

if __name__=='__main__':unittest.main()
