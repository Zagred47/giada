"""Regression checks for post-hoc Task40 interpretation; no training or DB writes."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
F=ROOT/'experiments/results/task40_audited_v1'
class AuditTests(unittest.TestCase):
    def read(self,n):return json.loads((F/n).read_text())
    def test_unchanged_contract(self):
        self.assertEqual(self.read('preregistration.json'),json.loads((ROOT/'experiments/task40_preregistration_v1.json').read_text()))
        self.assertFalse(self.read('final_report.json')['task41_preparation_authorized'])
    def test_all_original_cases_preserved(self):
        rows=self.read('causal_results.json')
        self.assertEqual(sum(len(v['cases']) for v in rows.values()),6336)
        self.assertEqual(sum(len(v.get('interventions',[])) for v in rows.values()),672)
        self.assertEqual(rows['J1/all/autonomous']['passed_cases'],264)
    def test_failure_localization(self):
        rows=[r for r in self.read('failure_details.json') if r['model']=='J1' and r['mode']=='autonomous']
        self.assertEqual(len(rows),22)
        self.assertEqual(sum(c['channel']=='Ca_LVAst' for r in rows for c in r['channels']),18)
        self.assertTrue(all('nrmse' in c['failed_metrics'] for r in rows for c in r['channels']))
    def test_numerical_floor_and_replay(self):
        self.assertTrue(all(r['metrics']['passed'] for r in self.read('differential_formula_floor.json')))
        a=self.read('audit.json');self.assertEqual(a['recomputed_interventions'],672)
        self.assertTrue(a['subtraction64_verdicts_identical'])
        self.assertTrue(self.read('attribution_audit.json')['cpu_gpu_168_flags_equal'])
    def test_scoped_oracles(self):
        a=self.read('attribution_audit.json')
        self.assertEqual(a['coefficient_controls_failures'],dict(learned=22,oracle_u=22,oracle_z=0,oracle_m_z=3,oracle_u_z=0))
        self.assertEqual(a['autonomous_failures']['all_except_calcium'],0)
        self.assertTrue(a['no_candidate_promotion'])
        self.assertFalse(a['thresholds_changed'])
if __name__=='__main__':unittest.main()
