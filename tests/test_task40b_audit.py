import json,unittest
from pathlib import Path
F=Path(__file__).resolve().parents[1]/'experiments/results/task40b_audited_v1'
class AuditTests(unittest.TestCase):
    def read(self,n):return json.loads((F/n).read_text())
    def test_full_scope(self):
        a=self.read('audit.json');self.assertTrue(a['valid']);self.assertFalse(a['thresholds_changed'])
        self.assertEqual(a['autonomous_interventions_recomputed'],1344)
        self.assertEqual(a['autonomous_trajectories_recomputed'],2112)
        self.assertEqual(a['cpu_B6_passed'],dict(development=72,stress_diagnostic=72))
        self.assertFalse(a['imposed_trace_recomputed'])
    def test_counts_and_nomination(self):
        a=self.read('audit.json');s=a['summaries']
        self.assertEqual(s['B0']['autonomous']['differential_passed'],146)
        self.assertEqual(s['B7']['autonomous']['differential_passed'],165)
        for name in ('B1','B2','B3','B4','B5','B6'):self.assertEqual(s[name]['autonomous']['differential_passed'],168)
        self.assertEqual(s['B1']['stress_diagnostic']['passed'],71)
        self.assertEqual(s['B6']['stress_diagnostic']['passed'],72)
        self.assertEqual(a['preferred_arm'],'B6');self.assertTrue(a['task41_preparation_authorized'])
        self.assertFalse(a['independent_confirmation']);self.assertFalse(a['task41_executed'])
    def test_ablation_does_not_promote_oracle(self):
        a=self.read('stress_ablation.json');self.assertFalse(a['candidate_eligible']);self.assertFalse(a['training_performed'])
        for rows in a['results'].values():self.assertEqual(sum(r['metrics']['passed'] for r in rows),11)
    def test_saved_case_completeness(self):
        rr=self.read('causal_results.json')
        self.assertEqual(sum(len(v['cases']) for v in rr.values()),4224)
        self.assertEqual(sum(len(v['interventions']) for v in rr.values()),2688)
        self.assertEqual(len(self.read('recurrence_evaluation.json')),1296)
if __name__=='__main__':unittest.main()
