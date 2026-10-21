import json
import unittest
from pathlib import Path
import numpy as np
from src.giada_teacher.task37_ionic_playground import event_metrics

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'experiments/results/task41_audited_v1'

class Task41Audit(unittest.TestCase):
    def read(self,n):return json.loads((D/n).read_text())

    def test_secondary_crossing_is_not_silently_filtered(self):
        reference=np.array([0.,0.,1.,.5,0.,0.,.099,0.])
        prediction=reference.copy();prediction[6]=.101
        result=event_metrics(reference,prediction,.1,1e-7)
        self.assertAlmostEqual(result['offset_error_ms'],.3)
        self.assertEqual(result['onset_error_ms'],0)

    def test_original_no_go_preserved(self):
        a=self.read('audit.json')
        self.assertTrue(a['valid']);self.assertTrue(a['original_scientific_NO_GO_preserved'])
        self.assertFalse(a['task42_authorized']);self.assertFalse(a['thresholds_changed'])
        self.assertEqual(a['metrics_recomputed'],3492)
        self.assertTrue(a['checkpoint_bitwise_equal_to_frozen_B6'])

    def test_precise_localization(self):
        rows=self.read('stress_single_gate_oracles.json')['rows']
        for r in rows:
            if (r['channel'],r['coefficient'],r['gate']) in [('Nap_Et2','u',8),('Im','z',11)]:
                self.assertLessEqual(r['event']['metrics']['offset_error_ms'],.3)
        n=self.read('nats2_coefficient_interventions.json')['interventions']
        self.assertEqual(n['oracle_z_gate7']['17']['metrics']['offset_error_ms'],0)
        self.assertGreater(n['oracle_z_gate6']['17']['metrics']['offset_error_ms'],.3)

    def test_numerical_control_and_scope(self):
        a=self.read('audit.json');v=a['stress_probe_passes']
        self.assertEqual(v['learned32'],v['learned32coeff_state64'])
        self.assertEqual(v['formula_torch32'],36)
        self.assertEqual(v['three_local_coefficient_oracles'],36)
        s=self.read('synthesis.json')
        self.assertEqual(s['next_experiment']['status'],'proposal_not_preregistered_not_launched')
        self.assertEqual(np.prod([len(v) for v in s['next_experiment']['factorial_axes'].values()]),8)
        self.assertFalse(s['training_performed'])

if __name__=='__main__':unittest.main()
