import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_task47_cahva import describe
from finalize_task47_cahva_audit import local_peaks
from src.giada_teacher.task37_ionic_playground import event_metrics

class CaHVAAudit(unittest.TestCase):
    def test_tiny_amplitude_difference_can_switch_global_island_without_delay(self):
        ref=np.zeros(400);ref[20:41]=1.;ref[300]=.100005;pred=ref.copy();pred[300]=.099995
        result=describe(ref,pred)
        self.assertEqual(result['active_mask_disagreement_indices'],[300]);self.assertEqual(result['original_event']['offset_error_ms'],26.)
        self.assertEqual(result['original_event']['peak_error_ms'],0.)
        peaks=local_peaks(ref,pred,np.array([10,290]));self.assertEqual(peaks[-1]['peak_error_ms'],0.)
        self.assertFalse(peaks[-1]['candidate_above_original_threshold']);self.assertTrue(peaks[-1]['reference_above_original_threshold'])
    def test_known_delay_is_not_excused_by_local_peak_probe(self):
        ref=np.zeros(400);ref[20]=1.;ref[300]=.2;pred=ref.copy();pred[300]=0.;pred[310]=.2
        self.assertEqual(local_peaks(ref,pred,np.array([10,290]))[-1]['peak_error_ms'],1.)
        self.assertEqual(event_metrics(ref,pred,.1,1e-7)['offset_error_ms'],1.)
    def test_amplitude_error_is_explicit_even_with_equal_peak_time(self):
        ref=np.zeros(400);ref[20]=1.;ref[300]=.2;pred=ref.copy();pred[300]=.1
        result=local_peaks(ref,pred,np.array([10,290]))[-1]
        self.assertEqual(result['peak_error_ms'],0.);self.assertEqual(result['candidate_relative_peak_amplitude_error'],.5)
    def test_saved_case_and_original_verdict_are_retained(self):
        folder=ROOT/'experiments/results/task47_cahva_audit_v1'
        if not (folder/'case8_node4_evidence.npz').exists():self.skipTest('Saved forensic evidence is local, not bundled with source')
        audit=json.loads((folder/'audit.json').read_text());self.assertFalse(audit['thresholds_changed']);self.assertFalse(audit['model_outputs_modified']);self.assertFalse(audit['evaluator_code_modified']);self.assertIn('NO_GO',audit['official_task47_verdict_unchanged'])
        with np.load(folder/'case8_node4_evidence.npz') as n:
            ref=n['reference_current'][:,0]
            for mode in ('imposed','autonomous'):
                result=describe(ref,n[mode+'_current'][:,0,2,0]);self.assertEqual(result['active_mask_disagreement_indices'],[1983]);self.assertAlmostEqual(result['original_event']['offset_error_ms'],44.9)
        self.assertTrue(all(r['event']['offset_error_ms']>44.8 for r in audit['precision_controls'] if r['seed']==17))
        self.assertTrue(all(r['event_after']['offset_error_ms']==0. for r in audit['single_sample_counterfactuals']))
if __name__=='__main__':unittest.main()
