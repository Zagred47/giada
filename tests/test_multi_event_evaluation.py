import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
from src.giada_teacher.multi_event_evaluation import evaluate_events,amplitude_pass,original_event_pass,event_metrics
Q=json.loads((ROOT/'experiments/task47_preregistration_v1.json').read_text())['quality']

def fixture():
    r=np.zeros(300,dtype=np.float64);r[30:61]=-np.r_[np.linspace(.12,1.,16),np.linspace(.95,.12,15)];r[140:171]=-np.r_[np.linspace(.12,.6,16),np.linspace(.55,.12,15)]
    return r
def shift(r,n):
    out=np.full_like(r,r[0]);out[n:]=r[:-n];return out

class MultiEvent(unittest.TestCase):
    def test_identity_flat_and_plateau(self):
        for r in (fixture(),np.zeros(300),np.ones(300)*-.02):self.assertTrue(evaluate_events(r,r,.1,Q)['passed'])
        r=np.zeros(300);r[30:61]=-1.;r[140:171]=-.5;self.assertTrue(evaluate_events(r,r,.1,Q)['passed'])
    def test_delay_boundaries_and_genuine_delays(self):
        r=fixture()
        self.assertTrue(evaluate_events(r,shift(r,2),.1,Q)['passed'])
        self.assertFalse(evaluate_events(r,shift(r,3),.1,Q)['passed'])
        v=evaluate_events(r,shift(r,10),.1,Q);self.assertEqual(v['status'],'timing_failure');self.assertAlmostEqual(v['max_peak_error_ms'],1.)
    def test_missing_and_spurious(self):
        r=fixture();pred=r.copy();pred[140:171]=0.;v=evaluate_events(r,pred,.1,Q);self.assertFalse(v['passed']);self.assertEqual(v['missing_events'],1)
        pred=r.copy();pred[220]=-.5;v=evaluate_events(r,pred,.1,Q);self.assertFalse(v['passed']);self.assertEqual(v['spurious_events'],1)
    def test_split_and_merge_are_not_silently_accepted(self):
        r=fixture();pred=r.copy();pred[44:46]=0.;v=evaluate_events(r,pred,.1,Q);self.assertFalse(v['passed']);self.assertEqual(v['spurious_events'],1)
        pred=r.copy();pred[61:140]=-.2;v=evaluate_events(r,pred,.1,Q);self.assertFalse(v['passed']);self.assertEqual(v['missing_events'],1)
    def test_amplitude_and_sign_are_not_normalized_away(self):
        r=fixture()
        for p in (1.5*r,-r,np.zeros_like(r)):self.assertFalse(amplitude_pass(r,p,.1,Q))
        self.assertFalse(evaluate_events(np.zeros(300),r,.1,Q)['passed'])
    def test_ordered_matching_rejects_displaced_equal_count_event(self):
        r=fixture();p=r.copy();p[140:171]=0.;p[220:251]=r[140:171]
        v=evaluate_events(r,p,.1,Q);self.assertFalse(v['passed']);self.assertGreater(v['max_peak_error_ms'],Q['peak_timing_max_ms'])
    def test_borderline_single_sample_is_retained_as_missing(self):
        r=fixture();r[220]=-.100005;p=r.copy();p[220]=-.099995
        old=event_metrics(r,p,.1,1e-7);new=evaluate_events(r,p,.1,Q)
        self.assertGreater(old['offset_error_ms'],Q['onset_offset_max_ms']);self.assertFalse(new['passed']);self.assertEqual(new['missing_events'],1);self.assertEqual(new['max_offset_error_ms'],0.)
    def test_right_boundary_is_explicit_and_identity_passes(self):
        r=fixture();r[160:]=-.4;v=evaluate_events(r,r,.1,Q);self.assertTrue(v['passed']);self.assertGreater(v['censored_offsets'],0)
    def test_nonfinite_prediction_fails(self):
        r=fixture();p=r.copy();p[10]=np.nan;self.assertFalse(evaluate_events(r,p,.1,Q)['passed'])
    def test_saved_case_not_waived(self):
        path=ROOT/'experiments/results/task47_cahva_audit_v1/case8_node4_evidence.npz'
        if not path.exists():self.skipTest('Local saved trace evidence')
        with np.load(path) as n:
            for mode in ('imposed','autonomous'):
                r=n['reference_current'][:,0];p=n[mode+'_current'][:,0,2,0];v=evaluate_events(r,p,.1,Q)
                self.assertFalse(v['passed']);self.assertEqual(v['missing_events'],1);self.assertEqual(v['max_peak_error_ms'],0.);self.assertEqual(v['max_onset_error_ms'],0.);self.assertEqual(v['max_offset_error_ms'],0.)
if __name__=='__main__':unittest.main()
