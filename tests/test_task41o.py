import unittest
import numpy as np
from src.giada_teacher.task41o_causal_localization import recurrence,ARMS

class CausalControls(unittest.TestCase):
    def test_oracle_identity_and_one_step_intervention(self):
        initial=np.array([[.2,.8]],dtype='float64');u=np.full((120,1,2),.6);z=np.full_like(u,.01)
        truth=recurrence(u,z,initial,np.zeros_like(u),'oracle_both')
        self.assertTrue(np.array_equal(recurrence(u,z,initial,truth,'oracle_both'),truth))
        biased=u+.01
        full=recurrence(biased,z,initial,truth,'learned64')
        local=recurrence(biased,z,initial,truth,'teacher_previous')
        self.assertLess(abs(local-truth).max(),abs(full-truth).max())
        self.assertEqual(len(ARMS),8)

    def test_reset_uses_previous_not_future_target(self):
        initial=np.array([[.2]]);u=np.full((102,1,1),.5);z=np.full_like(u,.1)
        truth=recurrence(u,z,initial,u,'oracle_both');biased=u+.01
        pred=recurrence(biased,z,initial,truth,'reset_10ms')
        np.testing.assert_array_equal(pred[100],truth[99]+z[100]*(biased[100]-truth[99]))

    def test_notebook_kernel_and_worker(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import task41o_kaggle as k
        doc=k.notebook('a'*40,'b'*64)
        self.assertEqual(doc['metadata']['kernelspec']['name'],'python3')
        code=''.join(doc['cells'][1]['source']);compile(code,'notebook','exec')
        self.assertIn('scripts/run_task41o.py',code);self.assertIn('task41o_final',code)
        self.assertNotIn('scripts/run_task41n_recovery.py',code)

if __name__=='__main__':unittest.main()
