import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task51_initial_loss_memory as q
ROOT=Path(__file__).resolve().parents[1]
class Diagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch;torch.set_num_threads(1);cls.p=q.plan(ROOT)
    def test_pairing_and_independent_gradients(self):
        t=self.t;net=q.paired_model(t,self.p,'cpu');g=t.rand(24,2,18);h=net.initialize(g);loss=net.opening(h)[0].sum();loss.backward()
        for par in net.parameters():
            for arm in range(1,8):self.assertTrue(t.equal(par[:3],par[arm*3:arm*3+3]))
            if par.grad is not None:self.assertEqual(float(par.grad[1:].abs().sum()),0.)
    def test_loss_zero_and_small_currents(self):
        t=self.t;x=t.full((24,2,11),1e-7);self.assertTrue(t.equal(q.log_error(t,x,x),t.zeros(24)));self.assertTrue(t.isfinite(q.log_error(t,x*100,x)).all())
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task51_kaggle as k
        doc=k.notebook('a'*40);text=json.dumps(doc);self.assertIn('run_task51.py',text);self.assertNotIn('run_task50.py',text);self.assertIn('training_updates_performed=600',text)
        for c in doc['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'51','exec')
    def test_mini_real_pipeline(self):
        p=copy.deepcopy(self.p);p.update(history_ms=1.,updates=2,checkpoints=[1,2],chunk_ticks=4,bptt_ticks=2,batch_size=2,evaluation_cuts_ms=[.5],horizons_ms=[.1,.2],future_protocols=['rest'],probe_ms=.2);p['cost'].update(ticks=1,repetitions=1,batch=2)
        for arm in p['arms']:arm['horizon_ticks']=4 if arm['horizon_ticks']>64 else 2
        for role in p['roles'].values():role['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(self.t,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            report=json.loads((out/'final_report.json').read_text());self.assertTrue(report['valid'] and report['P_frozen'] and report['finite_parameters']);self.assertTrue((out/'resumable_training.pt').exists());self.assertFalse(json.loads((out/'selection_freeze.json').read_text())['fresh_accessed'])
if __name__=='__main__':unittest.main()
