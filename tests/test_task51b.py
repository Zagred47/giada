import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task51b_small_openings as q
ROOT=Path(__file__).resolve().parents[1]
class Small(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch;torch.set_num_threads(1);cls.p=q.plan(ROOT)
    def test_controls(self):self.assertTrue(q.controls(self.t,'cpu')['valid'])
    def test_paired_weights_finite_gradients(self):
        t=self.t;net=q.model_class(t)(self.p,t.full((4,11),1e-8),'cpu');x=t.zeros(18,2,18);y=net(x);y.sum().backward()
        self.assertTrue(t.isfinite(y).all());self.assertTrue((y>0).all())
        for par in (net.w,net.o,net.skip):
            for i in range(3,len(net.members)):self.assertTrue(t.equal(par[i],par[i%3]))
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task51b_kaggle as k
        doc=k.notebook('a'*40);text=json.dumps(doc);self.assertIn('run_task51b.py',text);self.assertNotIn('run_task51.py',text)
        for c in doc['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'51b','exec')
    def test_real_mini_pipeline(self):
        p=copy.deepcopy(self.p);p.update(history_ms=1.,updates=2,checkpoints=[1,2],batch_size=4,reachable_stride=2,synthetic_per_role=8)
        for role in p['roles'].values():role['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(self.t,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            r=json.loads((out/'final_report.json').read_text());self.assertTrue(r['valid'] and r['P_frozen']);self.assertFalse(r['temporal_dynamics_tested'])
if __name__=='__main__':unittest.main()
