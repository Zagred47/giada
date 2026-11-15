import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task50_compact_memory as q
ROOT=Path(__file__).resolve().parents[1]
class Compact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch;torch.set_num_threads(1);cls.p=q.plan(ROOT)
    def test_model_forward_gradients(self):
        t=self.t;n=q.model_class(t)(self.p,'cpu');g=t.rand(15,2,18);h=n.initialize(g);x=t.zeros(15,2,2);y=n.step(x,h);loss=n.opening(y).sum();loss.backward();self.assertTrue(t.isfinite(y).all());self.assertTrue(((y>=0)&(y<=1)).all());self.assertTrue(all(p.grad is not None for p in n.parameters()));self.assertEqual(len(n.members),15)
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task50_kaggle as k
        d=k.notebook('a'*40);text=json.dumps(d);self.assertIn('run_task50.py',text);self.assertNotIn('run_task49.py',text)
        for c in d['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'50','exec')
    def test_mini_real_pipeline(self):
        p=copy.deepcopy(self.p);p.update(history_ms=1.,updates=2,checkpoints=[1,2],chunk_ticks=3,batch_size=2,evaluation_cuts_ms=[.5],horizons_ms=[.1,.2],future_protocols=['rest']);p['cost'].update(ticks=1,repetitions=1,batch=2)
        for role in p['roles'].values():role['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(self.t,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            r=json.loads((out/'final_report.json').read_text());self.assertTrue(r['valid']);self.assertTrue((out/'selected.pt').exists());self.assertFalse(json.loads((out/'selection_freeze.json').read_text())['fresh_accessed'])
if __name__=='__main__':unittest.main()
