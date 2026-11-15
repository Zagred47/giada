import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task51e_readout_conditioning as q
ROOT=Path(__file__).resolve().parents[1]
class Conditioning(unittest.TestCase):
    def test_mini(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=1.,updates=2,checkpoints=[1,2],batch_size=4,reachable_stride=2,synthetic_per_role=20)
        for r in p['roles'].values():r['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            self.assertTrue(json.loads((out/'final_report.json').read_text())['valid']);self.assertFalse(json.loads((out/'linear_solver_control.json').read_text())['analytic_powers_supplied'])
    def test_masks_and_pairing(self):
        import torch
        p=q.plan(ROOT);net=q.model_class(torch)(p,torch.full((5,11),.1),'cpu');x=torch.zeros(24,1,18);first=net(x);x[:,:,17]=1;second=net(x)
        self.assertTrue(torch.equal(first[12:18,:,:10],second[12:18,:,:10]));self.assertTrue(torch.equal(net.skip[0],net.skip[3]));self.assertEqual(float(net.wm[18:].sum()),0.)
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task51e_kaggle as k
        d=k.notebook('a'*40);self.assertIn('run_task51e.py',json.dumps(d))
        for c in d['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'51e','exec')
if __name__=='__main__':unittest.main()
