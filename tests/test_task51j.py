import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task51j_confirmation as q
ROOT=Path(__file__).resolve().parents[1]
class Preservation(unittest.TestCase):
    def test_mini(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=1.,updates=2,checkpoints=[0,1,2],batch_size=4,reachable_stride=2,synthetic_per_role=20)
        for r in p['roles'].values():r['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            self.assertTrue(json.loads((out/'final_report.json').read_text())['valid'])
            initial=torch.load(out/'checkpoint_0.pt',weights_only=True);final=torch.load(out/'checkpoint_2.pt',weights_only=True)
            for name in initial:self.assertTrue(torch.equal(initial[name][:3],final[name][:3]))
            self.assertTrue((out/'final_evaluation.json.gz').exists())
    def test_optimizer_equivalence(self):
        import torch
        members=[dict(optimizer='adam',lr=.002,eps=1e-8),dict(optimizer='adam',lr=.002,eps=1e-4),dict(optimizer='adam',lr=.00002,eps=1e-8),dict(optimizer='sgd',lr=.002,eps=1e-8),dict(optimizer='frozen',lr=0.,eps=1e-8)]
        v=torch.nn.Parameter(torch.ones(5,3));refs=[torch.nn.Parameter(v[i].detach().clone()) for i in range(5)]
        opts=[torch.optim.Adam([refs[i]],lr=m['lr'],eps=m['eps']) if m['optimizer']=='adam' else torch.optim.SGD([refs[i]],lr=m['lr']) for i,m in enumerate(members)]
        moments=[(torch.zeros_like(v),torch.zeros_like(v))]
        for step in range(1,5):
            v.grad=torch.tensor([[1e-9,-1e-7,.01]]).expand(5,-1).clone()*step
            for i,opt in enumerate(opts):refs[i].grad=v.grad[i].clone();opt.step()
            q.packed_step(torch,[v],moments,members,step)
            self.assertTrue(torch.allclose(v,torch.stack(refs),atol=2e-7,rtol=0))
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task51j_kaggle as k
        doc=k.notebook('a'*40);self.assertIn('run_task51j.py',json.dumps(doc));self.assertNotIn('run_task51i.py',json.dumps(doc))
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'51j','exec')
    def test_matrix(self):
        p=q.plan(ROOT);self.assertEqual(len(p['arms']),4);self.assertFalse(p['thresholds_changed']);self.assertIn('task52',p['stop_policy'])
        self.assertFalse(set(p['seeds']) & {17,29,43})
        arms={a['id']:a for a in p['arms']}
        self.assertEqual(arms['adam_large_eps']['eps'],1e-4)
        self.assertEqual(arms['adam_large_eps']['lr'],.002)
        self.assertEqual(arms['sgd']['lr'],.002)
        self.assertEqual(p['roles']['fresh']['count'],48)
if __name__=='__main__':unittest.main()
