import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task51h_reach_preserve as q
ROOT=Path(__file__).resolve().parents[1]
class Optimization(unittest.TestCase):
    def test_mini(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=1.,updates=2,intervention_update=1,checkpoints=[0,1,2],batch_size=4,reachable_stride=2,synthetic_per_role=20)
        for r in p['roles'].values():r['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            self.assertTrue(json.loads((out/'final_report.json').read_text())['valid'])
            curve=json.loads((out/'learning_curve.json').read_text());self.assertEqual(curve[0]['update'],0)
            self.assertLess(max([r for r in curve if r['stage']=='post_projection'][0]['common_score'][12:]),1e-10)
            self.assertTrue((out/'checkpoint_0.pt').exists());self.assertTrue((out/'checkpoint_2.pt').exists())
            pre=torch.load(out/'checkpoint_1_pre.pt',weights_only=True);post=torch.load(out/'checkpoint_1.pt',weights_only=True)
            for name in pre:
                for i in range(18):self.assertTrue(torch.equal(pre[name][i],pre[name][i%3]),name)
                self.assertTrue(torch.equal(pre[name][:6],post[name][:6]),name)
            self.assertTrue(torch.equal(pre['w'],post['w']))
            self.assertTrue(torch.equal(pre['b'],post['b']))
            self.assertTrue(torch.equal(pre['o'][:12],post['o'][:12]))
            self.assertTrue(bool((post['o'][12:]==0).all()))
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task51h_kaggle as k
        doc=k.notebook('a'*40);self.assertIn('run_task51h.py',json.dumps(doc));self.assertNotIn('run_task51g.py',json.dumps(doc))
        for c in doc['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'51h','exec')
        self.assertEqual(k.k.SLUG,'alessandrobelli/giada-task51h-reach-preserve')
    def test_matrix(self):
        p=q.plan(ROOT);self.assertEqual(len(p['arms']),6);self.assertFalse(p['thresholds_changed'])
        self.assertEqual({(a['projection'],a['schedule']) for a in p['arms']},{(x,y) for x in ('none','residual','linearize') for y in ('constant','late')})
if __name__=='__main__':unittest.main()
