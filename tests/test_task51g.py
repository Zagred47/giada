import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task51g_optimization_gap as q
ROOT=Path(__file__).resolve().parents[1]
class Optimization(unittest.TestCase):
    def test_mini(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=1.,updates=2,checkpoints=[0,1,2],batch_size=4,reachable_stride=2,synthetic_per_role=20)
        for r in p['roles'].values():r['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            self.assertTrue(json.loads((out/'final_report.json').read_text())['valid'])
            curve=json.loads((out/'learning_curve.json').read_text());self.assertEqual(curve[0]['update'],0)
            self.assertLess(max(curve[0]['common_score'][6:]),1e-10)
            self.assertTrue((out/'checkpoint_0.pt').exists());self.assertTrue((out/'checkpoint_2.pt').exists())
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task51g_kaggle as k
        doc=k.notebook('a'*40);self.assertIn('run_task51g.py',json.dumps(doc));self.assertNotIn('run_task51f.py',json.dumps(doc))
        for c in doc['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'51g','exec')
        self.assertEqual(k.k.SLUG,'alessandrobelli/giada-task51g-optimization-gap')
    def test_matrix(self):
        p=q.plan(ROOT);self.assertEqual(len(p['arms']),4);self.assertFalse(p['thresholds_changed'])
        self.assertEqual({(a['initialization'],a['schedule']) for a in p['arms']},{('standard','constant'),('standard','decay'),('fit','constant'),('fit','decay')})
if __name__=='__main__':unittest.main()
