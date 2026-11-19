import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task51f_local_targets as q
ROOT=Path(__file__).resolve().parents[1]
class Targets(unittest.TestCase):
    def test_mini(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=1.,updates=2,checkpoints=[1,2],batch_size=4,reachable_stride=2,synthetic_per_role=20)
        for r in p['roles'].values():r['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            self.assertTrue(json.loads((out/'final_report.json').read_text())['valid']);r=json.loads((out/'solver_controls.json').read_text());self.assertLess(r['results']['exact']['exact_log_rmse'],1e-10)
    def test_exact_small_target(self):
        import torch
        g=torch.full((1,18),1e-12);logs=q.log_targets(torch,g);self.assertLess(float(logs.min()),-30);self.assertTrue(torch.isfinite(logs).all())
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task51f_kaggle as k
        d=k.notebook('a'*40);self.assertIn('run_task51f.py',json.dumps(d))
        for c in d['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'51f','exec')
if __name__=='__main__':unittest.main()
