import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task51c_hybrid_openings as q
ROOT=Path(__file__).resolve().parents[1]
class Hybrid(unittest.TestCase):
    def test_mini(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=1.,updates=2,checkpoints=[1,2],batch_size=4,reachable_stride=2,synthetic_per_role=8)
        for r in p['roles'].values():r['count']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',out,'cpu',p)
            self.assertTrue(json.loads((out/'final_report.json').read_text())['valid']);self.assertFalse(json.loads((out/'selection_freeze.json').read_text())['fresh_accessed'])
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task51c_kaggle as k
        d=k.notebook('a'*40);s=json.dumps(d);self.assertIn('run_task51c.py',s);self.assertNotIn('run_task51b.py',s)
        for c in d['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'51c','exec')
if __name__=='__main__':unittest.main()
