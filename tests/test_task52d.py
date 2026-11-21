import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task52d_balance_trajectory as q
ROOT=Path(__file__).resolve().parents[1]
class Balance(unittest.TestCase):
    def test_pipeline(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=2,updates=2,checkpoints=[1,2],batch_size=2,chunk_ticks=4,bptt_ticks=2)
        p['roles']={r:dict(seed=700+i,count=2) for i,r in enumerate(('fit','development','fresh'))};p['coupled_training'].update(duration_ms=2,batch_size=2,burnin_ticks=2,horizon_before300=4,horizon_after300=4);p['cost'].update(batch=2,ticks=2,repetitions=1)
        cp=copy.deepcopy(q.parent.plan(ROOT));cp.update(duration_ms=2,horizons_ms=[1,2],protocols=['burst'])
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',ROOT/'artifacts/task52d_input/models52d.zip.bin',out,'cpu',p,cp)
            report=json.loads((out/'final_report.json').read_text());self.assertTrue(report['complete']);self.assertTrue(report['encoder_unchanged']);self.assertEqual(len(report['summary']),6)
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task52d_kaggle as k
        doc=k.notebook('a'*40);self.assertIn('run_task52d.py',json.dumps(doc));self.assertIn('models52d.zip.bin',json.dumps(doc))
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'52d','exec')
    def test_contract(self):
        p=q.plan(ROOT);self.assertEqual(len(p['arms']),4);self.assertFalse(p['thresholds_changed']);self.assertFalse(p['abstraction_changed']);self.assertTrue(all(not r['encoder_trainable'] for r in p['arms']))
if __name__=='__main__':unittest.main()
