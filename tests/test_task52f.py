import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task52f_encoder_credit as q
ROOT=Path(__file__).resolve().parents[1]
class Protected(unittest.TestCase):
    def test_pipeline(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=2,updates=2,checkpoints=[1,2],batch_size=2,chunk_ticks=4,bptt_ticks=2)
        p['encoder_calibration']['updates']=2
        for i,arm in enumerate(p['arms']):arm['bptt']=2 if i<2 else 4
        p['roles']={r:dict(seed=700+i,count=2) for i,r in enumerate(('fit','development'))};p['support_protocol']['count_per_role']=2;p['coupled_training'].update(duration_ms=2,batch_size=2,burnin_ticks=2,horizon_before300=4,horizon_after300=4);p['cost'].update(batch=2,ticks=2,repetitions=1)
        cp=copy.deepcopy(q.parent.plan(ROOT));cp.update(duration_ms=2,horizons_ms=[1,2],protocols=['burst'])
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',ROOT/'artifacts/task52e_input/models52e.zip.bin',out,'cpu',p,cp)
            report=json.loads((out/'final_report.json').read_text());self.assertTrue(report['complete']);self.assertTrue(report['encoder_frozen_after_calibration']);self.assertEqual(len(report['summary']),6)
            freeze=json.loads((out/'selection_freeze.json').read_text());self.assertEqual(len(freeze['updates']),12)
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task52f_kaggle as k
        doc=k.notebook('a'*40);self.assertIn('run_task52f.py',json.dumps(doc));self.assertIn('models52e.zip.bin',json.dumps(doc))
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'52f','exec')
    def test_contract(self):
        p=q.plan(ROOT);self.assertEqual(len(p['arms']),4);self.assertFalse(p['thresholds_changed']);self.assertFalse(p['abstraction_changed']);self.assertTrue(all(not r['encoder_trainable'] for r in p['arms']))
if __name__=='__main__':unittest.main()
