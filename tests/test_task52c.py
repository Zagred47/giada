import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task52c_component_repair as q
ROOT=Path(__file__).resolve().parents[1]
class Repair(unittest.TestCase):
    def test_pipeline(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(history_ms=2.,updates=2,checkpoints=[1,2],batch_size=2,chunk_ticks=4,bptt_ticks=2,calibration_stride=2,diagnostic_ticks=10,probe_updates=2)
        p['roles']={r:dict(seed=400+i,count=2) for i,r in enumerate(('fit','development','fresh'))};p['oracle_initialization'].update(fit_ms=[.2,.5,1],starts=2,iterations=1,cases=1,checkpoint_block=2)
        cp=copy.deepcopy(q.parent.plan(ROOT));cp.update(duration_ms=2.,horizons_ms=[1,2],protocols=['burst']);p['cost'].update(batch=2,ticks=2,repetitions=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',ROOT/'artifacts/task52c_input/models52c.zip.bin',out,'cpu',p,cp)
            report=json.loads((out/'final_report.json').read_text());self.assertTrue(report['valid']);self.assertEqual(len(report['summary']),6)
            self.assertTrue(json.loads((out/'selection_freeze.json').read_text())['frozen_components_verified'])
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task52c_kaggle as k
        doc=k.notebook('a'*40);self.assertIn('run_task52c.py',json.dumps(doc));self.assertIn('models52c.zip.bin',json.dumps(doc))
        for c in doc['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'52c','exec')
    def test_contract(self):
        p=q.plan(ROOT);self.assertEqual([r['id'] for r in p['arms']],['D','DE','DR','DER']);self.assertFalse(p['thresholds_changed']);self.assertFalse(p['abstraction_changed'])
if __name__=='__main__':unittest.main()
