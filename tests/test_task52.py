import copy,json,sys,tempfile,unittest
from pathlib import Path
from src.giada_teacher import task52_closed_loop as q
ROOT=Path(__file__).resolve().parents[1]
class Composition(unittest.TestCase):
    def test_pipeline(self):
        import torch
        torch.set_num_threads(1);p=copy.deepcopy(q.plan(ROOT));p.update(duration_ms=25.,horizons_ms=[1,8,25]);p['protocols']=['burst']
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp)/'out';out.mkdir();q.run(torch,ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin',ROOT/'artifacts/task52_input/models52.zip.bin',out,'cpu',p)
            report=json.loads((out/'final_report.json').read_text());self.assertTrue(report['valid']);self.assertTrue(report['reference_floor_passed']);self.assertEqual(report['training_updates'],0)
            for row in report['summary']['oracle_all'].values():self.assertEqual(row['passed'],row['total'])
            import numpy as np
            with np.load(out/'traces.npz') as z:
                self.assertEqual(z['current'].shape,(250,3,15,4,11));self.assertTrue(np.isfinite(z['current']).all())
                for j,case in enumerate(q.design(p)[0]):
                    if case['intervention']=='block_Na':self.assertEqual(float(abs(z['current'][:,:,:,j,2:5]).max()),0)
    def test_notebook(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task52_kaggle as k
        doc=k.notebook('a'*40);self.assertIn('run_task52.py',json.dumps(doc));self.assertIn('str(package[0])',json.dumps(doc))
        for c in doc['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'52','exec')
    def test_quality(self):
        p=q.plan(ROOT);old=json.loads((ROOT/'experiments/task46_preregistration_v1.json').read_text())
        self.assertEqual(p['quality'],old['quality']);self.assertEqual(p['compartment_gates'],old['compartment_gates']);self.assertFalse(p['thresholds_changed'])
if __name__=='__main__':unittest.main()
