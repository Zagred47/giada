import tempfile,unittest,zipfile,json,hashlib,os
from pathlib import Path
import numpy as np
from src.giada_teacher import task41n_evaluation_recovery as r
ROOT=Path(__file__).resolve().parents[1]
class RecoveryTests(unittest.TestCase):
    def test_dispatch_notebook_has_explicit_python_kernel(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        from task41n_recovery_kaggle import notebook
        book=notebook('a'*40,'b'*64)
        self.assertEqual(book['metadata']['kernelspec']['name'],'python3')
        self.assertEqual(len({c['id'] for c in book['cells']}),len(book['cells']))
        compile(''.join(book['cells'][1]['source']),'notebook','exec')
    def test_source_and_immutable_floor(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as d:
            plan,spec,freeze,floors=r.verify_source(ROOT,ROOT/'artifacts/task41n_primary_external/recovery_final.zip',Path(d)/'source')
            result=r.classification(floors,True)
            self.assertFalse(result['original_confirmation_passed']);self.assertFalse(result['speedup_claim']);self.assertFalse(result['task42_authorized'])
            self.assertTrue(result['original_confirmation_blocked_by_failed_fp32_floor'])
            self.assertEqual(spec['training_updates'],0);self.assertEqual(freeze['eligible'],plan['eligible'])
    def test_cpu_coupled_exact_and_no_weight_change(self):
        import torch
        torch.set_num_threads(1);os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        nets,parent=r.n.load_models(torch,ROOT,'cpu');before={k:r.n.weights_hash(v) for k,v in nets.items()}
        self.assertTrue(all(torch.equal(v,nets['dual'].parent.state_dict()[k]) for k,v in parent.state_dict().items()))
        spec,cfg,ca,base=r.n.contract(ROOT)
        fake=dict(voltage=np.full((3,5),-65.),calcium=np.full((3,5),1e-4),gates=np.full((3,5,18),.01),multipliers=np.ones((3,11)),injection=np.full((3,4),.001),base_g_us=np.full((3,4,4),1e-5),area_um2=np.array(314.1592653589793))
        with torch.inference_mode():
            reference=r.n.m.rollout(torch,nets['dual'],fake,(cfg,ca,base),'cpu','voltage_solve64')
            for model in nets.values():
                pred=r.n.m.rollout(torch,model,fake,(cfg,ca,base),'cpu','voltage_solve64')
                self.assertTrue(all(r.n.array_bits_equal(pred[k],reference[k]) for k in pred))
        self.assertEqual(before,{k:r.n.weights_hash(v) for k,v in nets.items()})
if __name__=='__main__':unittest.main()
