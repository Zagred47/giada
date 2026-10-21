import json,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task40b_calcium_precision as b
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
class CalciumPrecision(unittest.TestCase):
    def test_unchanged_quality_and_design(self):
        spec=json.loads((ROOT/b.CONFIG).read_text());old=json.loads((ROOT/b.t.CONFIG).read_text())
        self.assertEqual(spec['current_quality'],old['current_quality'])
        self.assertEqual(spec['compartment_gates'],old['compartment_gates'])
        self.assertEqual(len(spec['arms']),8);self.assertEqual(spec['checkpoints'],[0,1000,3000,10000,30000])
    def test_masks_and_independent_optimizer(self):
        report=b.preflight(torch,ROOT,'cpu')
        self.assertTrue(report['valid']);self.assertEqual(report['optimizer_isolation_max_error'],0)
        self.assertEqual(report['zero_correction_error'],0);self.assertEqual(report['noncalcium_preservation_error'],0)
    def test_only_calcium_parameters_trainable_and_causal_core(self):
        parent=b.t.load_models(torch,ROOT,'cpu')['J1'];model=b.make_model(torch,parent,b.arms()[0],[17,29,43],'cpu')
        self.assertEqual([n for n,p in model.named_parameters() if p.requires_grad],['ca_correction'])
        _,cfg,calcium,base=b.t.contract(ROOT);data,_=b.t.design(ROOT,json.loads((ROOT/b.t.CONFIG).read_text()))
        core=b.t.make_core(torch,cfg,calcium,base,data['area_um2'],model,b.t.FAMILIES['all'])
        args=[torch.tensor(x,dtype=torch.float32)[None].expand(3,*x.shape).contiguous() for x in
              (data['voltage'][:2,0],data['calcium'][:2,0],data['gates'][:2,0],data['multipliers'][:2],data['injection'][:2,0],data['base_g_us'][:2,0])]
        old=b.t._rates
        def forbidden(*a,**k):raise AssertionError('Canonical gate fallback')
        try:
            b.t._rates=forbidden
            with torch.no_grad():result=core(*args)
            self.assertTrue(all(torch.isfinite(x).all() for x in result))
        finally:b.t._rates=old
    def test_independent_fit_stream(self):
        fit=b.atomic(40401,4096);hold=b.atomic(40402,256)
        self.assertFalse(np.array_equal(fit['v'][:256],hold['v']))
        self.assertTrue(np.isfinite(fit['z']).all());self.assertTrue((fit['z']>0).all());self.assertTrue((fit['z']<=1).all())
if __name__=='__main__':unittest.main()
