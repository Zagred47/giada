import unittest,json
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task40_causal_reinsertion as t
ROOT=Path(__file__).resolve().parents[1]

class FrozenCausalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.spec,cls.cfg,cls.calcium,cls.base=t.contract(ROOT)
        cls.data,cls.meta=t.design(ROOT,cls.spec)
        cls.models=t.load_models(torch,ROOT,'cpu')

    def test_quality_unchanged(self):
        self.assertEqual(self.spec['current_quality'],json.loads((ROOT/t.p.CONFIG).read_text())['quality'])
        self.assertEqual(self.spec['compartment_gates'],self.cfg['gates'])

    def test_design_pairing(self):
        self.assertEqual(len(self.meta),88)
        for i,row in enumerate(self.meta[32:],32):
            origin=row['source_index']
            np.testing.assert_array_equal(self.data['gates'][i],self.data['gates'][origin])
            if row['intervention'].startswith('block_'):
                k=t.ionic.CHANNELS.index(row['intervention'][6:])
                self.assertEqual(self.data['multipliers'][i,k],0)
                other=[j for j in range(11) if j!=k]
                np.testing.assert_array_equal(self.data['multipliers'][i,other],self.data['multipliers'][origin,other])

    def test_native_floor_and_temporal_shape(self):
        subset={k:v if k=='area_um2' else v[:32] for k,v in self.data.items()}
        ref=t.numpy_reference(subset,self.cfg,self.calcium,self.base)
        self.assertEqual(ref['current'].shape,(400,32,11))
        self.assertEqual(ref['voltage'].shape,(401,32))
        native=[dict(voltage=subset['voltage'][k],calcium=subset['calcium'][k]) for k in range(32)]
        result=t.t33.metrics({k:ref[k][None] for k in ('voltage','calcium','gates')},native)
        self.assertTrue(t.t33._floor_pass(result,self.cfg),result)

    def test_formula_equivalence_and_no_canonical_fallback(self):
        with torch.inference_mode():
            report=t.preflight(torch,ROOT,'cpu')
        self.assertTrue(report['valid'])

    def test_metric_self_reference_and_nonfinite(self):
        subset={k:v if k=='area_um2' else v[:1] for k,v in self.data.items()}
        ref=t.numpy_reference(subset,self.cfg,self.calcium,self.base)
        pred={k:np.repeat(v[:,None],3,axis=1) for k,v in ref.items()}
        good=t.metrics(pred,ref,subset,self.meta[:1],self.spec['current_quality'],self.cfg['gates'],'autonomous')
        self.assertTrue(good['passed'])
        pred['voltage'][2,0,0]=np.nan
        bad=t.metrics(pred,ref,subset,self.meta[:1],self.spec['current_quality'],self.cfg['gates'],'autonomous')
        self.assertFalse(bad['passed'])

    def test_new_calcium_sk_and_blocked_current(self):
        data=self.data
        def x(a):return torch.tensor(a,dtype=torch.float32)[None].expand(3,*a.shape).contiguous()
        args=[x(data[k][:8,0]) for k in ('voltage','calcium','gates')]
        args += [x(data['multipliers'][:8]),x(data['injection'][:8,0]),x(data['base_g_us'][:8,0])]
        core=t.make_core(torch,self.cfg,self.calcium,self.base,data['area_um2'],self.models['J1'],t.FAMILIES['all'])
        with torch.inference_mode():
            v,ca,g,cur=core(*args)
            expected=t.learned_sk(torch,self.models['J1'],ca,args[2][...,16])
            self.assertTrue(torch.equal(g[...,16],expected))
            args[3][...,9]=0
            self.assertTrue(torch.equal(core(*args)[3][...,9],torch.zeros_like(cur[...,9])))
            runner=t.Stepper(torch,core,args)
            a=[v.clone() for v in runner.tick()]
            b=runner.tick()
            self.assertFalse(torch.equal(a[0],b[0]))
            runner.reset(args)
            for old,new in zip(a,runner.tick()):self.assertTrue(torch.equal(old,new))

    def test_autonomous_cannot_read_future_teacher(self):
        data={k:(v if k=='area_um2' else v[:2].copy()) for k,v in self.data.items()}
        for k in ('voltage','calcium','gates','injection','base_g_us'):data[k]=data[k][:,:4].copy()
        changed={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in data.items()}
        for k in ('voltage','calcium','gates'):changed[k][:,1:]=12345
        core=t.make_core(torch,self.cfg,self.calcium,self.base,data['area_um2'],self.models['J1'],t.FAMILIES['all'])
        with torch.inference_mode():
            a=t.rollout(torch,core,data,'cpu');b=t.rollout(torch,core,changed,'cpu')
        for k in a:np.testing.assert_array_equal(a[k],b[k])

if __name__=='__main__':unittest.main()
