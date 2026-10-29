import ast,json,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task41m_frozen_contract as m
ROOT=Path(__file__).resolve().parents[1]

class FrozenContractTest(unittest.TestCase):
    def test_no_training_or_selection(self):
        tree=ast.parse((ROOT/'src/giada_teacher/task41m_frozen_contract.py').read_text())
        calls=[x.func.attr for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
        self.assertNotIn('backward',calls);self.assertNotIn('step',calls)
        spec,*_=m.contract(ROOT);self.assertEqual(spec['training_updates'],0);self.assertFalse(spec['selection_performed'])
        self.assertEqual(spec['max_gpu_seconds'],3600)
    def test_reference_comparison_keeps_identity(self):
        row=dict(passed_cases=2,total=3,passed=False,interventions=[dict(case=108,seed=43,metrics=dict(passed=False)),dict(case=122,seed=17,metrics=dict(passed=True))])
        s=m.summary(row);self.assertEqual(s['failed_responses'],[dict(case=108,seed=43)])
        self.assertEqual(s['responses_passed'],1);self.assertFalse(s['passed'])
    def test_old_sources_unchanged(self):
        spec,old,*_=m.contract(ROOT)
        for key in ('quality','compartment_gates','cost'):self.assertEqual(spec[key],old[key])
    def test_canonical_cpu_policy_shapes_and_finiteness(self):
        import torch
        _,_,cfg,ca,base=m.contract(ROOT)
        data=dict(voltage=np.full((2,3),-65.,dtype='float32'),calcium=np.full((2,3),1e-4,dtype='float32'),gates=np.full((2,3,18),.01,dtype='float32'),multipliers=np.ones((2,11),dtype='float32'),injection=np.full((2,3),.001,dtype='float32'),base_g_us=np.full((2,3,4),1e-5,dtype='float32'),area_um2=314.1592653589793)
        with torch.inference_mode():
            for policy in m.POLICIES:
                result=m.rollout(torch,None,data,(cfg,ca,base),'cpu',policy)
                self.assertEqual(result['current'].shape,(3,3,2,11));self.assertEqual(result['gates'].shape,(4,3,2,18))
                self.assertTrue(all(np.isfinite(v).all() for v in result.values()))
            a=m.rollout(torch,None,data,(cfg,ca,base),'cpu','primary')
            b=m.a.rollout(torch,None,data,(cfg,ca,base),'cpu',(False,False))
            for name in a:np.testing.assert_array_equal(a[name],b[name])
            # Native records contain the initial state plus each post-step state.
            aligned=dict(data)
            for key in ('voltage','calcium','gates'):aligned[key]=np.concatenate([data[key],data[key][:,-1:]],axis=1)
            meta=[dict(id=k,source_index=0,intervention='none' if k==0 else 'test') for k in range(2)]
            spec=m.contract(ROOT)[0]
            measured=m.measure(a,a,aligned,meta,spec,cfg,'autonomous','41')
            self.assertEqual(measured['total'],6);self.assertEqual(len(measured['interventions']),3)

if __name__=='__main__':unittest.main()
