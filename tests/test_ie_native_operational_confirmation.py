import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.giada_teacher import ie_native_operational_confirmation as p7

ROOT=Path(__file__).resolve().parents[1]

class FakeRandom:
    def __init__(self,seed,stream):self.gen=np.random.default_rng(seed+stream);self.position=0
    def repick(self):self.position+=1;return float(self.gen.exponential())
    def seq(self):return self.position

class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.spec,cls.cfg,cls.ca,cls.c3,cls.base=p7.contract(ROOT)

    def test_contract_and_case_splits(self):
        rows=p7.cases(self.spec);self.assertEqual(len(rows),32)
        self.assertTrue(set(self.spec['native_seeds']).isdisjoint({85317,85329,83317,83329,90601}))
        self.assertNotIn(self.spec['calibration']['seed'],self.spec['native_seeds'])
        for schedule in self.spec['schedules_ms'].values():
            for events in schedule.values():self.assertTrue(all(abs(t/.1-round(t/.1))>.1 for t in events))

    def test_forcing_determinism_and_no_target_dependency(self):
        constants={'clock':np.arange(401).tolist(),'initial_plastic':[{'u':0.,'Rstate':1,'tsyn_fac':0.,'tsyn':0.,'released':False} for _ in range(4)],
                   'parameters':{r:[1.,8.,1.] for r in p7.t33.RECEPTORS}}
        constants['clock']=(np.arange(401)*.1).tolist();rows=p7.cases(self.spec)[:2]
        a,sa=p7.independent_forcing(rows,self.cfg,self.c3,constants,401,FakeRandom)
        b,sb=p7.independent_forcing(rows,self.cfg,self.c3,constants,401,FakeRandom)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(sa,sb);self.assertEqual(a.shape,(2,401,4))
        self.assertTrue(np.isfinite(a).all() and (a>=0).all());self.assertTrue(np.array_equal(a[:,:20],np.zeros_like(a[:,:20])))
        # Prefix causality: adding future events cannot change an earlier prefix.
        short,_=p7.independent_forcing(rows,self.cfg,self.c3,constants,81,FakeRandom)
        self.assertTrue(np.array_equal(a[:,:81],short))

    def test_ems_half_step_ties_against_native_trace(self):
        path=ROOT/'experiments/results/ie_p7_calibration_blocked_v1/calibration_native.npz'
        data=np.load(path);clock=data['clock'];dt=self.cfg['dt_ms']
        # Independent archived calibration only, never confirmation outcomes.
        metadata=json.loads(path.with_name('metadata.json').read_text())
        corrected=np.zeros_like(data['native_ab']);continuous=np.zeros_like(corrected)
        for j,params in enumerate(self.c3['synapses']):
            primary='AMPA' if params['mechanism']=='ProbAMPANMDA_EMS' else 'GABAA'
            rise=metadata['parameters'][primary][0]
            a=data['native_ab'][:,j,p7.t33.AB_NAMES.index('A_'+primary)]
            jumps=a[1:]-a[:-1]*np.exp(-dt/rise)
            for event in self.cfg['schedules_ms']['p7_calibration'][params['id']]:
                first=np.searchsorted(clock,event)
                if jumps[first-1]<1e-7:continue
                age=p7.t33.fixed_step_event_age(clock,event,dt);mask=age>=0
                for receptor in (('AMPA','NMDA') if primary=='AMPA' else ('GABAA','GABAB')):
                    r,d,factor=metadata['parameters'][receptor]
                    ratio=self.c3['nmda_ratio'] if receptor=='NMDA' else self.c3['gabab_ratio'] if receptor=='GABAB' else 1.
                    amount=self.cfg['synapse_weight_ns']*ratio*factor
                    for prefix,tau in (('A',r),('B',d)):
                        q=p7.t33.AB_NAMES.index(prefix+'_'+receptor)
                        corrected[mask,j,q]+=amount*np.exp(-age[mask]/tau)
                        continuous[mask,j,q]+=amount*np.exp(-(clock[mask]-event)/tau)
        self.assertLess(np.max(np.abs(corrected-data['native_ab'])),1e-12)
        self.assertGreater(np.max(np.abs(continuous-data['native_ab'])),.05)
        self.assertEqual(p7.t33.fixed_step_event_age(clock,6.45,dt)[np.searchsorted(clock,6.45)],0.)
        self.assertEqual(p7.t33.fixed_step_event_age(clock,13.45,dt)[np.searchsorted(clock,13.45)],dt)

    def test_old_hybrid_unpadded_equivalence(self):
        from src.giada_teacher.joint_heterogeneous_comparison import model_factory
        for family in ('independent','shared_heads'):
            m=model_factory(torch,8,family,[17]);op=p7.hybrid_operator(torch,m);v=torch.linspace(-90,50,12).reshape(3,4)
            u,z=op(v);self.assertEqual(u.shape,(3,4,17))
            packed=torch.cat((v.flatten()[:,None],torch.zeros(12,10),torch.full((12,1),.1)),-1)
            _,ut,tau=m(packed[None]);ut=ut[0].transpose(0,1).reshape(3,4,10)
            self.assertTrue(torch.equal(u[...,:10],ut));self.assertTrue(bool(((z>=0)&(z<=1)).all()))
            fn=torch.compile(op,fullgraph=True,backend='eager');self.assertTrue(torch.equal(fn(v)[0],u))

    def test_stepper_state_and_reset(self):
        v=np.array([-80.,-58.]);ca=np.full(2,1e-4);g=p7.n1.t30.initial_states(v,ca)
        data={'voltage':v[:,None],'calcium':ca[:,None],'gates':g[:,None,:],'multipliers':np.ones((2,11)),
              'injection':np.zeros((2,8)),'base_g_us':np.zeros((2,8,4)),'area_um2':100*np.pi}
        step,reset,state,_=p7.stepper(torch,None,data,np.arange(2),8,self.cfg,self.ca,self.base,'cpu')
        # The compiled recurrent frame must never revisit NumPy area metadata.
        # This guards the Torch inference-mode dispatch-key failure on Kaggle.
        del data['area_um2']
        initial=[x.clone() for x in state]
        for _ in range(8):step()
        reference=[x.clone() for x in state];reset()
        for x,y in zip(state,initial):self.assertTrue(torch.equal(x,y))
        compiled=torch.compile(step,fullgraph=True,backend='eager')
        for _ in range(8):compiled()
        for x,y in zip(state,reference):self.assertTrue(torch.equal(x,y))

    def test_frozen_ops_and_thresholds(self):
        ops=p7.operators(torch,ROOT,'cpu');self.assertEqual(len(ops),14)
        v=torch.linspace(-100,60,57)
        for name,op in ops.items():
            if op is None:continue
            u,z=op(v);self.assertTrue(bool(torch.isfinite(u).all() and torch.isfinite(z).all()))
        zero=np.zeros((5,));g=np.zeros((5,18));target={'voltage':zero,'calcium':zero,'gates':g}
        metric=p7.array_metrics(zero,zero,g,target);self.assertTrue(p7.acceptable(metric,self.spec['quality']))
        metric['voltage_max_mv']=.101;self.assertFalse(p7.acceptable(metric,self.spec['quality']))

if __name__=='__main__':unittest.main()
