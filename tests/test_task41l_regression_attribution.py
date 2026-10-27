import copy
import itertools
import unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41l_regression_attribution as l
from src.giada_teacher import task41l_support as s
from src.giada_teacher import task41l_frozen_attribution as a


class Tiny(torch.nn.Module):
    def __init__(self,offset):
        super().__init__();self.a=torch.nn.Parameter(torch.full((3,1),float(offset)))
        for name in ('sa','sb','sc','sd'):self.register_buffer(name,torch.zeros(3,1))
        self.register_buffer('gbar',torch.ones(3,11))
    def raw(self,v,ca):
        value=self.a.reshape(3,*([1]*(v.ndim-1)),1).expand(*v.shape,18)
        return value,value+1
    def forward(self,v,ca,state):
        u,z=self.raw(v,ca);return state+torch.sigmoid(z)*(torch.sigmoid(u)-state)


class Tests(unittest.TestCase):
    def test_factorial(self):
        arms=l.arms();self.assertEqual(len(arms),8)
        self.assertEqual({(x['amplitude'],x['coverage'],x['protection']) for x in arms},set(itertools.product((0,1),repeat=3)))
        self.assertTrue(all(x['task']=='repair' and x['representation']=='chain' for x in arms))
    def test_nested_and_frozen_cutoff(self):
        fam=np.tile(np.arange(2),8);errors=np.arange(16.)
        sampler,cut=s.family_samplers(errors,fam)
        wide,again=s.family_samplers(np.tile(errors,4),np.tile(fam,4),cut)
        self.assertEqual(cut,again)
        for mapping in (sampler,wide):
            ids=s.draw(torch,mapping,np.array([.1,.6]),'cpu')
            np.testing.assert_array_equal(np.tile(fam,4)[ids],[0,1])
        x=[{'v':torch.ones(2,3)*i,'target':torch.ones(4,2,11)*i} for i in range(4)]
        y=s.combine(torch,x,True)
        self.assertTrue(torch.equal(y['target'][:,:2],x[0]['target']))
        self.assertEqual(y['target'].shape,(4,8,11))
    def test_swaps_joint_uz(self):
        parent,donor=Tiny(0),Tiny(2);v=torch.zeros(3,4);ca=torch.ones_like(v)*1e-4
        for bits in itertools.product((0,1),repeat=3):
            model=a.swap_class(torch)(parent,donor,bits)
            u,z=model.raw(v,ca)
            self.assertTrue(torch.equal(z,u+1))
            for channel,name in enumerate(l.c.ionic.CHANNELS):
                lo,hi=map(int,l.c.ionic.OFFSETS[channel:channel+2]);wanted=2 if name in a.CHANNELS and bits[a.CHANNELS.index(name)] else 0
                self.assertTrue(torch.equal(u[...,lo:hi],torch.full_like(u[...,lo:hi],wanted)))
            clone=copy.deepcopy(model);self.assertTrue(torch.equal(clone.raw(v,ca)[0],u))
    def test_factorial_effects(self):
        rows=[]
        for bits in itertools.product((0,1),repeat=3):
            value=2*bits[0]+3*bits[1]+4*bits[0]*bits[1]
            rows.append(dict(panel='x',policy='p',forcing='f',model='swap'+''.join(map(str,bits)),continuous={key:dict(rmse=[value]*3) for key in ('voltage','calcium','gates','current')}))
        effects=a.factorial(rows);self.assertEqual(len(effects),7)
        self.assertEqual(next(x for x in effects if x['interaction']==['Ca_HVA'])['rmse_effect']['current'],[4.]*3)


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.root=Path(__file__).resolve().parents[1]
        cls.spec,cls.cfg,cls.ca,cls.base=l.k.contract(cls.root)
        cls.spec=copy.deepcopy(cls.spec);cls.spec['new_seeds']={k:v+110000 for k,v in cls.spec['new_seeds'].items()}
        out=cls.root/'artifacts/task41l_cpu_preflight';out.mkdir(exist_ok=True)
        cls.cc,cls.models,cls.data,_,cls.pools,_,_,cls.samplers=l.setup(torch,cls.root,out,'cpu',cls.spec,cls.cfg,cls.ca,cls.base,True)
        cls.physics=(cls.cfg,cls.ca,cls.base)
    def test_initial_parent_and_nested_support(self):
        v=torch.full((3,4),-60.);ca=torch.full_like(v,1e-4)
        for model in self.models.values():
            self.assertEqual(sum(p.numel() for p in model.parameters())//3,4004)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(model.raw(v,ca),self.cc['H_nominee'].raw(v,ca))))
        for small,large in zip(self.pools[0],self.pools[1]):
            self.assertTrue(torch.equal(small['v'],large['v'][:len(small['v'])]))
    def test_objectives_probe_and_resume(self):
        models=copy.deepcopy(self.models);opts={n:torch.optim.Adam(l.j.parameters(m)) for n,m in models.items()}
        batch=s.minibatches(torch,np.random.default_rng(12),self.pools,self.samplers,self.spec,'cpu')
        l.update(torch,models,opts,self.pools,batch,self.physics,1);l.j.finite(torch,models,opts)
        copies=copy.deepcopy(models);co={n:torch.optim.Adam(l.j.parameters(m)) for n,m in copies.items()}
        for n in co:co[n].load_state_dict(copy.deepcopy(opts[n].state_dict()))
        for name in ('L_A0D0P0','L_A0D0P1'):
            result=l.adam_probe(torch,models[name],opts[name],*self.pools[0],batch[0],self.physics,name.endswith('1'),1e-6)
            self.assertTrue(result['candidate_and_optimizer_unchanged']);self.assertEqual(len(result['regime_gradients']),15)
        l.update(torch,models,opts,self.pools,batch,self.physics,2);l.update(torch,copies,co,self.pools,batch,self.physics,2)
        self.assertTrue(all(torch.equal(a,b) for n in models for a,b in zip(models[n].parameters(),copies[n].parameters())))
    def test_primary_canonical_core_exact(self):
        args=[torch.full((3,4),-65.),torch.full((3,4),1e-4),torch.full((3,4,18),.01),torch.ones(3,4,11),torch.full((3,4),.001),torch.full((3,4,4),1e-5)]
        expected=l.t.make_core(torch,*self.physics,100*np.pi)(*args)
        actual=a.canonical_core(torch,*self.physics,100*np.pi,False,False)(*args)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(expected,actual)))

if __name__=='__main__':unittest.main()
