"""Causal, numerical and allocation invariants; no persistent DB writes."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task39i_coupled_relaxation as i
ROOT=Path(__file__).resolve().parents[1]
class Task39iTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.spec=json.loads((ROOT/i.CONFIG).read_text())
    def test_losses_at_boundaries(self):self.assertTrue(i.synthetic_preflight(torch,'cpu')['valid'])
    def test_adapter_initial_function_and_equal_capacity(self):
        groups=i.groups(torch,ROOT,self.spec,'cpu');parent=i.load_parents(torch,ROOT,self.spec,'cpu',torch.float32)[0]
        counts={}
        for group in groups:
            if group['key']=='monotone':continue
            m=group['model'];n=m.a.shape[0];v=torch.linspace(-100,60,19)[None].expand(n,-1);ca=torch.logspace(-7,-2,19)[None].expand(n,-1)
            u,z,_=m.coefficients(v,ca);pu,pz,_=parent.coefficients(v[:3],ca[:3]);factor=n//3
            self.assertLess(float((u-pu.repeat(factor,1,1)).abs().max()),1e-5)
            self.assertLess(float((z-pz.repeat(factor,1,1)).abs().max()),1e-5)
            counts[group['key']]=sum(x.numel() for x in m.parameters())//n
        self.assertEqual(counts['channel']-counts['plain'],250);self.assertEqual(counts['channel'],counts['generic'])
    def test_channel_separation(self):
        m=i.model_class(torch)('channel',[17]);self.assertEqual(m.channel_index.shape,(34,))
        self.assertEqual(len(set(m.channel_index.tolist())),10)
        with torch.no_grad():m.vc.fill_(.1)
        ca=torch.tensor([[1e-5,1e-3]]);v=torch.tensor([[-80.,40.]])
        # Adapter must never add voltage dependence to the SK output.
        u,z=m.raw(v,ca[:,:1].expand_as(v))
        self.assertEqual(float(u[0,0,16]),float(u[0,1,16]));self.assertEqual(float(z[0,0,16]),float(z[0,1,16]))
    def test_monotonicity_and_active_budget(self):
        m=i.model_class(torch)('monotone',[17,29]);ca=torch.logspace(-10,0,1000)[None].expand(2,-1)
        u=m.coefficients(torch.zeros_like(ca),ca)[0][...,16]
        self.assertTrue((torch.diff(u)>=-1e-7).all());self.assertTrue(((u>=0)&(u<=1)).all())
        self.assertEqual(int(sum(x[0].sum() for x in i.masks(torch,m).values())),2404)
    def test_offline_decay_and_original_objective(self):
        x,_=i.q.windows(ROOT/self.spec['dataset'],'fit',['steps'],2,64,17);a=i.labelled(x)
        self.assertEqual(a.shape,(2,64,89));np.testing.assert_array_equal(a[...,:71],x)
        self.assertTrue(np.isfinite(a).all());self.assertTrue((a[...,71:]<=0).all())
        anchor=i.c.anchor_pool(x,17,True);group=i.groups(torch,ROOT,self.spec,'cpu')[0];n=group['model'].a.shape[0]
        s=dict(batch=torch.tensor(a)[None].expand(n,-1,-1,-1).clone(),anchor=torch.tensor(anchor)[None].expand(n,-1,-1).clone(),
            ticks=torch.ones(n,len(anchor),1)*1000,weights=torch.tensor([[.5,.5,1.,1.,0,0,0]]*n))
        scales=[torch.ones(18),torch.ones(11),torch.ones(18),torch.ones(18)]
        torch.testing.assert_close(i.terms(torch,group,s,scales)[:,:4],i.e.terms(torch,group,s,scales,64),rtol=0,atol=0)
    def test_isolated_optimizer(self):
        x,_=i.q.windows(ROOT/self.spec['dataset'],'fit',['steps'],2,8,17);a=i.labelled(x);anchor=i.c.anchor_pool(x,17,True)
        scales=[torch.ones(18),torch.ones(11),torch.ones(18),torch.ones(18)]
        for group in i.groups(torch,ROOT,self.spec,'cpu'):
            n=group['model'].a.shape[0];s=dict(batch=torch.tensor(a)[None].expand(n,-1,-1,-1).clone(),
                anchor=torch.tensor(anchor)[None].expand(n,-1,-1).clone(),ticks=torch.ones(n,len(anchor),1)*1000,weights=torch.ones(n,7))
            self.assertTrue(i.optimizer_preflight(torch,group,s,scales,self.spec['seeds'])['valid'])
    def test_protocol_thresholds_and_control_edges(self):
        old=json.loads((ROOT/i.h.CONFIG).read_text())
        self.assertEqual(self.spec['quality'],old['quality']);self.assertEqual(self.spec['selection'],old['selection'])
        self.assertEqual(i.p.digest(ROOT/i.p.CONFIG),self.spec['quality_config_sha256'])
        self.assertEqual(len(i.arms()),12);self.assertEqual(len(i.contrast_pairs()),14)
        self.assertIn(('channel_bias_matched_capacity','I6','I5'),i.contrast_pairs())
        self.assertIn(('sk_shape_matched_transfer','I10','I8'),i.contrast_pairs())
if __name__=='__main__':unittest.main()
