import json
from pathlib import Path
import unittest
import torch
import numpy as np
from src.giada_teacher import task39e_tail_support_heads as e
ROOT=Path(__file__).resolve().parents[1]
class Task39eTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data,_=e.q.windows(ROOT/'experiments/results/task37_playground_v2','fit',['steps'],8,256,99591)
        cls.anchor=e.c.anchor_pool(cls.data,12)
        cls.scales=[torch.ones(n) for n in (18,11,18,18)]
    def static(self,g,ticks=4):
        n=g['model'].a.shape[0]
        return dict(batch=torch.tensor(self.data[:2,:ticks])[None].expand(n,-1,-1,-1),
            anchor=torch.tensor(self.anchor[:3])[None].expand(n,-1,-1),ticks=torch.full((n,3,1),1000.),weights=torch.ones(n,4))
    def test_matrix_and_capacity(self):
        self.assertEqual(len(e.arms()),10);self.assertEqual(len(e.contrast_pairs(e.arms())),14)
        groups=e.groups(torch,e.arms(),[17,29,43],'cpu')
        self.assertEqual(sum(g['model'].a.shape[0] for g in groups),30)
        self.assertEqual([a for g in groups for a in g['definitions']],e.arms())
        self.assertEqual(sum(x.numel() for x in e.model_class(torch)('base',[17]).parameters()),1220)
        self.assertEqual(sum(x.numel() for x in e.model_class(torch)('split',[17]).parameters()),1208)
    def test_all_members_adam_parity(self):
        for g in e.groups(torch,e.arms(),[17,29,43],'cpu'):
            self.assertTrue(e.optimizer_preflight(torch,g,self.static(g),self.scales,[17,29,43])['valid'])
    def test_tail_censoring(self):
        raw=torch.tensor([-30.,30.,0.],requires_grad=True);target=torch.tensor([0.,1.,.5])
        loss=e.censored_logit_error(torch,raw,target)
        torch.testing.assert_close(loss,torch.zeros(3));loss.sum().backward();self.assertTrue(torch.isfinite(raw.grad).all())
        x=torch.tensor([0.,0.],requires_grad=True)
        y=e.censored_logit_error(torch,x,torch.tensor([0.,1.]));y.sum().backward()
        self.assertGreater(float(x.grad[0]),0);self.assertLess(float(x.grad[1]),0)
    def test_finite_gradient_diagnostic_overflow(self):
        x=torch.full((2,4,8),1e25,dtype=torch.float32)
        self.assertTrue(torch.isfinite(x).all())
        self.assertFalse(torch.isfinite(x.norm(dim=-1)).all())
        y=x.double();n=y.norm(dim=-1)
        cosine=torch.bmm(y,y.transpose(1,2))/(n[:,:,None]*n[:,None,:]).clamp_min(1e-20)
        self.assertTrue(torch.isfinite(cosine).all())
        torch.testing.assert_close(cosine,torch.ones_like(cosine))
    def test_equal_target_ticks_and_initial_states(self):
        a=self.data[:2];short=a.reshape(8,64,-1)
        np.testing.assert_array_equal(short.reshape(a.shape),a)
        np.testing.assert_array_equal(short[:,0,2:20],a[:,::64,2:20].reshape(8,18))
    def test_coverage_changes_labels_not_validation(self):
        aa=e.c.anchor_pool(self.data,12);bb=e.c.anchor_pool(self.data,12,True)
        self.assertFalse(np.array_equal(aa[:,0],bb[:,0]));self.assertTrue(np.isfinite(bb).all())
        u,z=e.p.coefficients(bb[:,0].astype(float),bb[:,1].astype(float),.1)
        np.testing.assert_allclose(bb[:,71:89],u,atol=2e-6);np.testing.assert_allclose(bb[:,89:107],z,atol=2e-6)
    def test_bounded_recurrence_and_gradient_buffers(self):
        for g in e.groups(torch,e.arms(),[17],'cpu'):
            s=self.static(g)
            x=e.d.rollout_training(torch,g['model'],s['batch'],torch.ones(len(s['batch']),dtype=torch.bool),64)
            self.assertTrue(bool(((x>=0)&(x<=1)).all()))
            for p in g['model'].parameters():p.grad=torch.zeros_like(p)
            buffers=[p.grad for p in g['model'].parameters()]
            rec,norm=e.gradient_probe(torch,g,s,self.scales,0)
            self.assertTrue(torch.isfinite(norm).all());self.assertTrue(all(p.grad is q for p,q in zip(g['model'].parameters(),buffers)))
    def test_selection_rank_and_no_timing_relaxation(self):
        spec=json.loads((ROOT/e.CONFIG).read_text())
        self.assertEqual(spec['long_ticks'],256);self.assertEqual(spec['checkpoints'][-1],180000)
        self.assertIn('72/72',spec['quality']);self.assertIn('NONSELECTABLE',spec['selection'])
        nb=json.loads((ROOT/'notebooks/giada_task39e_tail_support_heads.ipynb').read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
    def test_real_selection_panel_and_scanner(self):
        spec=json.loads((ROOT/e.CONFIG).read_text());spec.update(recurrence_chunk=4,held_ticks=[1,10])
        quality=json.loads((ROOT/e.p.CONFIG).read_text())
        cohorts=e.groups(torch,e.arms(),[17,29,43],'cpu')
        common,parts,ranks,panel=e.checkpoint_scores(torch,cohorts,[self.data[:2,:8]],self.anchor[:4],
            dict(gate=[1.]*18,current=[1.]*11),spec,'cpu',quality)
        self.assertEqual(len(ranks),30);self.assertEqual(len(panel),30);self.assertTrue(np.isfinite(common).all())
        self.assertTrue(all(len(r)==3 and np.isfinite(r).all() for r in ranks))
    def test_nonfinite_member_is_failed_not_crashed_or_selected(self):
        spec=json.loads((ROOT/e.CONFIG).read_text());spec.update(recurrence_chunk=4,held_ticks=[1,10])
        quality=json.loads((ROOT/e.p.CONFIG).read_text())
        cohorts=e.groups(torch,e.arms(),[17,29,43],'cpu')
        with torch.no_grad():cohorts[0]['model'].a[0,0,0]=float('nan')
        common,parts,ranks,panel=e.checkpoint_scores(torch,cohorts,[self.data[:2,:8]],self.anchor[:4],dict(gate=[1.]*18,current=[1.]*11),spec,'cpu',quality)
        self.assertIsNone(ranks[0]);self.assertFalse(panel[0]['selection_eligible'])
        self.assertTrue(panel[0]['numerically_invalid']);self.assertGreater(panel[0]['nonfinite_cases'],0)
        self.assertTrue(all(r is not None for r in ranks[1:]))
        encoded=json.dumps(e.diagnostic_json(dict(common=common.tolist(),panel=panel)),allow_nan=False)
        self.assertIn('null',encoded)
if __name__=='__main__':unittest.main()
