import json
from pathlib import Path
import unittest
import numpy as np
import torch
from src.giada_teacher import task39d_current_timing as t
ROOT=Path(__file__).resolve().parents[1]

class Task39dTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data,_=t.pool(ROOT/'experiments/results/task37_playground_v2','fit',['steps'],8,64,99491)
        cls.anchor=t.c.anchor_pool(cls.data,12)
        cls.scales=[torch.ones(n) for n in (18,11,18,18)]

    def static(self,g):
        n=g['model'].a.shape[0]
        return dict(batch=torch.tensor(self.data[:2])[None].expand(n,-1,-1,-1),
            anchor=torch.tensor(self.anchor[:3])[None].expand(n,-1,-1),ticks=torch.full((n,3,1),1000.),weights=torch.ones(n,6))

    def test_matrix_capacity_notebook(self):
        self.assertEqual(len(t.arms()),12);self.assertEqual(len(t.contrast_pairs(t.arms())),16)
        for arch in ('base','linear'):
            self.assertEqual(sum(x.numel() for x in t.model_class(torch)(arch,[17]).parameters()),1220)
        nb=json.loads((ROOT/'notebooks/giada_task39d_current_timing.ipynb').read_text(encoding='utf-8'))
        for k,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),str(k),'exec')

    def test_all36_adam_parity_and_gradients(self):
        for g in t.cohorts(torch,t.arms(),[17,29,43],'cpu'):
            s=self.static(g)
            self.assertTrue(t.optimizer_preflight(torch,g,s,self.scales,[17,29,43],16)['valid'])
            for x in g['model'].parameters():x.grad=torch.zeros_like(x)
            bufs=[x.grad for x in g['model'].parameters()]
            rec,norm=t.gradient_diagnostics(torch,g,s,self.scales,0)
            self.assertTrue(torch.isfinite(norm).all())
            self.assertTrue(all(x.grad is y for x,y in zip(g['model'].parameters(),bufs)))

    def test_calibration_common_and_fixed_control(self):
        groups=t.cohorts(torch,t.arms(),[17,29,43],'cpu');static=[self.static(g) for g in groups]
        t.calibrate(torch,groups,static,self.scales,0,initial=True)
        for s in static:torch.testing.assert_close(s['weights'][:,:4],static[0]['weights'][:3,:4].repeat(len(s['weights'])//3,1))
        before=[s['weights'].clone() for s in static]
        t.calibrate(torch,groups,static,self.scales,30000)
        for g,s,a in zip(groups,static,before):
            mask=~g['flags'][:,2].bool();torch.testing.assert_close(s['weights'][mask],a[mask])

    def test_rollout_precompute_and_resets(self):
        m=t.model_class(torch)('linear',[17]);batch=torch.tensor(self.data[:2])[None]
        actual=t.rollout_training(torch,m,batch,torch.tensor([True]),16)
        state=batch[:,:,0,2:20];states=[]
        for k in range(64):state=m(batch[:,:,k,0],batch[:,:,k,1],state);states.append(state)
        torch.testing.assert_close(actual,torch.stack(states,2),rtol=0,atol=1e-7)
        reset=t.rollout_training(torch,m,batch,torch.tensor([False]),16)
        self.assertGreater(float((actual-reset).abs().max()),1e-5)
        self.assertTrue(bool(((actual>=0)&(actual<=1)).all()))

    def test_targeted_and_parent_hash(self):
        data,info=t.pool(ROOT/'experiments/results/task37_playground_v2','fit',['steps'],8,64,99491,True)
        self.assertEqual(data.shape,self.data.shape);self.assertFalse(np.array_equal(data,self.data))
        self.assertTrue(np.isfinite(data).all());self.assertEqual(sum(info[0]['strata']),8)
        spec=json.loads((ROOT/t.CONFIG).read_text());models=t.load_parents(torch,ROOT,spec,'cpu',torch.float64)
        self.assertEqual(len(models),2)

    def test_short_training_finite_and_split(self):
        groups=t.cohorts(torch,t.arms(),[17,29,43],'cpu');ss=[self.static(g) for g in groups]
        t.calibrate(torch,groups,ss,self.scales,0,initial=True)
        for g,s in zip(groups,ss):
            opt=torch.optim.Adam(g['model'].parameters(),lr=.001)
            for _ in range(10):
                opt.zero_grad();loss=t.objective(torch,g,s,self.scales)
                self.assertTrue(torch.isfinite(loss).all());loss.sum().backward();t.old.clip(torch,g['model']);opt.step()
            self.assertTrue(all(torch.isfinite(x).all() for x in g['model'].parameters()))
        models=t.split_models(torch,groups,[17,29,43],'cpu');self.assertEqual(len(models),12)

    def test_event_identity_and_selection(self):
        r=torch.tensor(self.data[:2,:,60:71])[None];pred=r.clone().requires_grad_()
        a,b=t.event_terms(torch,pred,r);self.assertEqual(float(a+b),0)
        (a+b).sum().backward();self.assertTrue(torch.isfinite(pred.grad).all())
        groups=t.cohorts(torch,[t.arms()[0],t.arms()[-1]],[17,29,43],'cpu')
        score,_=t.c.selection(torch,groups,[self.data[:2]],self.anchor,dict(gate=np.ones(18),current=np.ones(11)),
            dict(recurrence_chunk=4,held_ticks=[1,1000],development_windows_per_family=2),'cpu')
        self.assertEqual(score.shape,(6,));self.assertTrue(np.isfinite(score).all())

    def test_float32_timing_report_has_only_json_scalars(self):
        ref=np.zeros((64,11),dtype=np.float32)
        ref[8:32]=np.linspace(0,.01,24,dtype=np.float32)[:,None]
        ref[32:56]=np.linspace(.01,0,24,dtype=np.float32)[:,None]
        report=t.timing_audit(ref,np.roll(ref,1,axis=0))
        self.assertEqual(len(report),11)
        for row in report:
            self.assertIs(type(row['threshold']),float)
            self.assertIs(type(row['perturbation_absolute']),float)
        self.assertEqual(len(json.loads(json.dumps(report,allow_nan=False))),11)

if __name__=='__main__':unittest.main()
