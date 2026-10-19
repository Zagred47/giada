import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task39f_temporal_recovery as f

class Task39fTests(unittest.TestCase):
    def test_matrix_and_contrasts(self):
        aa=f.arms();self.assertEqual(len(aa),8);self.assertEqual(len(f.contrast_pairs(aa)),12)
        for axis,a,b in f.contrast_pairs(aa):
            left=next(x for x in aa if x['name']==a);right=next(x for x in aa if x['name']==b)
            self.assertEqual(sum(left[k]!=right[k] for k in ('architecture','support','temporal')),1)

    def test_temporal_loss_is_finite_and_sensitive(self):
        ref=torch.linspace(0,1e-5,64)[None,None,:,None].expand(2,2,64,11).clone()
        pred=ref.clone().requires_grad_();zero=f.temporal_loss(torch,pred,ref)
        self.assertTrue(torch.equal(zero,torch.zeros_like(zero)))
        bad=(ref+1e-7).requires_grad_();loss=f.temporal_loss(torch,bad,ref)
        self.assertTrue(bool((loss>0).all()));loss.sum().backward();self.assertTrue(bool(torch.isfinite(bad.grad).all()))
        tiny=torch.zeros_like(ref);x=torch.full_like(ref,1e-12,requires_grad=True)
        f.temporal_loss(torch,x,tiny).sum().backward();self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_width_and_structure(self):
        for arch,count in [('base',1220),('wide',2404)]:
            model=f.model_class(torch)(arch,[17]);self.assertEqual(sum(x.numel() for x in model.parameters()),count)
            v=torch.full((1,5),-45.);ca=torch.full_like(v,1e-4);state=torch.rand(1,5,18)
            out=model(v,ca,state);self.assertTrue(bool(((out>=0)&(out<=1)).all()))
            changed=model(v,ca*5,state);self.assertTrue(torch.equal(out[...,:16],changed[...,:16]));self.assertTrue(torch.equal(out[...,17:],changed[...,17:]))

    def test_recovery_sampler_uses_valid_prior_state(self):
        parent=Path(__file__).resolve().parents[1]/'artifacts/task39f_test_tmp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp);t=400;v=np.full((t,2),-70.);v[100:200]=-20
            gates=np.arange(t*2*18).reshape(t,2,18)/1e5;initial=np.full((2,18),.123)
            current=np.sin(np.arange(t)[:,None,None]/30)*np.ones((t,2,11))*1e-4
            np.savez(root/'fit__fixture.npz',voltage=v,bridge_ca=np.full_like(v,1e-4),gate_targets=gates,initial_gates=initial,multipliers=np.ones((2,11)),reversals=np.zeros((2,11)),current_targets=current)
            a,report=f.recovery_windows(root,'fit',['fixture'],16,64,17);again,_=f.recovery_windows(root,'fit',['fixture'],16,64,17)
            self.assertTrue(np.array_equal(a,again));self.assertEqual(report[0]['modes'],[4]*4)
            for i,(start,case) in enumerate(zip(report[0]['starts'],report[0]['cases'])):
                expected=initial[case] if start==0 else gates[start-1,case]
                np.testing.assert_allclose(a[i,0,2:20],expected,atol=1e-7)
                np.testing.assert_allclose(a[i,:,60:71],current[start:start+64,case],atol=1e-10)
            with self.assertRaises(AssertionError):f.recovery_windows(root,'development',['fixture'],16,64,17)

    def test_independent_optimizer_with_objective(self):
        torch.set_num_threads(1)
        for arch in ('base','wide'):
            defs=[a for a in f.arms() if a['architecture']==arch]
            group=f.groups(torch,defs,[17], 'cpu')[0]
            n=4;batch=torch.zeros(n,1,4,71);batch[...,0]=-50;batch[...,1]=1e-4;batch[...,2:38]=.4;batch[...,38:49]=1
            anchor=torch.zeros(n,2,107);anchor[...,0]=-50;anchor[...,1]=1e-4;anchor[...,2:38]=.4;anchor[...,38:49]=1;anchor[...,71:89]=.4;anchor[...,89:107]=.05
            static=dict(batch=batch,anchor=anchor,ticks=torch.ones(n,2,1),weights=torch.ones(n,5))
            result=f.optimizer_preflight(torch,group,static,[torch.ones(18),torch.ones(11),torch.ones(18),torch.ones(18)],[17])
            self.assertTrue(result['valid'])

    def test_frozen39e_parent_extraction(self):
        root=Path(__file__).resolve().parents[1];spec=json.loads((root/f.CONFIG).read_text())
        models=f.load_parents(torch,root,spec,'cpu',torch.float32)
        self.assertEqual(len(models),2)
        for m in models:self.assertTrue(all(bool(torch.isfinite(v).all()) for v in m.parameters()))

    def test_timing_diagnostics_do_not_change_event(self):
        t=np.arange(300)*.1;r=np.exp(-((t-8)/3)**2)[:,None]*np.ones((1,11))*1e-4
        pred=np.roll(r,4,axis=0);before=f.p.event_metrics(r[:,0],pred[:,0],.1,1e-7)
        rows=f.timing_audit(r,pred);self.assertEqual(rows[0]['event'],before)
        self.assertEqual(len(rows[0]['crossings']),2);self.assertTrue(rows[0]['diagnostic_only'])

if __name__=='__main__':unittest.main()
