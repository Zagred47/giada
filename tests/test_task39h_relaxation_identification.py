"""Scientific invariants of Task39h; no production DB writes."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task39h_relaxation_identification as h
ROOT=Path(__file__).resolve().parents[1]
class Task39hTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.spec=json.loads((ROOT/h.CONFIG).read_text())
    def test_loss_local_error_and_empty_bins(self):
        self.assertTrue(h.synthetic_preflight(torch,'cpu')['valid'])
        z=torch.logspace(-8,-.001,18)[None,None].expand(2,64,18).clone()
        raw=torch.logit(z).requires_grad_();loss,diag=h.bin_loss(torch,raw,z,torch.ones(18));loss.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all());self.assertTrue(torch.isfinite(loss).all())
        self.assertLess(float(loss.abs().max()),1e-6)
    def test_factorial_edges_isolate_one_factor(self):
        aa=h.arms();self.assertEqual(len(aa),10);self.assertEqual(len(h.contrast_pairs()),14)
        for axis,a,b in h.contrast_pairs():
            x=next(v for v in aa if v['name']==a);y=next(v for v in aa if v['name']==b)
            changed=[k for k in ('learning_rate','z_only','auxiliary','expanded') if x[k]!=y[k]]
            self.assertEqual(changed,['expanded'] if axis=='recovery_support' else [axis])
    def test_warm_parent_and_inference_invariance(self):
        parents=h.load_parents(torch,ROOT,self.spec,'cpu',torch.float32)
        for group in h.groups(torch,ROOT,self.spec,'cpu'):
            for j in range(group['model'].a.shape[0]):
                for name,v in group['model'].named_parameters():
                    self.assertTrue(torch.equal(v[j],dict(parents[0].named_parameters())[name][j%3]))
            self.assertEqual(sum(v.numel() for v in group['model'].parameters())//group['model'].a.shape[0],2404)
    def test_independent_masked_adam_and_u_invariance(self):
        aa,_=h.windows(ROOT/self.spec['dataset'],'fit',['steps'],2,16,99591)
        anchor=h.c.anchor_pool(aa,17,True);anchor=np.tile(anchor,(4,1))
        scale=[torch.ones(18),torch.ones(11),torch.ones(18),torch.ones(18)]
        for group in h.groups(torch,ROOT,self.spec,'cpu'):
            n=group['model'].a.shape[0]
            s=dict(batch=torch.tensor(aa)[None].expand(n,-1,-1,-1).clone(),anchor=torch.tensor(anchor)[None].expand(n,-1,-1).clone(),
                ticks=torch.ones(n,len(anchor),1)*1000,weights=torch.ones(n,5))
            r=h.optimizer_preflight(torch,group,s,scale,self.spec['seeds'])
            self.assertTrue(r['valid']);self.assertEqual(r['frozen_u_output_error'],0)
    def test_recovery_support_is_independent_bounded_and_reproducible(self):
        quality=json.loads((ROOT/h.p.CONFIG).read_text());quality['families']={'steps':quality['families']['steps']}
        spec=dict(self.spec,recovery_ticks=320,recovery_nodes=128,windows_per_family=8)
        base=ROOT/'artifacts/task39h_test_tmp';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as t:
            out=Path(t)/'run/artifacts';out.mkdir(parents=True)
            windows,anchors=h.recovery_fit(ROOT,out,spec,quality);report=json.loads((out/'recovery_fit_manifest.json').read_text())
            self.assertEqual([v.shape for v in windows],[(8,256,71),(8,256,71)])
            self.assertTrue(all(np.isfinite(v).all() for v in windows))
            self.assertTrue(all(((v[...,20:38]>=0)&(v[...,20:38]<=1)).all() for v in windows))
            self.assertEqual(len(report['files']),2);self.assertEqual(anchors.shape,(16,107))
            for row in report['files']:self.assertEqual(len(set(row['input_signatures'])),36)
    def test_sampler_matches_previous_and_quality_selection_unchanged(self):
        x,_=h.windows(ROOT/self.spec['dataset'],'fit',['steps'],8,256,99591)
        y,_=h.g.windows(ROOT/self.spec['dataset'],'fit',['steps'],8,256,99591)
        np.testing.assert_array_equal(x,y[...,:71])
        previous=json.loads((ROOT/h.g.CONFIG).read_text())
        self.assertEqual(self.spec['quality'],previous['quality']);self.assertEqual(self.spec['selection'],previous['selection'])
        self.assertEqual(h.p.digest(ROOT/h.p.CONFIG),self.spec['quality_config_sha256'])
if __name__=='__main__':unittest.main()
