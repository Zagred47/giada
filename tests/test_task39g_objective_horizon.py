import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task39g_objective_horizon as g

ROOT=Path(__file__).resolve().parents[1]

class Task39gTests(unittest.TestCase):
    def test_decomposition_value_and_gradient(self):
        torch.set_num_threads(1)
        r=torch.linspace(-1e-4,2e-4,64)[None,None,:,None].expand(2,2,-1,11).clone()
        a=(r+torch.sin(torch.arange(64)[None,None,:,None])*.000001).requires_grad_()
        precision,derivative=g.temporal_components(torch,a,r);v=precision+derivative;old=g.f.temporal_loss(torch,a,r)
        self.assertTrue(torch.equal(v,old))
        x=torch.autograd.grad(v.sum(),a,retain_graph=True)[0];y=torch.autograd.grad(old.sum(),a)[0]
        self.assertTrue(torch.allclose(x,y,atol=1e-6,rtol=1e-6))

    def test_synthetic_sensitivity_and_degenerate_signals(self):
        self.assertTrue(g.synthetic_preflight(torch,'cpu')['valid'])
        t=np.arange(256)*.1
        for ref in [np.zeros(256),np.linspace(-1e-4,1e-4,256),np.exp(-((t-5)/1)**2)+.7*np.exp(-((t-18)/1)**2)]:
            r=np.tile(ref[:,None,None],(1,2,11)).astype('float32');meta=g.event_metadata(r)
            a=torch.tensor(r.transpose(1,0,2))[None].requires_grad_();rr=a.detach().clone()
            loss=g.aligned_loss(torch,a,rr,torch.tensor(meta.transpose(1,0,2))[None]);loss.sum().backward()
            self.assertTrue(bool(torch.isfinite(a.grad).all()));self.assertEqual(float(loss),0)
        self.assertFalse(bool(g.event_metadata(np.zeros((256,1,11)))[...,33:44].any()))

    def test_sampler_parity_and_full_episode_baseline(self):
        spec=json.loads((ROOT/g.CONFIG).read_text());data=ROOT/spec['dataset']
        a,info=g.windows(data,'fit',['steps','calcium_switch'],8,256,99591)
        b,_=g.q.windows(data,'fit',['steps','calcium_switch'],8,256,99591)
        np.testing.assert_array_equal(a[...,:71],b)
        for fi,family in enumerate(['steps','calcium_switch']):
            with np.load(data/f'fit__{family}.npz') as z:current=z['current_targets']
            meta=g.event_metadata(current)
            for j,(start,case) in enumerate(zip(info[fi]['starts'],info[fi]['cases'])):
                np.testing.assert_allclose(a[fi*8+j,:,71:],meta[start:start+256,case],atol=1e-7)
        with self.assertRaises(AssertionError):g.windows(data,'development',['steps'],4,64,17)

    def test_warm_initialization_all_arms(self):
        spec=json.loads((ROOT/g.CONFIG).read_text());parent=g.load_parents(torch,ROOT,spec,'cpu',torch.float32)[0]
        cohorts=g.groups(torch,ROOT,spec,'cpu');self.assertEqual(sum(len(x['definitions']) for x in cohorts),10)
        self.assertEqual(len(g.contrast_pairs()),11)
        for cohort in cohorts:
            for j in range(cohort['model'].a.shape[0]):
                for name,value in cohort['model'].named_parameters():
                    self.assertTrue(torch.equal(value[j],dict(parent.named_parameters())[name][j%3]))
            self.assertEqual(sum(v.numel() for v in cohort['model'].parameters())//cohort['model'].a.shape[0],2404)

    def test_independent_optimizer_every_loss_and_horizon_cohort(self):
        spec=json.loads((ROOT/g.CONFIG).read_text());cohorts=g.groups(torch,ROOT,spec,'cpu')
        data,_=g.windows(ROOT/spec['dataset'],'fit',['steps'],2,256,99591)
        anchors=g.c.anchor_pool(data[...,:71],17,True);anchors=np.tile(anchors,(2,1))
        scales=[torch.ones(18),torch.ones(11),torch.ones(18),torch.ones(18)]
        for cohort in cohorts:
            n=cohort['model'].a.shape[0];h=cohort['definitions'][0]['horizon']
            batch=torch.tensor(data).reshape(512//h,h,126)[None].expand(n,-1,-1,-1).clone()
            s=dict(batch=batch,anchor=torch.tensor(anchors)[None].expand(n,-1,-1).clone(),ticks=torch.ones(n,len(anchors),1),weights=torch.ones(n,7))
            self.assertTrue(g.optimizer_preflight(torch,cohort,s,scales,spec['seeds'])['valid'])

    def test_new_fit_signatures_and_reversal_channel_coverage(self):
        spec=json.loads((ROOT/g.CONFIG).read_text());quality=json.loads((ROOT/g.p.CONFIG).read_text())
        fresh=json.loads(json.dumps(quality));fresh['roles']={'fit':{'seed':spec['expanded_seeds']['fit'],'trajectories_per_family':12}}
        for family in ('reversal','same_v_history','activation_boundary'):
            _,metadata=g.p.make_paths(fresh,'fit',family);signatures={x['input_sha256'] for x in metadata}
            self.assertEqual(len(signatures),12)
            for role in quality['roles']:
                _,old=g.p.make_paths(quality,role,family);self.assertFalse(signatures&{x['input_sha256'] for x in old})
        self.assertEqual({j%11 for j in range(12)},set(range(11)))

    def test_expanded_teacher_materialization(self):
        spec=json.loads((ROOT/g.CONFIG).read_text());quality=json.loads((ROOT/g.p.CONFIG).read_text())
        quality['families']={'steps':quality['families']['steps']}
        parent=ROOT/'artifacts/task39g_test_tmp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            out=Path(temp)/'run/artifacts';out.mkdir(parents=True)
            folder=g.expanded_fit(ROOT,out,spec,quality)
            manifest=json.loads((out/'expanded_fit_manifest.json').read_text());self.assertEqual(len(manifest['files']),2)
            for row in manifest['files']:
                self.assertEqual(g.p.digest(folder/row['file']),row['sha256'])
                self.assertEqual(len(row['trajectories']),12)
                with np.load(folder/row['file']) as z:
                    self.assertEqual(z['gate_targets'].shape,(800,12,18));self.assertTrue(np.isfinite(z['current_targets']).all())

if __name__=='__main__':unittest.main()
