import copy,json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from src.giada_teacher import task41u_objective_features as u
ROOT=Path(__file__).resolve().parents[1]

class Task41u(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        cls.base=u.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(ROOT/'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz') as a:cls.inc=u.r.materialize(torch,cls.base,dict(a))

    def test_scan_and_gradient(self):
        rng=np.random.default_rng(1);a=torch.tensor(rng.uniform(.1,.9,(65,3,2,4)),requires_grad=True);z=torch.tensor(rng.uniform(.001,.4,a.shape),requires_grad=True);initial=torch.full((3,2,4),.4,dtype=torch.float64)
        scan=u.affine_scan(torch,a,z,initial);state=initial;seq=[]
        for tick in range(len(a)):state=state+z[tick]*(a[tick]-state);seq.append(state)
        expected=torch.stack(seq);torch.testing.assert_close(scan,expected,atol=1e-14,rtol=1e-14)
        ga=torch.autograd.grad(scan[-1].sum(),(a,z),retain_graph=True);gb=torch.autograd.grad(expected[-1].sum(),(a,z))
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=1e-13,rtol=1e-12)

    def test_seedwise_clipping_does_not_overflow(self):
        model=torch.nn.Module();model.register_parameter('weight',torch.nn.Parameter(torch.zeros((3,2))))
        model.weight.grad=torch.full_like(model.weight,1e30)
        norm=u.mask_and_clip(torch,model,[model.weight],1.)
        self.assertTrue(torch.isfinite(norm).all())
        torch.testing.assert_close(model.weight.grad.double().square().sum(1).sqrt(),torch.ones(3,dtype=torch.float64),rtol=1e-6,atol=1e-6)

    def test_loss_update_isolation(self):
        spec=dict(u.n.contract(ROOT)[0],stress_duration_ms=30.);d=u.tensor_data(torch,u.data_arrays(spec,[9419101,9419113,9419127]),'cpu')
        for features in (False,True):
            model=copy.deepcopy(self.inc);params=u.set_trainable(torch,model,features);opt=torch.optim.Adam(params,lr=1e-5)
            loss=u.loss_for(torch,model,d,torch.arange(2),lambda *x:u.objective(torch,*x),torch.tensor(1.));loss.sum().backward();u.mask_and_clip(torch,model,params,1.);opt.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            for key,value in model.state_dict().items():
                allowed=('parent.c','parent.d')+(('parent.a','parent.b') if features else ())
                if key not in allowed:self.assertTrue(torch.equal(value,self.inc.state_dict()[key]))
            columns=(*u.GATES,*(g+17 for g in u.GATES))
            for col in range(34):
                if col not in columns:self.assertTrue(torch.equal(model.parent.c[...,col],self.inc.parent.c[...,col]))

    def test_coverage_pairs_and_protocol(self):
        spec=dict(u.n.contract(ROOT)[0],stress_duration_ms=250.);d=u.data_arrays(spec,[9419101,9419113,9419127],True)
        for case in range(0,24,2):
            np.testing.assert_array_equal(d['v'][:,case],d['v'][:,case+1]);np.testing.assert_array_equal(d['ca'][:,case],d['ca'][:,case+1])
            self.assertGreater(float(abs(d['initial'][case]-d['initial'][case+1]).max()),.01)
        p=json.loads((ROOT/'experiments/task41u_preregistration_v1.json').read_text());old=json.loads((ROOT/'experiments/task41t_preregistration_v1.json').read_text())
        self.assertEqual(p['quality'],old['quality']);self.assertEqual(p['checkpoints'],[64,256,768]);self.assertEqual(p['arms'],['baseline','incumbent',*u.ARMS])

    def test_oracle_fixture_including_reset(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            out=Path(temp);folder=out/'opened41n_incumbent';folder.mkdir();ticks=1001;cases=2
            v=np.full((ticks,cases),-40.,dtype='float32');ca=np.full_like(v,1e-4);g=np.full((ticks,cases,18),.3);mult=np.ones((cases,11))
            np.savez_compressed(folder/'fresh_stress_inputs.npz',voltage=v,calcium=ca,initial=g[0],target_gates=g,multipliers=mult)
            np.savez_compressed(folder/'stress_dual.npz',gates=np.broadcast_to(g[:,None],(ticks,3,cases,18)))
            u.causal_probes(torch,self.inc,u.n.contract(ROOT)[0],out,'cpu')
            value=json.loads((out/'causal_probes.json').read_text());self.assertFalse(value['teacher_oracles_selectable']);self.assertEqual(len(value['results']),12)

    def test_notebook_training_entrypoint(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task41u_kaggle as k
        doc=k.notebook('a'*40,'b'*64);code=''.join(doc['cells'][1]['source']);compile(code,'41u','exec')
        self.assertIn('run_task41u.py',code);self.assertIn('training_updates_per_arm=768',code);self.assertNotIn('training_updates_performed=0',code)
        self.assertEqual(doc['metadata']['kernelspec']['name'],'python3')

    def test_training_checkpoint_selection_fixture(self):
        # Unregistered tiny fixture seeds/budget; no scientific development accessed.
        spec=json.loads((ROOT/'experiments/task41u_preregistration_v1.json').read_text())
        spec=dict(spec,fit_path_seeds=[101,103,107],development_seeds=[201,203,207],checkpoints=[1],compile_scan=False,batch_paths=2)
        prior=dict(u.n.contract(ROOT)[0],stress_duration_ms=60.,stress_seeds=[301,303,307])
        original=u.n.weights_hash(self.inc)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            out=Path(temp);models=u.train(torch,self.inc,prior,spec,out,'cpu')
            self.assertEqual(list(models),list(u.ARMS));self.assertEqual(u.n.weights_hash(self.inc),original)
            checkpoint=torch.load(out/'checkpoint_1.pt',map_location='cpu',weights_only=True)
            self.assertIn('masters',checkpoint);self.assertIsNotNone(checkpoint['rng_state'])
            for name in u.ARMS:self.assertTrue((out/f'selection_{name}.json').is_file())

if __name__=='__main__':unittest.main()
