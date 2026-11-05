import os,unittest
from pathlib import Path
import numpy as np
try:import torch
except ImportError:torch=None
from src.giada_teacher import task41r_direct_timing as r

class CausalJacobian(unittest.TestCase):
    def test_affine_derivative_matches_finite_difference(self):
        features=np.array([[1.,-.2],[1.,.3],[1.,.7],[1.,-.1]])
        wu=np.array([.2,-.4]);wz=np.array([-3.,.1])
        def solve(wu,wz):
            state=np.array(.6)
            for f in features:
                u=1/(1+np.exp(-f@wu));z=1/(1+np.exp(-f@wz));state+=z*(u-state)
            return state
        state=np.array(.6);su=np.zeros(2);sz=np.zeros(2)
        for f in features:
            u=np.array(1/(1+np.exp(-f@wu)));z=np.array(1/(1+np.exp(-f@wz)))
            state,su,sz=r.affine_sensitivity(state,su,sz,u,z,f)
        for j in range(2):
            shift=np.eye(2)[j]*1e-6
            self.assertAlmostEqual(su[j],(solve(wu+shift,wz)-solve(wu-shift,wz))/2e-6,places=8)
            self.assertAlmostEqual(sz[j],(solve(wu,wz+shift)-solve(wu,wz-shift))/2e-6,places=8)

    def test_channel_readout_matches_original(self):
        rng=np.random.default_rng(1);g=rng.uniform(.1,.9,(5,18));v=np.full(5,-40.);mult=np.ones((5,11))
        original=r.n.p.readout(v,g,mult,r.n.p.ionic.REVERSALS)[1]
        selected=g[:,list(r.p.EQ)]
        for name,(channel,ids) in r.GROUPS.items():
            np.testing.assert_allclose(r.channel_current(selected,v,mult,name),original[:,channel],rtol=1e-14)
            derivatives=r.current_derivatives(selected,v,mult,name)
            for j,pos in enumerate(ids):
                plus=selected.copy();minus=selected.copy();plus[:,pos]+=1e-6;minus[:,pos]-=1e-6
                numeric=(r.channel_current(plus,v,mult,name)-r.channel_current(minus,v,mult,name))/2e-6
                np.testing.assert_allclose(derivatives[j],numeric,rtol=1e-8)

    @unittest.skipIf(torch is None,'Torch absent')
    def test_whitened_step_matches_penalized_normal_equations(self):
        b=np.array([[1.,2.],[-1.,.5],[.3,-.4]]);y=np.array([.5,-.2,.3]);d=torch.tensor([.1,-.1],dtype=torch.float64)
        actual=r.timing_step(torch,d,torch.eye(2,dtype=torch.float64),b,y)
        expected=np.linalg.solve(np.eye(2)+b.T@b/len(y),d.numpy()+b.T@y/len(y))
        np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-12)

    @unittest.skipIf(torch is None,'Torch absent')
    def test_existing_head_materialization_identity(self):
        root=Path(__file__).resolve().parents[1];path=root/'artifacts/task41n_frozen.zip.bin'
        if not path.exists():self.skipTest('Frozen checkpoint absent')
        os.environ['GIADA_TASK41N_FROZEN']=str(path);base=r.n.load_models(torch,root,'cpu')[0]['dual']
        weights=r.head_weights(base);net=r.materialize(torch,base,weights)
        self.assertEqual(r.n.weights_hash(base),r.n.weights_hash(net))
        weights['h1_12']=weights['h1_12']+.00001;net=r.materialize(torch,base,weights)
        self.assertEqual(sum(x.numel() for x in base.parameters()),sum(x.numel() for x in net.parameters()))
        for key,value in base.state_dict().items():
            if key not in ('parent.c','parent.d'):self.assertTrue(torch.equal(value,net.state_dict()[key]))

    @unittest.skipIf(torch is None,'Torch absent')
    def test_saved_q_control_is_bit_identical_and_k_factor_isolated(self):
        root=Path(__file__).resolve().parents[1];folder=root/'experiments/results/task41r_protocol_v1'
        if not (folder/'fit_manifest.json').exists():self.skipTest('Prepared fits absent')
        os.environ['GIADA_TASK41N_FROZEN']=str(root/'artifacts/task41n_frozen.zip.bin');base=r.n.load_models(torch,root,'cpu')[0]['dual']
        with np.load(root/'experiments/results/task41q_audit_v1/direct_weights_ridge_1e-06.npz') as z:
            control=r.p.materialize(torch,base,torch.tensor(z['wv']),torch.tensor(z['wc']),'joint_nap')
        with np.load(folder/'heads_coefficient_frozen_k.npz') as z:uniform=r.materialize(torch,base,dict(z))
        self.assertEqual(r.n.weights_hash(control),r.n.weights_hash(uniform))
        with np.load(folder/'heads_coefficient_joint_k.npz') as z:joint=r.materialize(torch,base,dict(z))
        for col in range(34):
            if col not in (29,30):
                self.assertTrue(torch.equal(uniform.parent.c[:,:,col],joint.parent.c[:,:,col]))
                self.assertTrue(torch.equal(uniform.parent.d[:,col],joint.parent.d[:,col]))

    def test_dispatch_notebook_has_kernel_and_correct_scope(self):
        import sys
        root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
        import task41r_kaggle as k
        doc=k.notebook('a'*40,'b'*64);code=''.join(doc['cells'][1]['source'])
        self.assertEqual(doc['metadata']['kernelspec']['name'],'python3');compile(code,'41r','exec')
        self.assertIn('scripts/run_task41r.py',code);self.assertIn('development_nomination=True',code)
        self.assertNotIn('selection_recomputed=False',code)

if __name__=='__main__':unittest.main()
