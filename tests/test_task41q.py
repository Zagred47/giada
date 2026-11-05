import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:import torch
except ImportError:torch=None
import audit_task41q as q

@unittest.skipIf(torch is None,'Torch not installed')
class DiagnosticFit(unittest.TestCase):
    def test_known_small_same_feature_repair(self):
        x=torch.linspace(-1,1,300,dtype=torch.float64)
        f=torch.stack((torch.ones_like(x),x),-1);raw=.2+.1*x
        target=torch.sigmoid(raw+.001-.002*x)
        delta,info=q.ridge_delta(torch,f,raw,target,1e-6)
        old=(torch.sigmoid(raw)-target).square().mean()
        new=(torch.sigmoid(raw+f@delta)-target).square().mean()
        self.assertLess(float(new),float(old)*1e-4);self.assertEqual(info['effective_rank'],2)

    def test_zero_residual_returns_zero_delta(self):
        f=torch.eye(5,dtype=torch.float64);raw=torch.ones(5,dtype=torch.float64)
        delta,_=q.ridge_delta(torch,f,raw,torch.sigmoid(raw),1e-4)
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))

    def test_no_features_cannot_establish_capacity(self):
        f=torch.zeros((10,3),dtype=torch.float64);raw=torch.zeros(10,dtype=torch.float64)
        delta,info=q.ridge_delta(torch,f,raw,torch.full_like(raw,.6),1e-4)
        self.assertEqual(info['effective_rank'],0);self.assertTrue(torch.isfinite(delta).all())

    def test_fixed_rate_bound_zero_for_exact_rates(self):
        data=dict(u=torch.full((10,6),.3,dtype=torch.float64),z=torch.full((10,6),.001,dtype=torch.float64),base_z=torch.full((3,10,6),.001,dtype=torch.float64),initial=torch.full((10,6),.8,dtype=torch.float64))
        result=q.fixed_rate_lower_bound(data)
        self.assertLess(max(result['equilibrium_total_per_seed']),1e-20)
        data['base_z']*=1.01
        result=q.fixed_rate_lower_bound(data)
        self.assertGreater(min(result['equilibrium_total_per_seed']),0)
        self.assertLess(result['joint_total_per_seed'][0],result['equilibrium_total_per_seed'][0])

if __name__=='__main__':unittest.main()
