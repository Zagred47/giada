import copy,json,tempfile,unittest
from pathlib import Path
import torch
from tests.test_task52g import config,ROOT
from src.giada_teacher import task52g_gradient_preservation as g
class FinalGuard(unittest.TestCase):
    def test_coupled_guard_smoke(self):
        torch.set_num_threads(1);p=config();p['step_acceptance']=dict(ordinary_cases=1,support_cases=1,start_ms=0,ticks=2,long_ticks=4,coupled_cases=1,coupled_start_ms=0,coupled_ticks=4,trial_factors=[1,.5],minimum_lr_fraction=.05)
        for k,a in enumerate(p['arms']):a.update(decay=True,acceptance_guard=True,long_guard=bool(k%2),coupled_guard=bool(k//2))
        net,cal,_=g.f.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');base=copy.deepcopy(net);ordinary=g.f.c.b.data(torch,p,'cpu');data=g.f.merge_data(torch,ordinary,g.f.support_data(torch,p,'cpu'));aut=g.f.d.coupled_data(torch,p,ROOT,'cpu');cal=dict(cal,maxop=data['op'].amax((0,1)).clamp_min(1e-30))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp);g.train(torch,p,net,data,aut,cal,out,len(ordinary['meta']),baseline_model=base);r=json.loads((out/'final_step_guard_errors.json').read_text());v=torch.tensor(r['errors']);b=torch.tensor(r['baseline']);self.assertEqual(v.shape,(12,13));self.assertTrue((v<=b*(1+1e-6)+1e-12).all())
if __name__=='__main__':unittest.main()
