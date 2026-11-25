import copy,json,tempfile,unittest
from pathlib import Path
import torch
from tests.test_task52g import config,ROOT
from src.giada_teacher import task52g_gradient_preservation as g
from src.giada_teacher import task52i_solution_retention as i
class Retention(unittest.TestCase):
    def test_contract(self):
        p=i.plan(ROOT);self.assertEqual(p['warmstart']['indices'],[6,7,8]);self.assertEqual(p['warmstart']['selected_updates'],[300,0,300]);self.assertEqual([(a['decay'],a['acceptance_guard']) for a in p['arms']],[(False,False),(True,False),(False,True),(True,True)])
    def test_small_guarded_training(self):
        torch.set_num_threads(1);p=config();full=i.plan(ROOT)
        for arm,other in zip(p['arms'],full['arms']):arm.update(decay=other['decay'],acceptance_guard=other['acceptance_guard'])
        p['step_acceptance']=dict(full['step_acceptance'],ordinary_cases=1,support_cases=1,start_ms=0,ticks=4)
        net,cal,_=g.f.load_model(torch,p,ROOT/'artifacts/task52e_input/models52e.zip.bin','cpu');original=copy.deepcopy(net)
        data0=g.f.c.b.data(torch,p,'cpu');data=g.f.merge_data(torch,data0,g.f.support_data(torch,p,'cpu'));aut=g.f.d.coupled_data(torch,p,ROOT,'cpu');cal=dict(cal,maxop=data['op'].amax((0,1)).clamp_min(1e-30))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            out=Path(tmp);g.train(torch,p,net,data,aut,cal,out,len(data0['meta']),baseline_model=original)
            r=json.loads((out/'final_step_guard_errors.json').read_text());err=torch.tensor(r['errors']);base=torch.tensor(r['baseline']);mask=torch.tensor(r['guarded']);self.assertTrue((err[mask]<=base[mask]*(1+1e-6)+1e-12).all());self.assertEqual(len(json.loads((out/'step_acceptance_history.json').read_text())),2)
if __name__=='__main__':unittest.main()
