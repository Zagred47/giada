import copy,json,os,tempfile,unittest,sys
from pathlib import Path
import numpy as np,torch
from src.giada_teacher import task41v_targeted_coefficients as v
ROOT=Path(__file__).resolve().parents[1]
class Task41v(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin');base=v.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(ROOT/'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz') as a:cls.inc=v.r.materialize(torch,base,dict(a))
    def test_masks_and_composition(self):
        grid,labels=v.curve_labels(torch,'cpu');models={}
        for channel in ('k','nap'):
            for local in (False,True):
                model=copy.deepcopy(self.inc);params,masks=v.configure(torch,model,v.TARGETS[channel],local)
                loss=v.coefficient_loss(torch,model,grid,labels,v.TARGETS[channel]);loss.sum().backward()
                with torch.no_grad():
                    for key,p in params:p.add_(1e-6*masks[key])
                for key,val in model.state_dict().items():
                    if key not in masks:self.assertTrue(torch.equal(val,self.inc.state_dict()[key]))
                    else:self.assertTrue(torch.equal(val*(1-masks[key]),self.inc.state_dict()[key]*(1-masks[key])))
                models[(channel,local)]=model
        combined=v.compose(torch,self.inc,models[('k',True)],models[('nap',False)])
        self.assertEqual(sum(p.numel() for p in combined.parameters()),sum(p.numel() for p in self.inc.parameters()))
        self.assertTrue(torch.equal(combined.parent.correction[...,2],models[('k',True)].parent.correction[...,2]))
        self.assertTrue(torch.equal(combined.parent.c[...,8],models[('nap',False)].parent.c[...,8]))
    def test_protocol_and_challenge(self):
        spec=json.loads((ROOT/'experiments/task41v_preregistration_v1.json').read_text());old=json.loads((ROOT/'experiments/task41u_preregistration_v1.json').read_text());self.assertEqual(spec['quality'],old['quality'])
        prior=v.n.contract(ROOT)[0];original=v.n.stress_data
        with v.challenge_paths(spec):rows=v.n.stress_data(dict(prior,stress_seeds=spec['development_seeds']));self.assertEqual(len(rows),48)
        self.assertIs(v.n.stress_data,original);self.assertEqual(spec['arms'],['baseline','incumbent',*v.ARMS])
    def test_gpu_oracle_wrapper_fixture(self):
        prior=v.n.contract(ROOT)[0];original=v.r.stress;seen=[]
        def fake(torch,model,spec,out,label,device):
            m=v.n.stress_data(prior)[0];vv=torch.full((3,2),float(np.float32(m['v'][0])));ca=torch.full_like(vv,1e-4);state=torch.full((3,2,18),.3)
            result=model(vv,ca,state);self.assertTrue(torch.isfinite(result).all());seen.append(label)
            if label=='gpu_oracle_K_u12':
                target,_=v.n.p.coefficients(vv.numpy().astype('float64'),ca.numpy().astype('float64'),.1)
                with torch.no_grad():_,raw_z=self.inc.raw(vv,ca)
                expected=state[...,12]+torch.sigmoid(raw_z[...,12])*(torch.tensor(target[...,12],dtype=torch.float32)-state[...,12])
                torch.testing.assert_close(result[...,12],expected,rtol=0,atol=0)
            return dict(passed=66,total=72,failed=[])
        try:
            v.r.stress=fake
            with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:v.gpu_oracles(torch,self.inc,prior,Path(temp),'cpu')
        finally:v.r.stress=original
        self.assertEqual(len(seen),6)
    def test_checkpoint_selection_fixture(self):
        spec=json.loads((ROOT/'experiments/task41v_preregistration_v1.json').read_text());spec=dict(spec,fit_path_seeds=[101,103,107],development_seeds=[201,203,207],challenge_seeds=[301,303,307],checkpoints=[1])
        prior=dict(v.n.contract(ROOT)[0],stress_duration_ms=60.)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            result=v.train(torch,self.inc,prior,spec,Path(temp),'cpu');self.assertEqual(list(result),list(v.ARMS))
            p=torch.load(Path(temp)/'checkpoint_1.pt',map_location='cpu',weights_only=True);self.assertEqual(len(p['models']),10)
            self.assertTrue((Path(temp)/'selection_recomposed.json').is_file())
    def test_dispatch(self):
        sys.path.insert(0,str(ROOT/'scripts'));import task41v_kaggle as k
        doc=k.notebook('a'*40,'b'*64);code=''.join(doc['cells'][1]['source']);compile(code,'41v','exec');self.assertIn('run_task41v.py',code)
if __name__=='__main__':unittest.main()
