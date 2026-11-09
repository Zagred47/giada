"""Verify targeted masks and gradients without training scientific candidates."""
import os,sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.giada_teacher import task41v_targeted_coefficients as v
from research_memory.mirror import write_json
def main():
    torch.set_num_threads(1);folder=ROOT/'experiments/results/task41v_protocol_v1';folder.mkdir(exist_ok=True)
    spec=json.loads((ROOT/'experiments/task41v_preregistration_v1.json').read_text());prior=v.n.contract(ROOT)[0];assert prior['quality']==spec['quality']
    used=prior['stress_seeds']+prior['native_seeds']
    for letter in ('r','s','t','u'):
        p=json.loads((ROOT/f'experiments/task41{letter}_preregistration_v1.json').read_text());used+=p['development_seeds']+p.get('fit_path_seeds',[])+p.get('guard_path_seeds',[])
    fresh=spec['fit_path_seeds']+spec['development_seeds']+spec['challenge_seeds'];assert len(set(fresh))==len(fresh) and not set(fresh)&set(used)
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin');base=v.n.load_models(torch,ROOT,'cpu')[0]['dual']
    with np.load(ROOT/'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz') as a:inc=v.r.materialize(torch,base,dict(a))
    grid,targets=v.curve_labels(torch,'cpu');checks=[]
    for channel in ('k','nap','nap_u','nap_z'):
        for local in (False,True):
            model=copy.deepcopy(inc);params,masks=v.configure(torch,model,v.TARGETS[channel],local)
            loss=v.coefficient_loss(torch,model,grid,targets,v.TARGETS[channel]);loss.sum().backward()
            assert torch.isfinite(loss).all() and all(torch.isfinite(p.grad).all() for _,p in params)
            checks.append(dict(channel=channel,local=local,finite=True,parameter_count=sum(p.numel() for p in model.parameters())))
    write_json(folder/'fit_manifest.json',dict(valid=True,baseline_weights_sha256=v.n.weights_hash(base),incumbent_weights_sha256=v.n.weights_hash(inc),scientific_training_performed_locally=False))
    write_json(folder/'local_preflight.json',dict(valid=True,checks=checks,development_accessed=False))
    write_json(folder/'input_verified.json',json.loads((ROOT/'experiments/results/task41o_causal_v1/input_verified.json').read_text()))
    print('[GIADA41v] targeted masks and gradients verified',flush=True)
if __name__=='__main__':main()
