"""Local schema/gradient smoke test; no scientific model trained or development opened."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.giada_teacher import task41u_objective_features as u
from research_memory.mirror import write_json

def main():
    torch.set_num_threads(1);folder=ROOT/'experiments/results/task41u_protocol_v1';folder.mkdir(exist_ok=True)
    spec=json.loads((ROOT/'experiments/task41u_preregistration_v1.json').read_text());prior=u.n.contract(ROOT)[0]
    assert prior['quality']==spec['quality'];used=prior['stress_seeds']+prior['native_seeds']
    for letter in ('r','s','t'):
        previous=json.loads((ROOT/f'experiments/task41{letter}_preregistration_v1.json').read_text());used+=previous.get('fit_path_seeds',[])+previous.get('guard_path_seeds',[])+previous['development_seeds']
    assert not set(spec['fit_path_seeds']+spec['development_seeds'])&set(used)
    os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin');base=u.n.load_models(torch,ROOT,'cpu')[0]['dual']
    path=ROOT/'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz'
    with np.load(path) as a:inc=u.r.materialize(torch,base,dict(a))
    # Tiny local fixture, not a shortened scientific protocol or new training arm.
    fixture=dict(prior,stress_duration_ms=200.)
    datasets={flag:u.tensor_data(torch,u.data_arrays(fixture,spec['fit_path_seeds'],flag),'cpu') for flag in (False,True)}
    checks=[]
    for features in (False,True):
        for margin in (False,True):
            model=__import__('copy').deepcopy(inc);params=u.set_trainable(torch,model,features)
            loss=u.loss_for(torch,model,datasets[False],torch.arange(2),lambda *args:u.objective(torch,*args),torch.tensor(float(margin)))
            assert torch.isfinite(loss).all();loss.sum().backward();u.mask_and_clip(torch,model,params,1.)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            checks.append(dict(features=features,margin=margin,gradient_finite=True,total_parameters=sum(p.numel() for p in model.parameters())))
    manifest=dict(valid=True,baseline_weights_sha256=u.n.weights_hash(base),incumbent_weights_sha256=u.n.weights_hash(inc),incumbent_head_file_sha256=u.n.p.digest(path),scientific_training_performed_locally=False,development_accessed=False)
    write_json(folder/'fit_manifest.json',manifest);write_json(folder/'local_preflight.json',dict(valid=True,checks=checks,scope='CPU gradient fixtures only, no optimizer step/no development/no openedstress'))
    receipt=json.loads((ROOT/'experiments/results/task41o_causal_v1/input_verified.json').read_text());write_json(folder/'input_verified.json',receipt)
    print('[GIADA41u] local2x2 gradient preflight passed; frozen incumbent unchanged',flush=True)
if __name__=='__main__':main()
