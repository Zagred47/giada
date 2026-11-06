"""Fit-only prediction/actual gap. Does not open any behavioral evaluation panel."""
import prepare_task41s as s
import os,json
import numpy as np
import torch

def main():
    torch.set_num_threads(1);root=s.ROOT;folder=root/'experiments/results/task41s_protocol_v1'
    spec=json.loads((root/'experiments/task41s_preregistration_v1.json').read_text())
    prior=s.r.n.contract(root)[0]
    os.environ['GIADA_TASK41N_FROZEN']=str(root/'artifacts/task41n_frozen.zip.bin')
    base=s.r.n.load_models(torch,root,'cpu')[0]['dual']
    with np.load(folder/'heads_incumbent.npz') as a:start=dict(a)
    rows=[];events={}
    for window,name in [(False,'crossing_single'),(True,'window_single')]:
        initial,ev=s.crossing_rows(torch,s.r.materialize(torch,base,start),prior,spec['fit_path_seeds'],window,False)
        with np.load(folder/f'heads_{name}.npz') as a:weights=dict(a)
        actual,_=s.crossing_rows(torch,s.r.materialize(torch,base,weights),prior,spec['fit_path_seeds'],window,False)
        events[name]=ev
        for channel,(b,y) in initial.items():
            keys=s.r.keys_for(channel,True)
            for seed in range(3):
                displacement=s.unpack(weights,start,keys,seed).numpy()
                predicted=y[seed]-b[seed]@displacement
                rows.append(dict(arm=name,channel=channel,seed=(17,29,43)[seed],
                    baseline_mse=float(np.mean(y[seed]**2)),predicted_mse=float(np.mean(predicted**2)),
                    actual_mse=float(np.mean(actual[channel][1][seed]**2)),
                    linearization_gap_rmse=float(np.sqrt(np.mean((predicted-actual[channel][1][seed])**2)))))
    value=dict(valid=True,fit_only=True,development_accessed=False,opened_stress_accessed=False,rows=rows,reference_slope_events=events)
    s.write_json(folder/'fit_linearization_audit.json',value)
    print('[GIADA41s] fit-only nonlinear gap audit saved',flush=True)
if __name__=='__main__':main()
