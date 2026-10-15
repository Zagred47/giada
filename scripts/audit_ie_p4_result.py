"""Read-only IE-P4 audit; optionally save an audit JSON in the result folder."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def audit():
    import numpy as np
    import torch
    from src.giada_teacher import ie_discrete_ionic as p4
    torch.set_num_threads(1)
    folder=ROOT/'experiments/results/ie_p4_kaggle_v1'
    r=json.loads((folder/'final_report.json').read_text());spec=json.loads((folder/'preregistration.json').read_text())
    launch=json.loads((ROOT/'experiments/ie_p4_kaggle_launch_v1.json').read_text())
    assert r['valid'] and r['code_revision']==launch['code_revision']
    assert (folder/'preregistration.json').read_bytes()==(ROOT/p4.CONFIG).read_bytes()
    assert len(r['rows'])==84 and len(r['preflights'])==84 and all(x['valid'] for x in r['preflights']) and not r['graph_failures']
    assert not r['candidate_promoted'] and not r['deployment_authorized']
    assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest()==r['selection']['checkpoint_sha256']
    assert json.loads((folder/'selection_freeze.json').read_text())==r['selection']
    for name,key in [('fresh_evaluation.json','fresh'),('historical_evaluation.json','historical'),('validation.json','validation')]:
        assert json.loads((folder/name).read_text())==r[key]
    curves=json.loads((folder/'training_curves.json').read_text())
    for k in range(12):
        best=min(curves,key=lambda x:x['development_normalized_transition_rmse'][k])
        assert best['step']==r['selection']['selected_steps'][k]
        assert best['development_normalized_transition_rmse'][k]==r['selection']['development_scores'][k]
    for row in r['rows']:
        values=row['timings']['wall_ms'];assert len(values)==30 and all(np.isfinite(values)) and min(values)>0
        reference=next(x for x in r['rows'] if x['batch']==row['batch'] and x['mode']==row['mode'] and x['name']=='formula_vectorized')['timings']['wall_ms']
        assert abs(statistics.median(1-b/a for a,b in zip(reference,values))-row['saving_vs_formula']['median_saved_fraction'])<1e-12
    q=spec['quality']
    for name,f in r['fresh'].items():
        h=r['historical'][name]
        valid=(f['fresh_gate']['rmse']<=q['fresh_gate_rmse'] and f['fresh_gate']['max_abs']<=q['fresh_gate_max_abs'] and
            all(x['rmse']<=q['forced_path_gate_rmse'] and x['max_abs']<=q['forced_path_gate_max_abs'] for x in f['families'].values()) and
            h['voltage']['rmse']<=q['historical_voltage_rmse_mv'] and h['voltage']['max_abs']<=q['historical_voltage_max_abs_mv'] and
            h['calcium']['max_abs']<=q['historical_calcium_max_abs_mm'] and h['voltage_domain_exit_count']==0 and f['occupancy_violations']==0)
        assert bool(valid)==r['quality_qualified'][name]
    saved=np.load(folder/'fresh_imposed_paths.npz')
    v0,x0,families=p4.path_data()
    assert np.allclose(saved['voltage_mv'],v0,atol=1e-12) and np.array_equal(saved['initial_gates'],x0)
    assert list(saved['families'])==families
    v=torch.tensor(saved['voltage_mv'],dtype=torch.float32);initial=torch.tensor(x0,dtype=torch.float32)
    state=torch.load(folder/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)
    model=p4.ensemble(torch,state['scale'],'cpu');model.load_state_dict(state)
    ops={'formula_vectorized':None,'lut025':p4.lut_operator(torch,device='cpu')}
    for k,(loss,width,seed) in enumerate(p4.MEMBERS):ops[f'{loss}_w{width}_seed{seed}']=p4.operator(torch,model,k)
    def rollout(u,z):
        x=initial;values=[]
        for i in range(400):x=(1-z[i])*x+z[i]*u[i];values.append(x)
        return torch.stack(values)
    reference=rollout(*p4.coefficients(torch,v));maximum_rmse_error=0.;maximum_max_error=0.
    with torch.no_grad():
        for name,op in ops.items():
            actual=rollout(*(p4.coefficients(torch,v) if op is None else op(v)))
            error=actual-reference;metrics={'rmse':float(error.square().mean().sqrt()),'max_abs':float(error.abs().max())}
            maximum_rmse_error=max(maximum_rmse_error,abs(metrics['rmse']-r['fresh'][name]['forced_path_gate']['rmse']))
            maximum_max_error=max(maximum_max_error,abs(metrics['max_abs']-r['fresh'][name]['forced_path_gate']['max_abs']))
    assert maximum_rmse_error<2e-5 and maximum_max_error<2e-5
    contrasts=[]
    for width in (32,64):
        for seed in (17,29,43):
            a=f'transition_w{width}_seed{seed}';b=f'internal_w{width}_seed{seed}'
            contrasts.append({'width':width,'seed':seed,
                'fresh_improved':r['fresh'][b]['fresh_gate']['rmse']<r['fresh'][a]['fresh_gate']['rmse'],
                'forced_path_improved':r['fresh'][b]['forced_path_gate']['rmse']<r['fresh'][a]['forced_path_gate']['rmse'],
                'historical_voltage_improvement_factor':r['historical'][a]['voltage']['rmse']/r['historical'][b]['voltage']['rmse']})
    return {'valid':True,'code_revision':r['code_revision'],'timing_cells_verified':84,'numeric_preflights_verified':84,
        'freeze_checkpoint_verified':True,'checkpoint_selection_verified':True,'all_quality_flags_recomputed':True,
        'fresh_paths_regenerated':True,'forced_path_members_replayed_cpu':14,
        'maximum_cpu_gpu_forced_rmse_discrepancy':maximum_rmse_error,'maximum_cpu_gpu_forced_max_discrepancy':maximum_max_error,
        'replay_atol':2e-5,'paired_internal_supervision_contrasts':contrasts,
        'retraining_performed':False,'native_confirmation_claimed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--json',type=Path);args=parser.parse_args()
    result=audit()
    if args.json:args.json.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
