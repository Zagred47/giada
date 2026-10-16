"""IE-P5 read-only audit and CPU replay; no fitting or candidate selection."""
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
    from src.giada_teacher import ie_accuracy_execution as p5
    torch.set_num_threads(1);folder=ROOT/'experiments/results/ie_p5_kaggle_v1'
    r=json.loads((folder/'final_report.json').read_text());spec=json.loads((folder/'preregistration.json').read_text())
    launch=json.loads((ROOT/'experiments/ie_p5_kaggle_launch_v1.json').read_text())
    assert r['valid'] and r['code_revision']==launch['code_revision']
    assert (folder/'preregistration.json').read_bytes()==(ROOT/p5.CONFIG).read_bytes()
    assert len(r['rows'])==138 and len(r['preflights'])==138 and all(x['valid'] for x in r['preflights']) and not r['graph_failures']
    assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest()==r['selection']['checkpoint_sha256']
    assert json.loads((folder/'selection_freeze.json').read_text())==r['selection']
    for name,key in [('fresh_evaluation.json','fresh'),('historical_evaluation.json','historical'),('conversion_report.json','conversions'),('validation.json','validation')]:
        assert json.loads((folder/name).read_text())==r[key]
    assert not r['candidate_promoted'] and not r['deployment_authorized'] and not r['diagnostic_oracles_selection_eligible']
    curves=json.loads((folder/'training_curves.json').read_text())
    for k in range(6):
        best=min(curves,key=lambda x:x['development_score'][k])
        assert best['step']==r['selection']['selected_steps'][k] and best['development_score'][k]==r['selection']['development_scores'][k]
    for row in r['rows']:
        times=row['timings']['wall_ms'];assert len(times)==30 and min(times)>0 and np.isfinite(times).all()
        control=next(x for x in r['rows'] if x['batch']==row['batch'] and x['mode']==row['mode'] and x['name']=='formula_vectorized')['timings']['wall_ms']
        assert abs(statistics.median(1-b/a for a,b in zip(control,times))-row['saving_vs_formula']['median_saved_fraction'])<1e-12
    q=spec['quality']
    for name,f in r['fresh'].items():
        h=r['historical'][name]
        valid=(f['fresh_gate']['rmse']<=q['fresh_gate_rmse'] and f['fresh_gate']['max_abs']<=q['fresh_gate_max_abs'] and
            all(x['rmse']<=q['forced_path_gate_rmse'] and x['max_abs']<=q['forced_path_gate_max_abs'] for x in f['families'].values()) and
            h['voltage']['rmse']<=q['historical_voltage_rmse_mv'] and h['voltage']['max_abs']<=q['historical_voltage_max_abs_mv'] and
            h['calcium']['max_abs']<=q['historical_calcium_max_abs_mm'] and not h['voltage_domain_exit_count'] and not f['occupancy_violations'] and
            (name not in r['conversions'] or r['conversions'][name]['conversion_passed']))
        assert bool(valid)==r['quality_qualified'][name]
    parent=torch.load(ROOT/spec['parent_checkpoint'],map_location='cpu',weights_only=True)
    original=p5.ensemble(torch,parent,'cpu');model=p5.ensemble(torch,parent,'cpu')
    model.load_state_dict(torch.load(folder/'frozen_checkpoint.pt',map_location='cpu',weights_only=True))
    ops={'formula_vectorized':None,'lut025':p5.p4.lut_operator(torch,device='cpu')}
    for k,seed in enumerate(spec['seeds']):ops[f'frozen64_seed{seed}']=p5.operator(torch,original,k)
    saved_tables=torch.load(folder/'neural_deployment_tables.pt',map_location='cpu',weights_only=True)
    table_error=0.
    for k,(loss,seed) in enumerate(p5.MEMBERS):
        name=f'refit_{loss}_seed{seed}';op=p5.operator(torch,model,k);ops[name]=op
        for spacing,label in ((.25,'table025'),(1.,'table1')):
            key=name+'_'+label;compiled,table=p5.tabulate(torch,op,spacing,'cpu');ops[key]=compiled
            table_error=max(table_error,float((table-saved_tables[key]).abs().max()))
    assert table_error<2e-6
    saved=np.load(folder/'fresh_imposed_paths.npz');pv,x0,families=p5.path_data()
    assert np.allclose(pv,saved['voltage_mv'],atol=1e-12) and np.array_equal(x0,saved['initial_gates']) and list(saved['families'])==families
    v=torch.tensor(pv,dtype=torch.float32);initial=torch.tensor(x0,dtype=torch.float32)
    def rollout(u,z):
        state=initial;values=[]
        for t in range(400):state=(1-z[t])*state+z[t]*u[t];values.append(state)
        return torch.stack(values)
    reference=rollout(*p5.p4.coefficients(torch,v));maximum=0.
    with torch.no_grad():
        for name,op in ops.items():
            values=rollout(*(p5.p4.coefficients(torch,v) if op is None else op(v)));diff=values-reference
            for key,value in [('rmse',float(diff.square().mean().sqrt())),('max_abs',float(diff.abs().max()))]:
                maximum=max(maximum,abs(value-r['fresh'][name]['forced_path_gate'][key]))
    assert maximum<2e-5
    attribution=json.loads((folder/'diagnostic_oracle_attribution.json').read_text())
    assert not attribution['selection_eligible'] and len(attribution['metrics'])==18
    contrasts=[]
    for seed in spec['seeds']:
        a=f'refit_internal_seed{seed}';b=f'refit_balanced_seed{seed}';f=f'frozen64_seed{seed}'
        contrasts.append({'seed':seed,'standard_vs_frozen_voltage_improvement_factor':r['historical'][f]['voltage']['rmse']/r['historical'][a]['voltage']['rmse'],
            'balanced_better_fresh':r['fresh'][b]['fresh_gate']['rmse']<r['fresh'][a]['fresh_gate']['rmse'],
            'balanced_better_path':r['fresh'][b]['forced_path_gate']['rmse']<r['fresh'][a]['forced_path_gate']['rmse'],
            'balanced_better_voltage':r['historical'][b]['voltage']['rmse']<r['historical'][a]['voltage']['rmse'],
            'z_oracle_voltage_improvement_factor':r['historical'][f]['voltage']['rmse']/attribution['metrics'][f+'_z_all']['voltage']['rmse']})
    return {'valid':True,'code_revision':r['code_revision'],'timing_cells_verified':138,'preflights_verified':138,'freeze_selection_verified':True,
        'all_quality_flags_recomputed':True,'forced_path_maps_replayed_cpu':23,'cpu_gpu_path_metric_max_discrepancy':maximum,
        'neural_table_values_max_discrepancy':table_error,'attribution_rows_verified':18,'paired_contrasts':contrasts,
        'retraining_performed':False,'native_confirmation_claimed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--json',type=Path);args=parser.parse_args();result=audit()
    if args.json:args.json.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
