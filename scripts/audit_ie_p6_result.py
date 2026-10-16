"""Read-only IE-P6 integrity, selection and independent CPU path replay."""
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
    from src.giada_teacher import ie_residual_repair as p6
    torch.set_num_threads(1);folder=ROOT/'experiments/results/ie_p6_kaggle_v1'
    r=json.loads((folder/'final_report.json').read_text());spec=json.loads((folder/'preregistration.json').read_text())
    launch=json.loads((ROOT/'experiments/ie_p6_kaggle_launch_v1.json').read_text())
    assert r['valid'] and r['code_revision']==launch['expected_code_revision']==(folder/'code_revision.txt').read_text().strip()
    assert (folder/'preregistration.json').read_bytes()==(ROOT/p6.CONFIG).read_bytes()
    assert len(r['rows'])==210 and len(r['preflights'])==210 and all(x['valid'] for x in r['preflights']) and not r['graph_failures']
    assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest()==r['selection']['checkpoint_sha256']
    assert json.loads((folder/'selection_freeze.json').read_text())==r['selection'] and not r['selection']['fresh_accessed']
    for file,key in [('fresh_evaluation.json','fresh'),('historical_evaluation.json','historical'),('conversion_report.json','conversions'),('validation.json','validation')]:
        assert json.loads((folder/file).read_text())==r[key]
    assert not r['candidate_promoted'] and not r['deployment_authorized']
    curves=json.loads((folder/'training_curves.json').read_text())
    for key,steps in r['selection']['selected_steps'].items():
        for k,step in enumerate(steps):
            best=min(curves,key=lambda x:x['models'][key]['score'][k])
            assert step==best['step'] and r['selection']['development_scores'][key][k]==best['models'][key]['score'][k]
    for row in r['rows']:
        times=row['timings']['wall_ms'];assert len(times)==30 and min(times)>0 and np.isfinite(times).all()
        control=next(x for x in r['rows'] if x['batch']==row['batch'] and x['mode']==row['mode'] and x['name']=='formula_vectorized')['timings']['wall_ms']
        recomputed=p6.p4.paired_summary(control,times)
        assert recomputed==row['saving_vs_formula']
    for name,f in r['fresh'].items():
        assert p6.qualify(f,r['historical'][name],spec['quality'],r['conversions'].get(name,{}).get('conversion_passed',True))==r['quality_qualified'][name]
    parent=torch.load(ROOT/spec['parent_checkpoint'],map_location='cpu',weights_only=True)
    original=p6.dense(torch,parent,'cpu');models={'dense':p6.dense(torch,parent,'cpu'),**{f'local{k}':p6.local(torch,parent,k,'cpu') for k in (81,161)}}
    checkpoint=torch.load(folder/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)
    for key,m in models.items():m.load_state_dict(checkpoint[key])
    ops={'formula_vectorized':None,'lut025':p6.p4.lut_operator(torch,device='cpu')}
    for k,s in enumerate(spec['seeds']):ops[f'parent_seed{s}']=p6.p5.operator(torch,original,k)
    tables=torch.load(folder/'neural_deployment_tables.pt',map_location='cpu',weights_only=True);table_error=0.
    for k,(arm,s) in enumerate(p6.MEMBERS):
        name=f'{arm}_seed{s}';op=p6.p5.operator(torch,models['dense'],k);ops[name]=op
        table_op,table=p6.p5.tabulate(torch,op,.25,'cpu');ops[name+'_table025']=table_op
        table_error=max(table_error,float((table-tables[name+'_table025']).abs().max()))
    assert table_error<2e-6
    for knots in (81,161):
        for k,s in enumerate(spec['seeds']):
            m=models[f'local{knots}'];assert bool(((m.table>=0)&(m.table<=1)).all())
            ops[f'local{knots}_seed{s}']=p6.local_operator(torch,m,k)
    saved=np.load(folder/'fresh_imposed_paths.npz');v,x,f=p6.path_data(spec['path_seed'])
    regeneration_delta=float(np.max(np.abs(v-saved['voltage_mv'])))
    assert regeneration_delta<1e-12 and np.array_equal(x,saved['initial_gates']) and list(saved['families'])==f
    vv=torch.tensor(saved['voltage_mv'],dtype=torch.float32);xx=torch.tensor(x,dtype=torch.float32);reference=p6.rollout(torch,*p6.p4.coefficients(torch,vv),xx);maximum=0.
    with torch.no_grad():
        for name,op in ops.items():
            prediction=p6.rollout(torch,*(p6.p4.coefficients(torch,vv) if op is None else op(vv)),xx);diff=prediction-reference
            for key,value in [('rmse',float(diff.square().mean().sqrt())),('max_abs',float(diff.abs().max()))]:
                maximum=max(maximum,abs(value-r['fresh'][name]['forced_path_gate'][key]))
    assert maximum<2e-5
    oracle=json.loads((folder/'diagnostic_oracle_attribution.json').read_text());assert not oracle['selection_eligible'] and oracle['parent']=='IE-P5 standard' and len(oracle['metrics'])==18
    primary=[]
    for row in r['rows']:
        if row['batch']==642 and row['mode']=='cuda_graph' and row['name'].startswith('local'):
            saving=row['saving_vs_formula'];assert r['quality_qualified'][row['name']] and saving['median_saved_fraction']>=.1 and saving['bootstrap_95pct'][0]>0
            primary.append(row['name'])
    assert len(primary)==6
    return {'valid':True,'code_revision':r['code_revision'],'timing_cells_verified':210,'preflights_verified':210,'selection_verified':True,'all_quality_flags_recomputed':True,
        'forced_path_maps_replayed_cpu':len(ops),'path_regeneration_max_delta_mv':regeneration_delta,'cpu_gpu_path_metric_max_discrepancy':maximum,'neural_table_values_max_discrepancy':table_error,'oracle_rows_verified':18,
        'local_candidates_passing_registered_quality_and_graph_cost':primary,'retraining_performed':False,'new_native_confirmation_claimed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--json',type=Path);args=parser.parse_args();result=audit()
    if args.json:args.json.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
