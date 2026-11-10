"""Frozen level-one transfer; paired node-order audit, no model selection."""
import json
import shutil
from contextlib import contextmanager
import numpy as np
from . import task42_axial_exploration as x

def spec(root):
    value=json.loads((root/'experiments/task43_preregistration_v1.json').read_text())
    old,cfg,ca,base=x.n.contract(root)
    assert value['quality']==old['quality'] and value['compartment_gates']==cfg['gates']
    assert value['passive_gates']==x.passive.config(root,'iv_ad1_passive_axial_preregistration_v3.json')['gates']
    assert value['seeds']==[17,29,43] and value['dt_ms']==.1
    assert not value['thresholds_changed'] and value['training_updates']==0
    return value,cfg,ca,base

def design(geo,config):
    count=len(geo['parents']);clock=np.arange(round(config['duration_ms']/.1))*.1
    rows=[];injections=[]
    for coupling in config['coupling']:
        for protocol in config['protocols']:
            injection=np.zeros((len(clock),count))
            # Local conditioning creates distinct states before the challenge.
            for site in range(count):
                start=1.+2.*(site%4)
                injection[(clock>=start)&(clock<start+4.),site]=.004*(1+site%3)
            if protocol=='recovery':
                injection[(clock>=30)&(clock<42),0]=.016
                injection[(clock>=80)&(clock<92),count-1]=.016
            elif protocol=='staircase':
                for j in range(4):injection[(clock>=30+j*25)&(clock<45+j*25),count-1]=.004*(j+1)
            elif protocol=='alternating':
                for j in range(6):injection[(clock>=25+j*20)&(clock<30+j*20),0 if j%2==0 else count-1]=.02
            elif protocol=='relaxation':
                injection[(clock>=20)&(clock<70),0]=.012
            else:raise ValueError(protocol)
            rows.append(dict(coupling=coupling,protocol=protocol,initial_mv=-65.))
            injections.append(injection)
    mult=np.empty((len(rows),count,11))
    for site in range(count):
        for channel in range(11):mult[:,site,channel]=(.75,1.,1.25)[(site+2*channel)%3]
    return rows,np.stack(injections,axis=1),mult

def permutation_limit(config,field):
    # Diagnostic tolerances reuse registered bounds in the corresponding units.
    if field.endswith('voltage'):return config['passive_gates']['max_disconnected_cross_effect_mv']
    if field.endswith('calcium'):return config['compartment_gates']['max_formula_floor_worst_cai_rmse_mM']
    if field.endswith('current'):return config['quality']['blocked_current_atol_ma_cm2']
    if field.endswith('gates'):return config['quality']['occupancy_atol']
    raise ValueError(field)

@contextmanager
def configured(root,permuted=False):
    """Process-local reuse of audited42 physics, restored even on failure."""
    original={name:getattr(x,name) for name in ('spec','design','matrix','rollout')}
    def order(geo):return np.random.default_rng(43000+len(geo['parents'])).permutation(len(geo['parents']))
    def ordered_design(geo,config):
        rows,inj,mult=design(geo,config)
        if permuted:
            p=order(geo);inj=inj[:,:,p];mult=mult[:,p]
        return rows,inj,mult
    def ordered_matrix(geo,coupling):
        area,lap=original['matrix'](geo,coupling)
        if permuted:
            p=order(geo);area=area[p];lap=lap[np.ix_(p,p)]
        return area,lap
    def ordered_rollout(*args,**kwargs):
        # External traces remain in physical-node order, including imposed paths.
        geo=args[2];forcing=args[8] if len(args)>8 else kwargs.get('forcing')
        if permuted and forcing is not None:
            forcing={k:v[:,:,:,order(geo)] for k,v in forcing.items()}
            if len(args)>8:args=(*args[:8],forcing,*args[9:])
            else:kwargs['forcing']=forcing
        result,info=original['rollout'](*args,**kwargs)
        if permuted:
            inverse=np.argsort(order(geo));result={k:v[:,:,:,inverse] for k,v in result.items()}
        info['node_order']=order(geo).tolist() if permuted else list(range(len(geo['parents'])))
        return result,info
    x.spec=spec;x.design=ordered_design;x.matrix=ordered_matrix;x.rollout=ordered_rollout
    try:yield
    finally:
        for name,value in original.items():setattr(x,name,value)

def native(root,teacher,out):
    print('[GIADA43] native references for new200ms transfer protocols',flush=True)
    with configured(root):x.native(root,teacher,out)

def evaluate(torch,root,source,out,device='cuda'):
    config=spec(root)[0];reports={};coverage={}
    for label,permuted in [('canonical',False),('permuted',True)]:
        print('[GIADA43] '+label+' paired GPU evaluation',flush=True)
        folder=out/label;folder.mkdir(exist_ok=True)
        for item in out.glob('native_*.npz'):shutil.copy2(item,folder/item.name)
        shutil.copy2(out/'native_floor.json',folder/'native_floor.json')
        with configured(root,permuted):reports[label]=x.evaluate(torch,root,source,folder,device)
        # Verification refuses an existing extraction directory. Preserve each
        # verified input outside the result tree, avoiding input duplication in
        # the final output archive while still re-verifying the second order.
        cache=out/'verified_source42';target=out.parent/('verified_source43_'+label)
        assert cache.is_dir() and not target.exists()
        cache.rename(target)
        reports[label].update(task='43',shared_physics_engine='42',task43_automatically_authorized=False)
        x.n.p.save_json(folder/'final_report.json',reports[label])
        results_path=folder/'results.json'
        if results_path.exists():
            for row in json.loads(results_path.read_text()):
                for node in row['per_node']:
                    for channel in node['channels']:
                        key='/'.join((label,row['mode'],channel['channel']))
                        counts=coverage.setdefault(key,dict(total=0,evaluable=0,not_evaluable=0,missing_event=0))
                        status=channel['event']['status']
                        counts['total']+=1;counts[status]=counts.get(status,0)+1
    parity=[]
    for geo in config['geometries']:
        filename='traces_'+geo['id']+'.npz'
        with np.load(out/'canonical'/filename) as a,np.load(out/'permuted'/filename) as b:
            for key in a.files:
                error=float(np.max(np.abs(a[key]-b[key])))
                # Reuse existing occupancy numerical tolerance, not a relaxed quality gate.
                limit=permutation_limit(config,key)
                parity.append(dict(geometry=geo['id'],field=key,max_abs_error=error,limit=limit,passed=error<=limit))
    report=dict(task='43',valid=True,complete=True,exploratory=True,models=reports,
        permutation_checks=parity,permutation_passed=all(r['passed'] for r in parity),
        all_parameters_finite=all(r['all_parameters_finite'] for r in reports.values()),
        all_trace_comparisons_finite=all(np.isfinite(r['max_abs_error']) for r in parity),
        event_support_by_order_mode_channel=coverage,
        candidate_weights_unchanged=all(r['candidate_weights_unchanged'] for r in reports.values()),
        training_updates=0,selection_performed=False,thresholds_changed=False,abstraction_changed=False,
        task41_original_gate_passed=False,speedup_claim=False,limits=config['limits'],
        decision='EXPLORATORY_FROZEN_TRANSFER_COMPLETE_REQUIRES_AUDIT')
    x.n.p.save_json(out/'final_report.json',report)
    return report
