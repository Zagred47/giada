"""New inputs/geometries, unchanged frozen ionic models and evaluation gates."""
import json
from contextlib import contextmanager
import numpy as np
from . import task46_spatial_cost as a

def spec(root):
    p=json.loads((root/'experiments/task47_preregistration_v1.json').read_text());old,cfg,ca,base=a.n.contract(root)
    assert p['task']=='47' and p['quality']==old['quality'] and p['compartment_gates']==cfg['gates']
    assert tuple(p['arms'])==a.ARMS and p['seeds']==[17,29,43] and p['dt_ms']==.1
    assert p['training_updates']==0 and not any(p[k] for k in ('selection_performed','thresholds_changed','abstraction_changed'))
    return p,cfg,ca,base

def design(geo,p):
    index=next(i for i,g in enumerate(p['geometries']) if g['id']==geo['id']);rng=np.random.default_rng(p['input_seeds'][index])
    count=len(geo['parents']);clock=np.arange(round(p['duration_ms']/.1))*.1
    starts=75.+np.cumsum(rng.uniform(10.,21.,8));widths=rng.uniform(1.5,3.5,8);amps=rng.uniform(.016,.022,8)
    rows=[];injections=[];multipliers=[]
    for coupling in p['coupling']:
        for site_label,site in (('proximal',0),('distal',count-1)):
            for form in ('irregular_burst','long_rebound'):
                for density in ('uniform','graded'):
                    for block in (('none','sodium','Ih') if form=='irregular_burst' and density=='graded' else ('none',)):
                        inj=np.zeros((len(clock),count))
                        def pulse(start,end,node,amplitude):inj[(clock>=start)&(clock<end),node]+=amplitude
                        for j in range(count):pulse(2+1.5*(j%5),6+1.5*(j%5),j,.0035*(1+j%3))
                        if form=='irregular_burst':
                            for start,width,amp in zip(starts,widths,amps):pulse(start,start+width,site,amp)
                            onset=float(starts[0])
                        else:
                            pulse(35,245,site,-.0034);pulse(337,353,site,.0165);onset=35.
                        mult=np.ones((count,11))
                        if density=='graded':mult=np.array([[(.7,1.,1.3)[(2*j+k)%3] for k in range(11)] for j in range(count)])
                        if block=='sodium':mult[:,[2,3,4]]=0.
                        elif block=='Ih':mult[:,5]=0.
                        rows.append(dict(coupling=coupling,protocol=f'47_{form}_{site_label}_{density}_{block}',initial_mv=-65.,form=form,stimulus_site=site,site_label=site_label,density=density,block=block,stimulus_onset_ms=onset,input_seed=p['input_seeds'][index]))
                        injections.append(inj);multipliers.append(mult)
    return rows,np.stack(injections,1),np.stack(multipliers)

@contextmanager
def configured():
    old=a.spec,a.design
    a.spec=spec;a.design=design
    try:yield
    finally:a.spec,a.design=old

def finalize(out):
    path=out/'final_report.json';r=json.loads(path.read_text())
    r.update(task='47',task47_authorized=True,task48_authorized=False)
    if r.get('complete'):
        pools=[m for horizon in r['summary'].values() for m in horizon['P'].values()]
        r['independent_quality_confirmation_passed']=bool(pools) and all(q['eligible']==q['total']==q['passed'] and q['total']>0 for q in pools) and all(r[k] for k in ('passive_native_floor_passed','active_native_floor_passed','endpoint_parity_passed','null_substitution_control_passed','candidate_weights_unchanged','all_parameters_finite','all_traces_finite'))
        r['decision']='INDEPENDENT_QUALITY_PASS_COST_REQUIRES_AUDIT' if r['independent_quality_confirmation_passed'] else 'INDEPENDENT_QUALITY_NO_GO_REQUIRES_AUDIT'
    a.n.p.save_json(path,r)
