"""Conditional45 coverage audit: frozen nominee first; never automatic refit."""
import json,os
from contextlib import contextmanager
import numpy as np
from . import task44_composition_localization as y
x,n,t=y.x,y.n,y.t
ARMS=('none','all')

def spec(root):
    p=json.loads((root/'experiments/task45_preregistration_v1.json').read_text())
    old,cfg,ca,base=n.contract(root)
    assert p['quality']==old['quality'] and p['compartment_gates']==cfg['gates']
    assert p['passive_gates']==x.passive.config(root,'iv_ad1_passive_axial_preregistration_v3.json')['gates']
    assert p['arms']==list(ARMS) and p['seeds']==[17,29,43] and p['dt_ms']==.1
    assert not p['thresholds_changed'] and not p['abstraction_changed'] and p['training_updates']==0 and not p['selection_performed']
    return p,cfg,ca,base

def design(geo,p):
    count=len(geo['parents']);clock=np.arange(round(p['duration_ms']/.1))*.1
    rows=[];injections=[]
    for coupling in p['coupling']:
        for protocol in p['protocols']:
            inj=np.zeros((len(clock),count))
            def pulse(start,end,site,amplitude):inj[(clock>=start)&(clock<end),site]+=amplitude
            for site in range(count):pulse(1+2*(site%4),5+2*(site%4),site,.004*(1+site%3))
            if protocol=='45_hyper_rebound':
                pulse(40,200,0,-.008);pulse(300,315,count-1,.015)
            elif protocol=='45_burst_recovery':
                for j in range(8):pulse(40+j*25,43+j*25,0,.020)
                for j in range(10):pulse(350+j*10,352+j*10,count-1,.020)
            elif protocol=='45_slow_staircase':
                for j in range(4):pulse(50+j*100,110+j*100,0 if j%2==0 else count-1,.004*(j+1))
                pulse(430,600,0,.003)
            else:raise ValueError(protocol)
            rows.append(dict(coupling=coupling,protocol=protocol,initial_mv=-65.));injections.append(inj)
    mult=np.empty((len(rows),count,11))
    for site in range(count):
        for channel in range(11):mult[:,site,channel]=(.75,1.,1.25)[(site+2*channel)%3]
    return rows,np.stack(injections,axis=1),mult

@contextmanager
def configured():
    before=y.spec,y.ARMS,y.previous.design
    y.spec=spec;y.ARMS=ARMS;y.previous.design=design
    try:yield
    finally:y.spec,y.ARMS,y.previous.design=before

def native(root,teacher,out):
    print('[GIADA45] native references for prospective coverage panel',flush=True)
    with configured():y.native(root,teacher,out)

def crop(data,ticks):
    return {k:v[:ticks+(0 if k=='current' else 1)] for k,v in data.items()}

def coverage_probe(ref,geo):
    return dict(geometry=geo['id'],voltage_range_mv=[float(ref['voltage'].min()),float(ref['voltage'].max())],
        calcium_range_mM=[float(ref['calcium'].min()),float(ref['calcium'].max())],
        gate_min=ref['gates'].min(axis=(0,1,2,3,4)).tolist(),gate_max=ref['gates'].max(axis=(0,1,2,3,4)).tolist())

def decision(rows,prefixes,floor_valid):
    candidate=[r for r in rows+prefixes if r['arm']=='all']
    controls=[r for r in rows+prefixes if r['arm']=='none']
    control_valid=bool(controls and all(r['finite'] and r['current_passed'] and r['compartment_passed'] for r in controls))
    eligible=bool(floor_valid and control_valid and candidate and all(r['judgment_eligible'] for r in candidate))
    failed=sum(not(r['finite'] and r['current_passed'] and r['compartment_passed']) for r in candidate)
    return dict(reference_judgment_valid=eligible,reference_regime_control_valid=control_valid,frozen_coverage_passed=bool(eligible and not failed),
        candidate_failed_checks=failed,refit_recommended=bool(eligible and failed),refit_performed=False,
        training_mini_scaling_performed=False,automatic_refit_authorized=False,
        decision='REFERENCE_BLOCKED_REQUIRES_AUDIT' if not eligible else ('FROZEN_COVERAGE_PASSED_REFIT_NOT_INDICATED_IN_PANEL' if not failed else 'FROZEN_COVERAGE_GAP_REQUIRES_REFIT_DECISION'))

def evaluate(torch,root,source,out,device='cuda'):
    p,cfg,ca,base=spec(root)
    cache=out.parent/'verified_source45';x.repair.p.r.verify_source(root,source,cache)
    os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
    net=n.load_models(torch,root,device)[0]['dual']
    with np.load(root/p['freeze']['heads_path']) as z:net=x.repair.materialize(torch,net,dict(z))
    assert n.weights_hash(net)==p['freeze']['logical_weights_sha256']
    torch.save(net.state_dict(),out/'frozen41r.pt')
    floor=json.loads((out/'native_floor.json').read_text());rows=[];prefixes=[];execution={};parity=[];support={};probes=[]
    with torch.inference_mode(),configured():
        for geo in p['geometries']:
            parity.extend([dict(geometry=geo['id'],**r) for r in y.endpoint_preflight(torch,root,net,geo,p,cfg,ca,base,device)])
            n.p.save_json(out/'endpoint_preflight.json',parity)
            with np.load(out/('native_'+geo['id']+'.npz')) as z:teacher=dict(z)
            repeated,info=y.rollout(torch,None,geo,p,cfg,ca,base,device,ARMS)
            ref={k:v[:,:,:1].copy() for k,v in repeated.items()};del repeated
            execution[geo['id']+'/formula']=info;probes.append(coverage_probe(ref,geo))
            physical=design(geo,p)[0]
            for mode,forcing in [('imposed',ref),('autonomous',None)]:
                pred,info=y.rollout(torch,net,geo,p,cfg,ca,base,device,ARMS,forcing)
                controls=[]
                for key in ref:
                    error=float(abs(pred[key][:,:,0]-ref[key][:,:,0]).max());limit=y.previous.permutation_limit(p,key)
                    controls.append(dict(field=key,max_abs_error=error,limit=limit,passed=error<=limit))
                assert all(r['passed'] for r in controls),'Task45 zero-substitution execution control failed'
                info['null_control']=controls;execution[geo['id']+'/'+mode]=info
                measured=y.score(pred,ref,teacher,geo,physical,ARMS,mode,p,cfg,floor['valid'])
                for r in measured:r['horizon_ms']=p['duration_ms']
                rows.extend(measured)
                for horizon in p['prefix_horizons_ms']:
                    if horizon>=p['duration_ms']:continue
                    ticks=round(horizon/.1);tt={k:v[:ticks+1] for k,v in teacher.items()}
                    short=y.score(crop(pred,ticks),crop(ref,ticks),tt,geo,physical,ARMS,mode,p,cfg,floor['valid'])
                    for r in short:r['horizon_ms']=horizon
                    prefixes.extend(short)
                for r in measured:
                    for node in r['per_node']:
                        for channel in node['channels']:
                            key='/'.join((r['arm'],mode,channel['channel']));counts=support.setdefault(key,dict(total=0,evaluable=0,not_evaluable=0,missing_event=0));status=channel['event']['status']
                            counts['total']+=1;counts[status]=counts.get(status,0)+1
                np.savez_compressed(out/('traces_'+geo['id']+'_'+mode+'.npz'),**pred)
                n.p.save_json(out/'results.json',rows);n.p.save_json(out/'prefix_results.json',prefixes);n.p.save_json(out/'execution.json',execution);n.p.save_json(out/'coverage_probes.json',probes)
                print('[GIADA45] '+geo['id']+'/'+mode+' full horizon and prefixes complete',flush=True)
                del pred
            np.savez_compressed(out/('reference_'+geo['id']+'.npz'),**ref)
    assert n.weights_hash(net)==p['freeze']['logical_weights_sha256']
    paired=y.paired_contrasts(rows);coupling=y.coupling_contrasts(rows)
    n.p.save_json(out/'paired_contrasts.json',paired);n.p.save_json(out/'coupling_contrasts.json',coupling)
    coverage=decision(rows,prefixes,floor['valid'])
    report=dict(task='45',valid=True,complete=True,exploratory=True,training_updates=0,selection_performed=False,refit_performed=False,
        thresholds_changed=False,abstraction_changed=False,candidate_weights_unchanged=True,
        all_parameters_finite=all(bool(torch.isfinite(v).all()) for v in net.parameters()),all_traces_finite=all(r['finite'] for r in rows),
        passive_native_floor_passed=floor['valid'],active_native_floor_passed=all(r['native_floor_passed'] for r in rows+prefixes),
        endpoint_execution_parity_passed=all(r['passed'] for r in parity),null_substitution_control_passed=True,
        arms={arm:{mode:dict(total=sum(r['arm']==arm and r['mode']==mode for r in rows),joint_passed=sum(r['arm']==arm and r['mode']==mode and r['judgment_eligible'] and r['current_passed'] and r['compartment_passed'] for r in rows)) for mode in ('imposed','autonomous')} for arm in ARMS},
        event_support=support,new_autonomous_failures=sum(r['new_failure_in_autonomous'] for r in paired),new_coupled_failures=sum(r['new_failure_when_coupled'] for r in coupling),
        coverage_decision=coverage,decision=coverage['decision'],
        qualitative_propagation_audit={mode:dict(site_seed_cases=sum(len(r['propagation']) for r in rows if r['arm']=='all' and r['mode']==mode),
            crossing_count_mismatches=sum(v['crossing_count_mismatch'] for r in rows if r['arm']=='all' and r['mode']==mode for v in r['propagation']),
            diagnostic_only=True,requires_scoped_review=True) for mode in ('imposed','autonomous')},
        horizon_summary={str(h):{mode:dict(total=sum(r['arm']=='all' and r['mode']==mode and r['horizon_ms']==h for r in rows+prefixes),
            eligible_joint_passed=sum(r['arm']=='all' and r['mode']==mode and r['horizon_ms']==h and r['judgment_eligible'] and r['current_passed'] and r['compartment_passed'] for r in rows+prefixes)) for mode in ('imposed','autonomous')} for h in p['prefix_horizons_ms']},
        protocols_prospectively_registered=True,independent_task47_confirmation=False,native_references_regenerated=True,
        task41_original_gate_passed=False,speedup_claim=False,limits=p['limits'])
    n.p.save_json(out/'final_report.json',report);return report
