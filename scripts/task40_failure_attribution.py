"""Frozen post-hoc diagnostic controls, NOT new candidates or selection."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task40_causal_reinsertion as t
OUT=ROOT/'experiments/results/task40_audited_v1'
def main():
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    spec,cfg,ca,base=t.contract(ROOT);data,meta=t.design(ROOT,spec);q=spec['current_quality']
    models=t.load_models(torch,ROOT,'cpu');model=models['J1']
    with np.load(OUT/'reference_traces.npz') as z:ref={k:z['autonomous__'+k].copy() for k in ('voltage','calcium','gates','current')}
    with np.load(OUT/'primary_traces.npz') as z:gpu={k:z['J1__all__autonomous__'+k].copy() for k in ref}
    results={};curves={}
    masks={'all':tuple(range(11)), 'calcium_only':(0,1), 'all_except_calcium':tuple(range(2,11)),
           'all_except_Ca_LVAst':(0,*range(2,11)), 'all_except_Ca_HVA':tuple(range(1,11))}
    with torch.inference_mode():
        for name,channels in masks.items():
            core=t.make_core(torch,cfg,ca,base,data['area_um2'],model,channels)
            pred=t.rollout(torch,core,data,'cpu');rows=[]
            for s,seed in enumerate(spec['seeds']):
                for c in range(32,len(meta)):
                    o=meta[c]['source_index']
                    metric=t.p.evaluate_intervention(ref['current'][:,o],ref['current'][:,c],pred['current'][:,s,o],pred['current'][:,s,c],.1,q)
                    rows.append(dict(case=c,seed=seed,metrics=metric))
            results[name]=rows
            curves[name]=pred
            print(name,'failures',sum(not r['metrics']['passed'] for r in rows),flush=True)
    # Saved baseline GPU vs CPU numerical parity, reported rather than assumed identical.
    parity={k:float(np.max(np.abs(curves['all'][k]-gpu[k]))) for k in ref}
    original=json.loads((OUT/'causal_results.json').read_text())['J1/all/autonomous']['interventions']
    flag_parity=all(a['metrics']['passed']==b['metrics']['passed'] for a,b in zip(results['all'],original))
    # Replace gates in readout only, without rerunning or repairing dynamics: local error decomposition.
    readout={}
    for variant,gateids,force_v in [('reference_V',(),True),('Ca_m_reference',(0,2),False),('Ca_h_reference',(1,3),False),('Ca_mh_reference',(0,1,2,3),False),('Ca_mh_and_V_reference',(0,1,2,3),True)]:
        pp=gpu['gates'][1:].copy()
        for k in gateids:pp[...,k]=ref['gates'][1:,:,k][:,None]
        v=ref['voltage'][:-1,None] if force_v else gpu['voltage'][:-1]
        _,current=t.p.readout(v,pp,data['multipliers'][None,None],t.ionic.REVERSALS)
        rows=[]
        for s,seed in enumerate(spec['seeds']):
            for c in range(32,len(meta)):
                o=meta[c]['source_index']
                mm=t.p.evaluate_intervention(ref['current'][:,o],ref['current'][:,c],current[:,s,o],current[:,s,c],.1,q)
                rows.append(dict(case=c,seed=seed,metrics=mm))
        readout[variant]=rows
    # Coefficients along reference paths and close-pair sensitivity of u,z.
    v=ref['voltage'][:-1].reshape(-1);c=ref['calcium'][:-1].reshape(-1)
    with torch.inference_mode():
        x=torch.tensor(v)[None].expand(3,-1);y=torch.tensor(c)[None].expand(3,-1)
        uu,zz=model.raw(x,y);u=torch.sigmoid(uu).numpy().reshape(3,400,88,18);z=torch.sigmoid(zz).numpy().reshape(3,400,88,18)
    tu,tz=t.p.coefficients(v.astype('float64'),c.astype('float64'),.1);tu=tu.reshape(400,88,18);tz=tz.reshape(400,88,18)
    coefficient_controls={}
    for label,oracle_u,oracle_z in [('learned',(),()),('oracle_u',(0,1,2,3),()),('oracle_z',(),(0,1,2,3)),('oracle_m_z',(),(0,2)),('oracle_u_z',(0,1,2,3),(0,1,2,3))]:
        pu=u[...,:4].astype('float64').copy();pz=z[...,:4].astype('float64').copy()
        for k in oracle_u:pu[:,:,:,k]=tu[None,:,:,k]
        for k in oracle_z:pz[:,:,:,k]=tz[None,:,:,k]
        state=np.broadcast_to(data['gates'][None,:,0,:4],(3,88,4)).copy();states=[]
        for tick in range(400):
            state=state+pz[:,tick]*(pu[:,tick]-state);states.append(state.copy())
        gates=np.broadcast_to(ref['gates'][1:,None],(400,3,88,18)).copy();gates[...,:4]=np.stack(states)
        _,current=t.p.readout(ref['voltage'][:-1,None],gates,data['multipliers'][None,None],t.ionic.REVERSALS)
        rows=[]
        for s,seed in enumerate(spec['seeds']):
            for case in range(32,len(meta)):
                origin=meta[case]['source_index']
                mm=t.p.evaluate_intervention(ref['current'][:,origin],ref['current'][:,case],current[:,s,origin],current[:,s,case],.1,q)
                rows.append(dict(case=case,seed=seed,metrics=mm))
        coefficient_controls[label]=rows
    coefficients=[]
    for s,seed in enumerate(spec['seeds']):
        for k in range(4):
            coefficients.append(dict(seed=seed,gate=k,range_V=[float(v.min()),float(v.max())],
                u_rmse=float(np.sqrt(np.mean((u[s,:,:,k]-tu[:,:,k])**2))),
                z_log_rmse=float(np.sqrt(np.mean((np.log(z[s,:,:,k])-np.log(tz[:,:,k]))**2))),
                tau_ratio_min=float(np.min(np.log1p(-tz[:,:,k])/np.log1p(-z[s,:,:,k]))),
                tau_ratio_max=float(np.max(np.log1p(-tz[:,:,k])/np.log1p(-z[s,:,:,k])))))
    report=dict(valid=flag_parity,post_hoc=True,no_candidate_promotion=True,thresholds_changed=False,
        controls_are_diagnostic_not_allowed_deployment=True,cpu_gpu_max_error=parity,cpu_gpu_168_flags_equal=flag_parity,
        autonomous_failures={k:sum(not r['metrics']['passed'] for r in v) for k,v in results.items()},
        readout_only_failures={k:sum(not r['metrics']['passed'] for r in v) for k,v in readout.items()},
        coefficient_controls_failures={k:sum(not r['metrics']['passed'] for r in v) for k,v in coefficient_controls.items()},coefficient_probe=coefficients,
        limits='Readout-only replacements are counterfactual error probes, not dynamical repairs. CPU replay is same opened data, not independent confirmation.')
    for name,obj in [('attribution_controls.json',results),('readout_controls.json',readout),('coefficient_controls.json',coefficient_controls),('attribution_audit.json',report)]:
        (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
