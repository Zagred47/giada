"""Paired frozen spatial/channel substitutions; classical physics unchanged."""
import json,math,os,time
from contextlib import contextmanager
import numpy as np
from . import task43_frozen_transfer as previous
x=previous.x;n=x.n;t=x.t
ARMS=('none','site_proximal','site_distal','calcium','sodium','potassium','ih','all')
GROUPS=dict(calcium=(0,1),sodium=(2,3,4),potassium=(6,7,8,9,10),ih=(5,))

def spec(root):
    p=json.loads((root/'experiments/task44_preregistration_v1.json').read_text())
    old,cfg,ca,base=n.contract(root)
    assert p['quality']==old['quality'] and p['compartment_gates']==cfg['gates']
    assert p['passive_gates']==x.passive.config(root,'iv_ad1_passive_axial_preregistration_v3.json')['gates']
    assert p['arms']==list(ARMS) and p['seeds']==[17,29,43] and p['dt_ms']==.1
    assert not p['thresholds_changed'] and not p['abstraction_changed'] and p['training_updates']==0
    return p,cfg,ca,base

@contextmanager
def native_context(root):
    before=x.spec,x.design
    x.spec=spec;x.design=previous.design
    try:yield
    finally:x.spec,x.design=before

def native(root,teacher,out):
    print('[GIADA44] native paired reference generation',flush=True)
    with native_context(root):x.native(root,teacher,out)

def channel_masks(count,arms=ARMS):
    mask=np.zeros((len(arms),count,11),dtype=bool)
    for i,arm in enumerate(arms):
        if arm=='all':mask[i]=True
        elif arm=='site_proximal':mask[i,0]=True
        elif arm=='site_distal':mask[i,-1]=True
        elif arm in GROUPS:mask[i,:,GROUPS[arm]]=True
        elif arm!='none':raise ValueError(arm)
    return mask

def local_core(torch,net,mask,cfg,calcium):
    """Select non-SK gates before Ca integration; SK uses resulting hybrid Ca.

    Unselected channels are a causal diagnostic control, not a reduced-abstraction
    replacement candidate. No fitted or corrected online coefficients.
    """
    gate_indices=[k for k in range(11) for _ in t.ionic.STATE_NAMES[k]]
    gate_mask=mask[...,gate_indices]
    persistence=math.exp(-cfg['dt_ms']/calcium['decay_ms'])
    def core(v,ca,state,mult):
        v=v.float();ca=ca.float();state=state.float();mult=mult.float()
        neural=net(v,ca,state) if net is not None else None
        parts=[]
        for k,name in enumerate(t.ionic.CHANNELS):
            a,b=map(int,t.ionic.OFFSETS[k:k+2])
            if k==9:value=state[...,a:b]
            else:
                u,tau=t._rates(name,v,ca,torch)
                if b-a==1:u,tau=u[...,None],tau[...,None]
                z=-torch.expm1(-cfg['dt_ms']/tau)
                value=(1-z)*state[...,a:b]+z*u
                if neural is not None:value=torch.where(gate_mask[...,a:b],neural[...,a:b],value)
            parts.append(value)
        g=torch.cat(parts,-1);_,current=t.readout(torch,v,g,mult)
        source=-10000*current[...,:2].sum(-1)*calcium['gamma']/(2*96485.33212*calcium['depth_um'])
        nc=1e-4+(ca-1e-4)*persistence+source*calcium['decay_ms']*(1-persistence)
        inf=1/(1+(.00043/nc.clamp_min(1e-7)).pow(4.8))
        sk=inf+(state[...,16]-inf)*math.exp(-cfg['dt_ms'])
        if neural is not None:sk=torch.where(mask[...,9],t.learned_sk(torch,net,nc,state[...,16]),sk)
        g=torch.cat((g[...,:16],sk[...,None],g[...,17:]),-1)
        conductance,current=t.readout(torch,v,g,mult)
        return nc,g,conductance,current
    return core

def setup(torch,net,geo,config,cfg,ca,base,device,arms):
    rows,inj,mult=previous.design(geo,config);cases,count=mult.shape[:2];ac=len(arms)
    area,unused=x.matrix(geo,0);lap=np.stack([x.matrix(geo,r['coupling'])[1] for r in rows])
    dtype=torch.float64
    factor=torch.tensor(area*.01,device=device,dtype=dtype)
    lap=torch.tensor(lap,device=device,dtype=dtype)[None]
    eye=torch.eye(count,device=device,dtype=dtype)
    cmask=channel_masks(count,arms)[:,None].repeat(cases,1).reshape(ac*cases*count,11)
    cmask=torch.tensor(cmask,device=device)[None].expand(3,-1,-1)
    ionic=local_core(torch,net,cmask,cfg,ca)
    cap=.001*base['cm_uf_cm2']/.1
    rev=torch.tensor(t.ionic.REVERSALS,device=device,dtype=dtype)
    def core(v,c,state,multipliers,injection,syn):
        nc,g,conductance,current=ionic(v,c,state,multipliers)
        conductance=conductance.double()
        diagonal=(cap+base['g_pas_s_cm2']+conductance.sum(-1)).reshape(3,ac,cases,count)*factor
        rhs=(cap*v+base['g_pas_s_cm2']*base['e_pas_mv']+(conductance*rev).sum(-1)+injection).reshape(3,ac,cases,count)*factor
        mat=lap[None]+eye*diagonal[...,None]
        nv=torch.linalg.solve(mat,rhs[...,None])[...,0].reshape(3,-1)
        return nv,nc,g,current
    voltage=np.tile(np.repeat([r['initial_mv'] for r in rows],count),ac).astype('float64')
    calcium=np.full_like(voltage,1e-4,dtype='float32')
    initial=t.t30.initial_states(voltage,calcium).astype('float32')
    expand=lambda a,dt:torch.tensor(a,device=device,dtype=dt)[None].expand(3,*a.shape).contiguous()
    mult=np.tile(mult[None],(ac,1,1,1)).reshape(-1,11)
    args=[expand(voltage,dtype),expand(calcium,torch.float32),expand(initial,torch.float32),expand(mult,torch.float32),expand(np.zeros(len(voltage)),dtype),expand(np.zeros((len(voltage),4)),dtype)]
    injections=torch.tensor(inj,device=device,dtype=dtype)[:,None].expand(-1,ac,-1,-1).reshape(len(inj),-1)
    return core,args,injections,dict(arms=list(arms),cases=cases,nodes=count,graph_used=False,execution='eager resident arm/case/seed vectorization; no discarded local membrane solve')

def endpoint_preflight(torch,root,net,geo,config,cfg,ca,base,device):
    """No optimization accepted without original-core parity at nontrivial state."""
    result=[]
    for copies in (1,len(ARMS)):
        for model,arm in ((None,'none'),(net,'all')):
            core,args,inj,info=setup(torch,model,geo,config,cfg,ca,base,device,(arm,)*copies)
            # Explicit device index keeps the audited runner in its eager branch;
            # same CUDA device/core, without known-unsupported graph capture.
            legacy_device='cuda:0' if device=='cuda' else device
            def repeated_design(g,p):
                rows,injection,multipliers=previous.design(g,p)
                return rows*copies,np.tile(injection,(1,copies,1)),np.tile(multipliers,(copies,1,1))
            with native_context(root):
                x.design=repeated_design
                legacy,old,_,legacy_core,_=x.runner(torch,model,geo,config,cfg,ca,base,legacy_device)
            # Deliberately distinct V/Ca/gates; test real arm-batch width as well.
            args[0].copy_(torch.linspace(-75.,20.,args[0].numel(),device=device,dtype=torch.float64).reshape_as(args[0]))
            args[1].copy_(torch.linspace(.0001,.001,args[1].numel(),device=device).reshape_as(args[1]))
            args[2].copy_(torch.linspace(.02,.98,args[2].numel(),device=device).reshape_as(args[2]))
            a=core(*args);b=legacy_core(*args)
            equal=[torch.equal(v,w) for v,w in zip(a,b)]
            result.append(dict(arm=arm,batch_copies=copies,bitwise_fields=equal,max_abs=[float((v-w).abs().max()) for v,w in zip(a,b)],passed=all(equal)))
    assert all(r['passed'] for r in result),'Task44 endpoint execution parity failed'
    return result

def rollout(torch,net,geo,config,cfg,ca,base,device,arms,forcing=None):
    core,args,inj,info=setup(torch,net,geo,config,cfg,ca,base,device,arms)
    run=t.Stepper(torch,core,args,False);values=[[a.clone()] for a in args[:3]]+[[]]
    start=time.perf_counter();ac=len(arms);cases=info['cases'];nodes=info['nodes']
    forcing_tensors=None
    if forcing is not None:
        forcing_tensors={k:torch.tensor(forcing[k][:,:,0],device=device,dtype=args[j].dtype)[:,:,None].expand(-1,-1,ac,-1,-1).reshape(len(forcing[k]),3,-1) for j,k in enumerate(('voltage','calcium'))}
    for tick in range(len(inj)):
        if forcing_tensors is not None:
            for j,k in enumerate(('voltage','calcium')):run.args[j].copy_(forcing_tensors[k][tick])
        out=run.tick(inj[tick][None].expand(3,-1),args[5])
        for dst,value in zip(values,out):dst.append(value.clone())
    if device=='cuda':torch.cuda.synchronize()
    info['wall_seconds']=time.perf_counter()-start
    result={k:torch.stack(v).cpu().numpy().reshape(len(v),3,ac,cases,nodes,*(() if k in ('voltage','calcium') else (18 if k=='gates' else 11,))) for k,v in zip(('voltage','calcium','gates','current'),values)}
    return result,info

def score(pred,ref,native_data,geo,rows,arms,mode,p,cfg,native_valid):
    result=[]
    for case,row in enumerate(rows):
        ve=ref['voltage'][:,0,0,case]-native_data['voltage'][:,case]
        ce=ref['calcium'][:,0,0,case]-native_data['calcium'][:,case]
        vr=np.sqrt((ve**2).mean(0));cr=np.sqrt((ce**2).mean(0));gates=cfg['gates']
        floor=bool(np.sqrt((ve**2).mean())<=gates['max_formula_floor_pooled_voltage_rmse_mv'] and vr.max()<=gates['max_formula_floor_worst_voltage_rmse_mv'] and cr.max()<=gates['max_formula_floor_worst_cai_rmse_mM'])
        for ai,arm in enumerate(arms):
            for si,seed in enumerate(p['seeds']):
                v=pred['voltage'][:,si,ai,case];c=pred['calcium'][:,si,ai,case]
                dv=v-ref['voltage'][:,0,0,case];dc=c-ref['calcium'][:,0,0,case]
                metrics=[];nodes=[]
                for j in range(v.shape[-1]):
                    metrics.append(n.p.evaluate_currents(ref['current'][:,0,0,case,j],pred['current'][:,si,ai,case,j],.1,p['quality'],ref['gates'][1:,0,0,case,j],pred['gates'][1:,si,ai,case,j]))
                    rv=ref['voltage'][:,0,0,case,j]
                    crossing=lambda a:np.flatnonzero((a[:-1]<0)&(a[1:]>=0))
                    rc,pc=crossing(rv),crossing(v[:,j])
                    nodes.append(dict(node=j,max_abs_voltage_error_mv=float(abs(dv[:,j]).max()),voltage_rmse_mv=float(np.sqrt((dv[:,j]**2).mean())),
                        reference_peak_mv=float(rv.max()),candidate_peak_mv=float(v[:,j].max()),reference_peak_ms=float(np.argmax(rv)*.1),candidate_peak_ms=float(np.argmax(v[:,j])*.1),
                        reference_upcrossings_ms=((rc+1)*.1).tolist(),candidate_upcrossings_ms=((pc+1)*.1).tolist(),
                        crossing_count_mismatch=bool(len(rc)!=len(pc))))
                finite=all(np.isfinite(a[:,si,ai,case]).all() for a in pred.values())
                comp=bool(finite and np.sqrt((dv**2).mean())<=gates['max_candidate_pooled_voltage_rmse_mv'] and np.sqrt((dv**2).mean(0)).max()<=gates['max_candidate_worst_voltage_rmse_mv'] and np.sqrt((dc**2).mean(0)).max()<=gates['max_candidate_worst_cai_rmse_mM'] and not np.any((v<-140)|(v>100)))
                result.append(dict(geometry=geo['id'],case=case,**row,arm=arm,mode=mode,seed=seed,finite=bool(finite),
                    native_floor_passed=floor,judgment_eligible=bool(floor and native_valid),native_voltage_rmse_mv=vr.tolist(),native_calcium_rmse_mM=cr.tolist(),
                    current_passed=all(m['passed'] for m in metrics),compartment_passed=comp,per_node=metrics,propagation=nodes,
                    voltage_rmse_mv=float(np.sqrt((dv**2).mean())),max_abs_voltage_error_mv=float(abs(dv).max()),calcium_rmse_mM=float(np.sqrt((dc**2).mean()))))
    return result

def paired_contrasts(rows):
    index={(r['geometry'],r['case'],r['seed'],r['arm'],r['mode']):r for r in rows}
    result=[]
    for key,a in index.items():
        if key[-1]!='imposed':continue
        b=index[(*key[:-1],'autonomous')]
        result.append(dict(geometry=key[0],case=key[1],seed=key[2],arm=key[3],coupling=a['coupling'],protocol=a['protocol'],
            imposed_voltage_rmse_mv=a['voltage_rmse_mv'],autonomous_voltage_rmse_mv=b['voltage_rmse_mv'],
            autonomous_minus_imposed_voltage_rmse_mv=b['voltage_rmse_mv']-a['voltage_rmse_mv'],
            new_failure_in_autonomous=bool(a['current_passed'] and a['compartment_passed'] and not(b['current_passed'] and b['compartment_passed']))))
    return result

def coupling_contrasts(rows):
    index={(r['geometry'],r['protocol'],r['seed'],r['arm'],r['mode'],r['coupling']):r for r in rows}
    result=[]
    for key,row in index.items():
        if key[-1]==0:continue
        disconnected=index[(*key[:-1],0.)]
        result.append(dict(geometry=key[0],protocol=key[1],seed=key[2],arm=key[3],mode=key[4],coupling=key[5],
            disconnected_voltage_rmse_mv=disconnected['voltage_rmse_mv'],coupled_voltage_rmse_mv=row['voltage_rmse_mv'],
            coupled_minus_disconnected_voltage_rmse_mv=row['voltage_rmse_mv']-disconnected['voltage_rmse_mv'],
            new_failure_when_coupled=bool(disconnected['current_passed'] and disconnected['compartment_passed'] and not(row['current_passed'] and row['compartment_passed']))))
    return result

def family_interactions(pred,ref,geo,mode,config):
    """Descriptive nonadditivity; RMS errors themselves are not additive."""
    rows=[];group_indices=[ARMS.index(k) for k in GROUPS]
    for key in pred:
        delta=pred[key][:,:,ARMS.index('all')].astype('float64')-ref[key][:,:,0]
        sum_groups=sum(pred[key][:,:,i].astype('float64')-ref[key][:,:,0] for i in group_indices)
        residual=delta-sum_groups
        for si,seed in enumerate(config['seeds']):
            for case in range(delta.shape[2]):
                rows.append(dict(geometry=geo['id'],mode=mode,seed=seed,case=case,field=key,
                    all_substitution_delta_rmse=float(np.sqrt((delta[:,si,case]**2).mean())),
                    nonadditive_residual_rmse=float(np.sqrt((residual[:,si,case]**2).mean())),
                    nonadditive_residual_max_abs=float(abs(residual[:,si,case]).max()),diagnostic_only=True))
    return rows

def evaluate(torch,root,source,out,device='cuda'):
    p,cfg,ca,base=spec(root)
    cache=out.parent/'verified_source44';x.repair.p.r.verify_source(root,source,cache)
    os.environ['GIADA_TASK41N_FROZEN']=str(cache/'frozen_source.zip.bin')
    net=n.load_models(torch,root,device)[0]['dual']
    with np.load(root/p['freeze']['heads_path']) as z:net=x.repair.materialize(torch,net,dict(z))
    assert n.weights_hash(net)==p['freeze']['logical_weights_sha256']
    torch.save(net.state_dict(),out/'frozen41r.pt')
    floor=json.loads((out/'native_floor.json').read_text());summaries=[];parity=[];coverage={};execution={};interactions=[]
    with torch.inference_mode():
        for geo in p['geometries']:
            parity.extend([dict(geometry=geo['id'],**r) for r in endpoint_preflight(torch,root,net,geo,p,cfg,ca,base,device)])
            n.p.save_json(out/'endpoint_preflight.json',parity)
            with np.load(out/('native_'+geo['id']+'.npz')) as z:teacher=dict(z)
            # Match the candidate's complete arm-batch shape: changes in batching
            # must not masquerade as a causal substitution effect.
            repeated_ref,info=rollout(torch,None,geo,p,cfg,ca,base,device,ARMS)
            ref={k:v[:,:,:1].copy() for k,v in repeated_ref.items()};del repeated_ref
            execution[geo['id']+'/formula']=info
            physical=previous.design(geo,p)[0]
            for mode,forcing in [('imposed',ref),('autonomous',None)]:
                pred,info=rollout(torch,net,geo,p,cfg,ca,base,device,ARMS,forcing)
                execution[geo['id']+'/'+mode]=info
                # The zero-substitution arm is a mandatory execution control.
                null=[]
                for key in ref:
                    error=float(abs(pred[key][:,:,0]-ref[key][:,:,0]).max())
                    limit=previous.permutation_limit(p,key)
                    null.append(dict(field=key,max_abs_error=error,limit=limit,passed=error<=limit))
                assert all(r['passed'] for r in null),'Task44 zero-substitution control failed'
                execution[geo['id']+'/'+mode]['null_control']=null
                measured=score(pred,ref,teacher,geo,physical,ARMS,mode,p,cfg,floor['valid']);summaries.extend(measured)
                interactions.extend(family_interactions(pred,ref,geo,mode,p))
                for row in measured:
                    for node in row['per_node']:
                        for channel in node['channels']:
                            key='/'.join((row['arm'],mode,channel['channel']));counts=coverage.setdefault(key,dict(total=0,evaluable=0,not_evaluable=0,missing_event=0))
                            status=channel['event']['status'];counts['total']+=1;counts[status]=counts.get(status,0)+1
                np.savez_compressed(out/('traces_'+geo['id']+'_'+mode+'.npz'),**pred)
                n.p.save_json(out/'results.json',summaries);n.p.save_json(out/'execution.json',execution);n.p.save_json(out/'family_interactions.json',interactions)
                print('[GIADA44] '+geo['id']+'/'+mode+' all8 paired arms complete',flush=True)
                del pred
            np.savez_compressed(out/('reference_'+geo['id']+'.npz'),**ref)
    assert n.weights_hash(net)==p['freeze']['logical_weights_sha256']
    contrasts=paired_contrasts(summaries);n.p.save_json(out/'paired_contrasts.json',contrasts)
    coupling=coupling_contrasts(summaries);n.p.save_json(out/'coupling_contrasts.json',coupling)
    report=dict(task='44',valid=True,complete=True,exploratory=True,training_updates=0,selection_performed=False,
        thresholds_changed=False,abstraction_changed=False,candidate_weights_unchanged=True,
        all_parameters_finite=all(bool(torch.isfinite(v).all()) for v in net.parameters()),all_traces_finite=all(r['finite'] for r in summaries),
        passive_native_floor_passed=floor['valid'],active_native_floor_passed=all(r['native_floor_passed'] for r in summaries),
        endpoint_execution_parity_passed=all(r['passed'] for r in parity),
        null_substitution_control_passed=True,
        arms={arm:{mode:dict(total=sum(r['arm']==arm and r['mode']==mode for r in summaries),joint_passed=sum(r['arm']==arm and r['mode']==mode and r['current_passed'] and r['compartment_passed'] for r in summaries),
            judgment_eligible=sum(r['arm']==arm and r['mode']==mode and r['judgment_eligible'] for r in summaries),
            eligible_joint_passed=sum(r['arm']==arm and r['mode']==mode and r['judgment_eligible'] and r['current_passed'] and r['compartment_passed'] for r in summaries)) for mode in ('imposed','autonomous')} for arm in ARMS},
        event_support=coverage,new_autonomous_failures=sum(r['new_failure_in_autonomous'] for r in contrasts),new_coupled_failures=sum(r['new_failure_when_coupled'] for r in coupling),
        masked_arms_diagnostic_only=True,all_arm_candidate='41r/timing_frozen_k',opened_protocols_from43=True,native_references_regenerated=True,
        speedup_claim=False,task41_original_gate_passed=False,decision='EXPLORATORY_LOCALIZATION_COMPLETE_REQUIRES_AUDIT',limits=p['limits'])
    n.p.save_json(out/'final_report.json',report);return report
