"""Same-head deterministic repairs and unchanged GPU behavioral tests."""
import copy,io,json,os,zipfile
from pathlib import Path
import numpy as np
from . import task41p_tail_repair as p
n=p.n
GROUPS={'Nap_Et2':(4,(0,1)), 'Im':(6,(2,)), 'K_Pst':(7,(3,4)), 'SK_E2':(9,(5,))}
HEADS=((0,8),(0,9),(0,11),(0,12),(0,13),(1,8),(1,9),(1,12),(1,13),(0,16))

def materialize(torch,base,weights):
    net=copy.deepcopy(base)
    with torch.no_grad():
        for kind,gate in HEADS:
            value=torch.tensor(weights[f'h{kind}_{gate}'],device=net.a.device,dtype=torch.float32)
            if gate==16:net.parent.sc[:,:,0].copy_(value[:,:-1]);net.parent.sd[:,0].copy_(value[:,-1])
            else:
                col=gate+(17 if kind else 0);net.parent.c[:,:,col].copy_(value[:,:-1]);net.parent.d[:,col].copy_(value[:,-1])
    for value in net.parameters():value.requires_grad_(False)
    return net.eval()

def head_weights(base):
    result={}
    for kind,gate in HEADS:
        if gate==16:w=base.parent.sc[:,:,:1];b=base.parent.sd[:,None,:1]
        else:
            col=gate+(17 if kind else 0);w=base.parent.c[:,:,col:col+1];b=base.parent.d[:,None,col:col+1]
        result[f'h{kind}_{gate}']=np.concatenate((w.detach().cpu().numpy(),b.detach().cpu().numpy()),1)[...,0]
    return result

def channel_current(gates,v,mult,name):
    channel,ids=GROUPS[name];x=gates[...,ids[0]];power=int(n.p.ionic.POWERS[channel])
    opening=x**power
    if len(ids)==2:opening*=gates[...,ids[1]]
    return opening*n.p.ionic.GBAR[channel]*mult[...,channel]*(v-n.p.ionic.REVERSALS[channel])

def current_derivatives(gates,v,mult,name):
    channel,ids=GROUPS[name];x=gates[...,ids[0]];power=int(n.p.ionic.POWERS[channel])
    scale=n.p.ionic.GBAR[channel]*mult[...,channel]*(v-n.p.ionic.REVERSALS[channel])
    if len(ids)==1:return [scale*power*x**(power-1)]
    return [scale*power*x**(power-1)*gates[...,ids[1]],scale*x**power]

def affine_sensitivity(state,su,sz,u,z,features):
    """Exact derivative of the frozen-feature affine recurrence in real arithmetic."""
    new_su=(1-z)[...,None]*su+(z*u*(1-u))[...,None]*features
    new_sz=(1-z)[...,None]*sz+(z*(1-z)*(u-state))[...,None]*features
    return state+z*(u-state),new_su,new_sz

def timing_step(torch,delta0,whiten,jacobian,target):
    """Whitened regularized least squares: no SGD or architecture change."""
    if not len(target):return delta0
    b=torch.tensor(jacobian,dtype=torch.float64)/np.sqrt(len(target));y=torch.tensor(target,dtype=torch.float64)/np.sqrt(len(target))
    c=b@whiten;rhs=y-b@delta0
    a=torch.cat((torch.eye(len(delta0),dtype=torch.float64),c),0)
    theta=torch.linalg.lstsq(a,torch.cat((torch.zeros_like(delta0),rhs)),rcond=1e-12,driver='gelsd').solution
    return delta0+whiten@theta

def keys_for(name,joint):
    _,ids=GROUPS[name];gates=[p.EQ[i] for i in ids]
    return [(0,g) for g in gates]+([(1,g) for g in gates] if name=='Nap_Et2' or (name=='K_Pst' and joint) else [])

def tail_probe(folder,spec):
    """Read-only sensitivity diagnostic; original timing evaluator is untouched."""
    with np.load(folder/'fresh_stress_inputs.npz') as a:data=dict(a)
    with np.load(folder/'stress_dual.npz') as a:gates=a['gates']
    v=data['voltage'];mult=data['multipliers']
    truth=n.p.readout(v,data['target_gates'],mult,n.p.ionic.REVERSALS)[1]
    metadata=n.stress_data(spec);rows=[]
    for si,seed in enumerate(spec['seeds']):
        pred=n.p.readout(v,gates[:,si],mult,n.p.ionic.REVERSALS)[1]
        for ci,meta in enumerate(metadata):
            for channel in (4,7):
                target=truth[:,ci,channel];actual=pred[:,ci,channel]
                event=n.p.event_metrics(target,actual,.1,spec['quality']['current_scale_floor_ma_cm2'],spec['quality']['peak_plateau_relative_band'])
                if event['status']!='evaluable':continue
                excursion=np.abs(target-target[0]);ix=int(np.flatnonzero(excursion>=.1*excursion.max())[-1])
                slope=float((target[min(ix+1,len(target)-1)]-target[max(ix-1,0)])/.2)
                residual=float(actual[ix]-target[ix])
                # The event baseline also changes with candidate current; show both.
                baseline_residual=float(actual[0]-target[0])
                rows.append(dict(seed=seed,case=ci,family=meta['family'],channel=n.p.ionic.CHANNELS[channel],tick=ix,
                    slope=slope,residual=residual,baseline_residual=baseline_residual,
                    local_shift_ms=None if abs(slope)<1e-20 else -residual/slope,
                    baseline_adjusted_shift_ms=None if abs(slope)<1e-20 else -(residual-baseline_residual)/slope,
                    observed_event=event,diagnostic_only=True))
    n.p.save_json(folder/'tail_sensitivity_probe.json',dict(rows=rows,evaluator_changed=False,interpretation='Local tangent approximation only; multiple crossings/flat tails can invalidate it.'))

def stress(torch,net,spec,out,label,device):
    result=p.stress(torch,net,spec,out,label,device)
    tail_probe(out/label,spec)
    return result

def head_precision_probe(torch,models,device):
    values={}
    with torch.no_grad():
        v=torch.linspace(-100,60,257,device=device)[None].expand(3,-1)
        for name,net in models.items():
            f=torch.nn.functional.silu(torch.bmm(((v+40)/80)[...,None],net.parent.a)+net.parent.b)
            raw=torch.bmm(f,net.parent.c)+net.parent.d[:,None]
            diagnostic=torch.bmm(f.double(),net.parent.c.double())+net.parent.d[:,None].double()
            values[name]={str(i):float((raw.double()-diagnostic)[...,i].abs().max()) for i in (8,9,12,13,25,26,29,30)}
    return dict(rows=values,scope='Same FP32 features; final GEMM reduction32 vs64, diagnostic only, no changed inference')

def run(torch,root,source,archive,out,device='cuda'):
    out.mkdir(exist_ok=False);spec=json.loads((root/'experiments/task41t_preregistration_v1.json').read_text());manifest=json.loads((root/'experiments/results/task41t_protocol_v1/fit_manifest.json').read_text())
    assert n.p.digest(archive)==spec['recovery_sha256']
    _,prior,freeze,floors=p.r.verify_source(root,source,out.parent/'verified_original_41t')
    assert prior['quality']==spec['quality'];os.environ['GIADA_TASK41N_FROZEN']=str(out.parent/'verified_original_41t/frozen_source.zip.bin')
    nets,parent=n.load_models(torch,root,device);base=nets['dual'];base_hash=n.weights_hash(base)
    pre=n.parity(torch,nets,device,prior);assert pre['eligible']==freeze['eligible'];assert base_hash==manifest['baseline_weights_sha256']
    models={'baseline':base}
    for name,entry in manifest['models'].items():
        path=root/entry['path'];assert n.p.digest(path)==entry['sha256']
        with np.load(path) as a:models[name]=materialize(torch,base,dict(a))
        assert n.weights_hash(models[name])==entry['weights_sha256']
        assert sum(x.numel() for x in models[name].parameters())==sum(x.numel() for x in base.parameters())
        assert all(torch.isfinite(x).all() for x in models[name].parameters())
        torch.save(models[name].state_dict(),out/f'frozen_{name}.pt')
    assert n.weights_hash(models['incumbent'])==manifest['incumbent_weights_sha256']
    assert list(models)==spec['arms'][:-1];n.p.save_json(out/'preregistration.json',spec);n.p.save_json(out/'fit_manifest.json',manifest);n.p.save_json(out/'gpu_preflight.json',pre)
    n.p.save_json(out/'numerical_preflight.json',json.loads((root/'experiments/results/task41t_protocol_v1/numerical_preflight.json').read_text()))
    n.p.save_json(out/'gpu_head_reduction_probe.json',head_precision_probe(torch,models,device))
    development=dict(prior,stress_seeds=spec['development_seeds']);dev={}
    for name,net in models.items():dev[name]=stress(torch,net,development,out,'development_'+name,device)
    base_dev=set(map(tuple,dev['incumbent']['failed']))
    def choose(prefix):
        candidates=['incumbent']+[k for k in models if k.startswith(prefix)]
        candidates=[k for k in candidates if not(set(map(tuple,dev[k]['failed']))-base_dev) and dev[k]['passed']>=dev['incumbent']['passed']]
        return max(candidates,key=lambda k:(dev[k]['passed'],-spec['arms'].index(k)))
    chosen_k,chosen_nap=choose('k_'),choose('nap_')
    combined=head_weights(models['incumbent'])
    for source_name,gates in ((chosen_k,(12,13)),(chosen_nap,(8,9))):
        values=head_weights(models[source_name])
        for gate in gates:
            for kind in (0,1):combined[f'h{kind}_{gate}']=values[f'h{kind}_{gate}'].copy()
    models['recomposed']=materialize(torch,base,combined)
    assert sum(x.numel() for x in models['recomposed'].parameters())==sum(x.numel() for x in base.parameters())
    assert all(torch.isfinite(x).all() for x in models['recomposed'].parameters())
    torch.save(models['recomposed'].state_dict(),out/'frozen_recomposed.pt')
    n.p.save_json(out/'composition_freeze.json',dict(K_Pst=chosen_k,Nap_Et2=chosen_nap,weights_sha256=n.weights_hash(models['recomposed']),sources={k:n.weights_hash(models[k]) for k in (chosen_k,chosen_nap)},opened41n_used=False))
    dev['recomposed']=stress(torch,models['recomposed'],development,out,'development_recomposed',device)
    assert list(models)==spec['arms']
    eligible=[k for k in spec['selectable'] if not(set(map(tuple,dev[k]['failed']))-base_dev) and dev[k]['passed']>=dev['incumbent']['passed']]
    nominee=max(eligible,key=lambda k:(dev[k]['passed'],-spec['arms'].index(k)))
    n.p.save_json(out/'nomination_freeze.json',dict(nominee=nominee,eligible=eligible,development=dev,opened41n_used_for_selection=False))
    cache=out.parent/'verified_original_41t'
    with np.load(cache/'native_shadow_traces.npz') as a:data=dict(a)
    meta=json.loads((cache/'cases.json').read_text());_,cfg,calcium,physics_base=n.contract(root);cfg=n.old_native.x.calibration_config(prior,cfg)
    refs={}
    for policy in ('primary','voltage_solve64'):
        with np.load(cache/f'reference_autonomous_{policy}.npz') as a:refs[policy]=dict(a)
    imposed={k:data[k].T for k in ('voltage','calcium')};irefs={k:n.m.rollout(torch,None,data,(cfg,calcium,physics_base),device,k,imposed) for k in refs}
    with zipfile.ZipFile(archive) as z:
        old=json.loads(z.read('final_report.json'));oldrows=json.loads(z.read('fresh_stress_results.json'))['dual']['rows']
    oldfail={(x['model_seed'],x['family'],x['input_seed'],x['history'],x['panel']) for x in oldrows if not x['metrics']['passed']}
    baseline_responses={(106,s) for s in prior['seeds']}|{(122,s) for s in prior['seeds']};results={};opened={}
    for name,net in models.items():
        opened[name]=stress(torch,net,prior,out,'opened41n_'+name,device)
        if name=='incumbent':assert opened[name]['passed']==66,'Frozen41r incumbent reproduction failed'
        if name=='baseline':assert opened[name]['passed']==61 and set(map(tuple,opened[name]['failed']))==oldfail
        summaries={}
        if name=='baseline':
            summaries={f'{mode}/{ref}':old['summaries'][f'dual/{mode}/{ref}'] for mode in ('autonomous','imposed') for ref in refs}
        else:
            for mode,forcing,truths in [('autonomous',None,refs),('imposed',imposed,irefs)]:
                pred=n.m.rollout(torch,net,data,(cfg,calcium,physics_base),device,'voltage_solve64',forcing)
                assert all(np.isfinite(x).all() for x in pred.values())
                for ref,truth in truths.items():
                    measured=n.m.measure(pred,truth,data,meta,prior,cfg,mode,'41n');summaries[f'{mode}/{ref}']=n.m.summary(measured)
                    n.p.save_json(out/f'{name}_{mode}_{ref}.json',measured)
        incumbent_failed=set(map(tuple,opened['incumbent']['failed'])) if 'incumbent' in opened else oldfail
        newstress=set(map(tuple,opened[name]['failed']))-incumbent_failed;newdev=set(map(tuple,dev[name]['failed']))-base_dev
        primary=summaries['autonomous/primary'];newresponses=[x for x in primary['failed_responses'] if (x['case'],x['seed']) not in baseline_responses]
        acquired_ok=not newresponses and all(x['trajectory_passed']==x['trajectory_total'] for x in summaries.values()) and all(summaries[k]['passed'] for k in ('autonomous/voltage_solve64','imposed/primary','imposed/voltage_solve64'))
        results[name]=dict(development=dev[name],opened_stress=opened[name],summaries=summaries,new_stress_failures=sorted(newstress),new_development_failures=sorted(newdev),new_native_response_failures=newresponses,
            concrete_diagnostic_progress=opened[name]['passed']>66 and not newstress and not newdev and acquired_ok)
        n.p.save_json(out/'progress_results.json',results)
    timed={'dual':base,**{k:v for k,v in models.items() if k!='baseline'}}
    costs=n.benchmark(torch,timed,parent,data,(cfg,calcium,physics_base),prior,out,device)
    assert n.weights_hash(base)==base_hash
    report=dict(valid=True,complete=True,task='41t',decision='NONLINEAR_RELAXATION_FACTORIAL_COMPLETE_REQUIRES_AUDIT',nominee=nominee,arms=results,
        original_fp32_floor_failed_checks=[218,234],baseline_unchanged=True,thresholds_changed=False,abstraction_changed=False,inference_parameter_count_unchanged=True,
        stochastic_training_updates=0,deterministic_fits_prepared_locally=True,teacher_at_inference=False,all_parameters_finite=True,cost_rows=len(costs),task42_authorized=False,speedup_claim=False,independent_confirmation=False,limits=spec['limits'])
    n.p.save_json(out/'final_report.json',report);return report
