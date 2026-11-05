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

def stress(torch,net,spec,out,label,device):return p.stress(torch,net,spec,out,label,device)

def run(torch,root,source,archive,out,device='cuda'):
    out.mkdir(exist_ok=False);spec=json.loads((root/'experiments/task41r_preregistration_v1.json').read_text());manifest=json.loads((root/'experiments/results/task41r_protocol_v1/fit_manifest.json').read_text())
    assert n.p.digest(archive)==spec['recovery_sha256']
    _,prior,freeze,floors=p.r.verify_source(root,source,out.parent/'verified_original_41r')
    assert prior['quality']==spec['quality'];os.environ['GIADA_TASK41N_FROZEN']=str(out.parent/'verified_original_41r/frozen_source.zip.bin')
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
    assert list(models)==spec['arms'];n.p.save_json(out/'preregistration.json',spec);n.p.save_json(out/'fit_manifest.json',manifest);n.p.save_json(out/'gpu_preflight.json',pre)
    development=dict(prior,stress_seeds=spec['development_seeds']);dev={}
    for name,net in models.items():dev[name]=stress(torch,net,development,out,'development_'+name,device)
    base_dev=set(map(tuple,dev['baseline']['failed']));eligible=[k for k in models if not(set(map(tuple,dev[k]['failed']))-base_dev) and dev[k]['passed']>=dev['baseline']['passed']]
    nominee=max(eligible,key=lambda k:(dev[k]['passed'],-spec['arms'].index(k)))
    n.p.save_json(out/'nomination_freeze.json',dict(nominee=nominee,eligible=eligible,development=dev,opened41n_used_for_selection=False))
    cache=out.parent/'verified_original_41r'
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
        newstress=set(map(tuple,opened[name]['failed']))-oldfail;newdev=set(map(tuple,dev[name]['failed']))-base_dev
        primary=summaries['autonomous/primary'];newresponses=[x for x in primary['failed_responses'] if (x['case'],x['seed']) not in baseline_responses]
        acquired_ok=not newresponses and all(x['trajectory_passed']==x['trajectory_total'] for x in summaries.values()) and all(summaries[k]['passed'] for k in ('autonomous/voltage_solve64','imposed/primary','imposed/voltage_solve64'))
        results[name]=dict(development=dev[name],opened_stress=opened[name],summaries=summaries,new_stress_failures=sorted(newstress),new_development_failures=sorted(newdev),new_native_response_failures=newresponses,
            concrete_diagnostic_progress=opened[name]['passed']>61 and not newstress and not newdev and acquired_ok)
        n.p.save_json(out/'progress_results.json',results)
    timed={'dual':base,**{k:v for k,v in models.items() if k!='baseline'}}
    costs=n.benchmark(torch,timed,parent,data,(cfg,calcium,physics_base),prior,out,device)
    assert n.weights_hash(base)==base_hash
    report=dict(valid=True,complete=True,task='41r',decision='DIRECT_TIMING_FACTORIAL_COMPLETE_REQUIRES_AUDIT',nominee=nominee,arms=results,
        original_fp32_floor_failed_checks=[218,234],baseline_unchanged=True,thresholds_changed=False,abstraction_changed=False,inference_parameter_count_unchanged=True,
        stochastic_training_updates=0,deterministic_fits_prepared_locally=True,teacher_at_inference=False,all_parameters_finite=True,cost_rows=len(costs),task42_authorized=False,speedup_claim=False,independent_confirmation=False,limits=spec['limits'])
    n.p.save_json(out/'final_report.json',report);return report
