"""Targeted existing-parameter repairs; direct supervision vs trajectory loss."""
import copy,json,os,zipfile
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from . import task41u_objective_features as u
r=u.r;n=u.n
TARGETS={'k':((0,12),),'nap':((0,8),(1,9)),'nap_u':((0,8),),'nap_z':((1,9),)}
BANK={(0,8):0,(0,9):1,(0,12):2,(0,13):3,(1,8):4,(1,9):5,(1,12):6,(1,13):7}
ATOMS=tuple(f'{channel}_{loss}_{basis}' for channel in ('k','nap') for loss in ('indirect','direct') for basis in ('heads','local'))+('nap_u_direct_local','nap_z_direct_local')
ARMS=ATOMS+('recomposed',)

@contextmanager
def challenge_paths(spec):
    original=n.stress_data
    def generate(prior):
        ordinary=original(prior);extra=original(dict(prior,stress_seeds=spec['challenge_seeds']))
        for i,row in enumerate(extra):
            row['v']=(row['v']+(-2.,0.,2.)[(i//2)%3]).astype('float32');row['family']='challenge_'+row['family']
        return ordinary+extra
    n.stress_data=generate
    try:yield
    finally:n.stress_data=original

def development(torch,model,prior,spec,out,label,device):
    with challenge_paths(spec):summary=r.stress(torch,model,dict(prior,stress_seeds=spec['development_seeds']),out,label,device)
    rows=json.loads((out/label/'fresh_stress_results.json').read_text())['dual']['rows']
    return summary,rows

def margin(rows,channel=None):
    names=('K_Pst',) if channel=='k' else ('Nap_Et2',) if channel=='nap' else ('K_Pst','Nap_Et2')
    values=[]
    for row in rows:
        for c in row['metrics']['channels']:
            if c['channel'] not in names:continue
            e=c['event']
            if e['status']=='missing_event':values.append(1e12)
            elif e['status']=='evaluable':values.append(max(e['onset_error_ms']/.3,e['offset_error_ms']/.3,e['peak_error_ms']/.2))
    return max(values,default=0.)

def rank(summary,rows,channel=None):
    return summary['passed'],-margin(rows,channel)

def configure(torch,model,keys,local):
    for p in model.parameters():p.requires_grad_(False)
    masks={}
    for name,p in model.named_parameters():
        if local and name=='parent.correction':
            mask=torch.zeros_like(p);mask[..., [BANK[k] for k in keys]]=1
        elif not local and name in ('parent.c','parent.d'):
            mask=torch.zeros_like(p);mask[..., [g+17*k for k,g in keys]]=1
        else:continue
        p.requires_grad_(True);masks[name]=mask
    return [(name,p) for name,p in model.named_parameters() if p.requires_grad],masks

def curve_labels(torch,device):
    # Fixed curves, not opened stress inputs; quantized exactly like inference.
    v=np.unique(np.r_[np.linspace(-100,60,1024),np.linspace(-70,-30,1024)]).astype('float32')
    a,b=n.p.coefficients(v.astype('float64'),np.full(len(v),1e-4),.1)
    target={(kind,gate):torch.tensor(np.log(np.clip((a if kind==0 else b)[:,gate],1e-12,1-1e-12)/(1-np.clip((a if kind==0 else b)[:,gate],1e-12,1-1e-12))),device=device) for keys in TARGETS.values() for kind,gate in keys}
    return torch.tensor(v,device=device),target

def coefficient_loss(torch,model,grid,target,keys):
    ru,rz=model.raw(grid[None].expand(3,-1),torch.full((3,len(grid)),1e-4,device=grid.device))
    return sum((((ru if k==0 else rz)[...,g].double()-target[(k,g)][None])/1e-3).square().mean(1) for k,g in keys)/len(keys)

def compose(torch,incumbent,k_model,nap_model):
    model=copy.deepcopy(incumbent)
    with torch.no_grad():
        for source,keys in ((k_model,TARGETS['k']),(nap_model,TARGETS['nap'])):
            for kind,gate in keys:
                col=gate+17*kind;model.parent.c[...,col].copy_(source.parent.c[...,col]);model.parent.d[...,col].copy_(source.parent.d[...,col])
                bank=BANK[(kind,gate)];model.parent.correction[...,bank].copy_(source.parent.correction[...,bank])
    return model

def train(torch,incumbent,prior,spec,out,device):
    grid,targets=curve_labels(torch,device);data=u.tensor_data(torch,u.data_arrays(prior,spec['fit_path_seeds']),device)
    with torch.no_grad():
        assert tuple(incumbent.parent.correction.shape)==(3,32,8)
        ru,rz=incumbent.raw(grid[None].expand(3,-1),torch.full((3,len(grid)),1e-4,device=device))
        basis=(1-(grid[:,None]-torch.linspace(-100,60,32,device=device)).abs()/(160/31)).clamp_min(0)
        extras=.1*torch.tanh(torch.bmm(basis[None].expand(3,-1,-1).double(),incumbent.parent.correction.double()).float()/.1)
        bounds={}
        for keys in TARGETS.values():
            for kind,gate in keys:
                required=targets[(kind,gate)][None]-((ru if kind==0 else rz)[...,gate]-extras[...,BANK[(kind,gate)]])
                bounds[f'{kind}_{gate}']=dict(max_required=required.abs().amax(1).cpu().tolist(),fraction_outside_local_bound=(required.abs()>.1).double().mean(1).cpu().tolist())
        n.p.save_json(out/'local_bank_feasibility.json',dict(rows=bounds,scope='Offline anchor representability diagnostic only; no scientific gates changed'))
    models={name:copy.deepcopy(incumbent) for name in ATOMS};params={};masks={};masters={};optimizers={}
    for name,model in models.items():
        channel='k' if name.startswith('k_') else 'nap_u' if name.startswith('nap_u_') else 'nap_z' if name.startswith('nap_z_') else 'nap'
        params[name],masks[name]=configure(torch,model,TARGETS[channel],name.endswith('local'))
        masters[name]=[torch.nn.Parameter(p.detach().double().clone()) for _,p in params[name]]
        optimizers[name]=torch.optim.Adam(masters[name],lr=spec['learning_rate'])
    rng=np.random.default_rng(spec['minibatch_seed']);order=np.arange(24);records=[]
    curve_order=torch.tensor(np.random.default_rng(spec['minibatch_seed']+1).permutation(len(grid)),device=device)
    u.export_checkpoint(out,0,models,optimizers,records,masters,rng.bit_generator.state,order)
    print('[GIADA41v] targeted10arm training starting',flush=True)
    for step in range(1,max(spec['checkpoints'])+1):
        offset=((step-1)*spec['batch_paths'])%24
        if offset==0:order=rng.permutation(24)
        ids=torch.tensor(order[offset:offset+spec['batch_paths']],device=device)
        for name,model in models.items():
            channel='k' if name.startswith('k_') else 'nap_u' if name.startswith('nap_u_') else 'nap_z' if name.startswith('nap_z_') else 'nap'
            opt=optimizers[name];opt.zero_grad(set_to_none=True)
            for _,p in params[name]:p.grad=None
            if '_direct_' in name:
                count=24 if step%2 else 48
                curve_ids=curve_order[(torch.arange(count,device=device)+(step-1)*24)%len(grid)]
                loss=coefficient_loss(torch,model,grid[curve_ids],{key:value[curve_ids] for key,value in targets.items()},TARGETS[channel])
            else:
                which=1 if channel=='k' else 0;restricted=dict(data);restricted['window']=data['window'].clone();restricted['window'][...,1-which]=0
                loss=u.loss_for(torch,model,restricted,ids,lambda *args:u.objective(torch,*args),torch.tensor(0.,device=device))
            assert torch.isfinite(loss).all();loss.sum().backward()
            for key,p in params[name]:
                assert p.grad is None or torch.isfinite(p.grad).all()
                if p.grad is not None:p.grad.mul_(masks[name][key])
            norm=sum(p.grad.double().square().flatten(1).sum(1) for _,p in params[name] if p.grad is not None).sqrt().clamp_min(1e-30);factor=(1/norm).clamp_max(1)
            for master,(_,p) in zip(masters[name],params[name]):master.grad=None if p.grad is None else p.grad.double()*factor.reshape(3,*([1]*(p.ndim-1)))
            opt.step()
            with torch.no_grad():
                for master,(_,p) in zip(masters[name],params[name]):p.copy_(master.to(p.dtype))
            assert all(torch.isfinite(p).all() for _,p in params[name])
            if step%64==0:records.append(dict(step=step,arm=name,loss=loss.detach().cpu().tolist(),gradient_norm=norm.cpu().tolist()))
        if step in spec['checkpoints']:u.export_checkpoint(out,step,models,optimizers,records,masters,rng.bit_generator.state,order)
        if step%64==0:n.p.save_json(out/'training_progress.json',records);print(f'[GIADA41v] {step}/{max(spec["checkpoints"])} all10arms finite',flush=True)
    # Assert only the exact declared parameter entries changed, not merely shapes.
    for name,model in models.items():
        for key,value in model.state_dict().items():
            original=incumbent.state_dict()[key]
            if key in masks[name]:assert torch.equal(value*(1-masks[name][key]),original*(1-masks[name][key]))
            else:assert torch.equal(value,original),(name,key)
    inc_dev,inc_rows=development(torch,incumbent,prior,spec,out,'development_incumbent',device);basefail=set(map(tuple,inc_dev['failed']))
    chosen={};selected={};budget_rows=[]
    for name in ATOMS:
        channel='k' if name.startswith('k_') else 'nap';options=[(0,incumbent,inc_dev,inc_rows)]
        for step in spec['checkpoints']:
            payload=torch.load(out/f'checkpoint_{step}.pt',map_location=device,weights_only=True);model=copy.deepcopy(incumbent);model.load_state_dict(payload['models'][name])
            for p in model.parameters():p.requires_grad_(False)
            summary,rows=development(torch,model,prior,spec,out,f'development_{name}_{step}',device)
            options.append((step,model,summary,rows));budget_rows.append(dict(arm=name,step=step,development=summary,target_worst_margin=margin(rows,channel)))
        eligible=[x for x in options if not(set(map(tuple,x[2]['failed']))-basefail)]
        winner=max(eligible,key=lambda x:(*rank(x[2],x[3],channel),-x[0]));chosen[name]=winner[1].eval();selected[name]=winner
        n.p.save_json(out/f'selection_{name}.json',dict(step=winner[0],development=winner[2],target_worst_margin=margin(winner[3],channel),overall_worst_margin=margin(winner[3]),weights_sha256=n.weights_hash(winner[1])))
    def pick(prefix,channel):
        names=[k for k in ATOMS if k.startswith(prefix)]
        best=max(names,key=lambda name:(*rank(selected[name][2],selected[name][3],channel),-spec['arms'].index(name)))
        return best if rank(selected[best][2],selected[best][3],channel)>rank(inc_dev,inc_rows,channel) else 'incumbent'
    k_name,nap_name=pick('k_','k'),pick('nap_','nap')
    chosen['recomposed']=compose(torch,incumbent,chosen.get(k_name,incumbent),chosen.get(nap_name,incumbent))
    summary,rows=development(torch,chosen['recomposed'],prior,spec,out,'development_recomposed',device)
    n.p.save_json(out/'selection_recomposed.json',dict(step=None,development=summary,weights_sha256=n.weights_hash(chosen['recomposed']),K_Pst=k_name,Nap_Et2=nap_name,target_worst_margin=margin(rows),overall_worst_margin=margin(rows)))
    n.p.save_json(out/'budget_development.json',dict(rows=budget_rows,opened41n_used=False,training_finished_before_development=True))
    return chosen

def gpu_oracles(torch,incumbent,prior,out,device):
    meta=n.stress_data(prior);values=np.unique(np.stack([x['v'] for x in meta]).astype('float32'));aa,bb=n.p.coefficients(values.astype('float64'),np.full(len(values),1e-4),.1)
    class Oracle(torch.nn.Module):
        def __init__(self,changes):
            super().__init__();self.inner=incumbent;self.changes=changes
            self.register_buffer('grid',torch.tensor(values,device=device));self.register_buffer('u',torch.tensor(aa,device=device,dtype=torch.float32));self.register_buffer('z',torch.tensor(bb,device=device,dtype=torch.float32))
        def forward(self,v,ca,state):
            a,b=self.inner.raw(v,ca);a=torch.sigmoid(a);b=torch.sigmoid(b);ix=torch.searchsorted(self.grid,v.contiguous())
            for kind,gate in self.changes:(a if kind==0 else b)[...,gate]=(self.u if kind==0 else self.z)[ix,gate]
            return state+b*(a-state)
    probes={'control':(),'K_u12':((0,12),),'Nap_u8':((0,8),),'Nap_z9':((1,9),),'Nap_pair':((0,8),(1,9)),'combined':((0,12),(0,8),(1,9))}
    results={label:r.stress(torch,Oracle(keys),prior,out,'gpu_oracle_'+label,device) for label,keys in probes.items()}
    assert results['control']['passed']==66
    n.p.save_json(out/'gpu_causal_replication.json',dict(results=results,teacher_oracles_selectable=False,canonical_fp32_state_update=True))

def run(torch,root,source,archive,out,device='cuda'):
    out.mkdir(exist_ok=False);spec=json.loads((root/'experiments/task41v_preregistration_v1.json').read_text());manifest=json.loads((root/'experiments/results/task41v_protocol_v1/fit_manifest.json').read_text())
    assert n.p.digest(archive)==spec['recovery_sha256']
    _,prior,freeze,floors=r.p.r.verify_source(root,source,out.parent/'verified_original_41v')
    assert prior['quality']==spec['quality'];os.environ['GIADA_TASK41N_FROZEN']=str(out.parent/'verified_original_41v/frozen_source.zip.bin')
    nets,parent=n.load_models(torch,root,device);base=nets['dual'];base_hash=n.weights_hash(base)
    pre=n.parity(torch,nets,device,prior);assert pre['eligible']==freeze['eligible'];assert base_hash==manifest['baseline_weights_sha256']
    with np.load(root/'experiments/results/task41r_protocol_v1/heads_timing_frozen_k.npz') as a:incumbent=r.materialize(torch,base,dict(a))
    assert n.weights_hash(incumbent)==manifest['incumbent_weights_sha256']
    n.p.save_json(out/'preregistration.json',spec);n.p.save_json(out/'gpu_preflight.json',pre)
    trained=train(torch,incumbent,prior,spec,out,device)
    models={'baseline':base,'incumbent':incumbent,**trained}
    assert list(models)==spec['arms']
    for name,model in models.items():
        assert all(torch.isfinite(x).all() for x in model.parameters())
        assert sum(x.numel() for x in model.parameters())==sum(x.numel() for x in base.parameters())
        torch.save(model.state_dict(),out/f'frozen_{name}.pt')
    inc_rows=json.loads((out/'development_incumbent/fresh_stress_results.json').read_text())['dual']['rows']
    dev={'incumbent':dict(passed=sum(x['metrics']['passed'] for x in inc_rows),total=len(inc_rows),failed=[(x['model_seed'],x['family'],x['input_seed'],x['history'],x['panel']) for x in inc_rows if not x['metrics']['passed']])}
    dev['baseline'],_=development(torch,base,prior,spec,out,'development_baseline',device)
    for name in ARMS:dev[name]=json.loads((out/f'selection_{name}.json').read_text())['development']
    base_dev=set(map(tuple,dev['incumbent']['failed']))
    eligible=[k for k in spec['selectable'] if not(set(map(tuple,dev[k]['failed']))-base_dev) and dev[k]['passed']>=dev['incumbent']['passed']]
    def selected_margin(k):
        if k=='incumbent':return margin(inc_rows)
        return json.loads((out/f'selection_{k}.json').read_text())['overall_worst_margin']
    nominee=max(eligible,key=lambda k:(dev[k]['passed'],-selected_margin(k),-spec['arms'].index(k)))
    n.p.save_json(out/'nomination_freeze.json',dict(nominee=nominee,eligible=eligible,development=dev,opened41n_used_for_selection=False))
    cache=out.parent/'verified_original_41v'
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
        opened[name]=r.stress(torch,net,prior,out,'opened41n_'+name,device)
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
    gpu_oracles(torch,incumbent,prior,out,device)
    timed={'dual':base,**{k:v for k,v in models.items() if k!='baseline'}}
    costs=n.benchmark(torch,timed,parent,data,(cfg,calcium,physics_base),prior,out,device)
    assert n.weights_hash(base)==base_hash
    report=dict(valid=True,complete=True,task='41v',decision='TARGETED_COEFFICIENT_FACTORIAL_COMPLETE_REQUIRES_AUDIT',nominee=nominee,arms=results,
        original_fp32_floor_failed_checks=[218,234],baseline_unchanged=True,thresholds_changed=False,abstraction_changed=False,inference_parameter_count_unchanged=True,
        stochastic_training_updates_per_arm=max(spec['checkpoints']),deterministic_fits_prepared_locally=False,training_performed_on_gpu=True,teacher_at_inference=False,all_parameters_finite=True,cost_rows=len(costs),task42_authorized=False,speedup_claim=False,independent_confirmation=False,limits=spec['limits'])
    n.p.save_json(out/'final_report.json',report);return report
