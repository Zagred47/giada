"""Task24b: paired optimizer diagnosis, unchanged physical gate contracts."""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from . import sodium_family_composition as s
from .hh_family_transfer import write, sha
from .joint_gate_symmetric_confirmation import _clip_per_seed


def config(root):
    base=json.loads((root/'experiments/task24_sodium_family_composition.json').read_text())
    amendment=json.loads((root/'experiments/task24b_sodium_control_diagnosis.json').read_text())
    cfg={**base,**amendment};cfg['data_seeds']={**base['data_seeds'],'fresh':amendment['fresh_seeds']}
    cfg['base_contract_sha256']=sha(root/'experiments/task24_sodium_family_composition.json')
    return cfg


def learning_rate(arm,step,cfg):
    if not arm.startswith('decay') or step <= cfg['decay_start']:return cfg['learning_rate']
    fraction=min(1.,(step-cfg['decay_start'])/(cfg['checkpoints'][-1]-cfg['decay_start']))
    return cfg['learning_rate_final']+.5*(cfg['learning_rate']-cfg['learning_rate_final'])*(1+math.cos(math.pi*fraction))


def channel_norms(model,torch):
    return torch.sqrt(sum(p.grad.square().flatten(2).sum(2) for p in model.parameters() if p.grad is not None))


def clip(model,torch,scope):
    norms=channel_norms(model,torch)
    if scope=='channel':
        scale=(1/norms.clamp_min(1e-12)).clamp_max(1)
        for p in model.parameters():
            if p.grad is not None:p.grad.mul_(scale.reshape(*scale.shape,*([1]*(p.ndim-2))))
    elif scope=='bundle':_clip_per_seed(model,torch,1.)
    else:raise ValueError(scope)
    return norms


def channel_step_equivalence(torch,width,seeds,device):
    """Per-channel clipping/Adam vs truly separate parameter groups."""
    model=s.model_factory(torch,width,'independent',seeds).to(device)
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    for p in model.parameters():
        p.grad=torch.ones_like(p)*torch.tensor([.1,1.,10.],device=device)[None,:,*([None]*(p.ndim-2))]
    grads={k:p.grad.clone() for k,p in model.named_parameters()}
    clip(model,torch,'channel');optimizer.step();error=0.
    for index in range(len(seeds)):
        for channel in range(3):
            params=[torch.nn.Parameter(v[index,channel].clone()) for v in before.values()]
            single=torch.optim.Adam(params,lr=.003)
            for p,g in zip(params,grads.values()):p.grad=g[index,channel].clone()
            torch.nn.utils.clip_grad_norm_(params,1.);single.step()
            error=max(error,max(float((p-v[index,channel]).detach().abs().max()) for p,v in zip(params,model.parameters())))
    return dict(width=width,clipping='channel',separate_adam_max_error=error,valid=error<1e-5)


def composed_hold(model,seeds,torch,device,voltage_seed):
    """Exact frozen-rate semigroup probe, not a claim of iterative FP32 fidelity."""
    v=np.random.default_rng(voltage_seed).uniform(-135,75,64)
    vals=np.c_[np.repeat(v,2),np.tile([[0.]*6,[1.]*6],(64,1)),np.ones(128)]
    result=[{} for _ in seeds]
    for horizon in (1000,10000):
        x=vals.copy();x[:,7]=horizon
        mm=s.evaluate(model,x,seeds,torch,device)
        for k,row in enumerate(mm):result[k][str(horizon)]=row
    return result


def held_score(metrics,cfg):
    return max([metrics[c][k]/t for c in s.CHANNELS for k,t in cfg['held_gates'].items() if t>0]+[metrics['pair'][k]/t for k,t in cfg['pair_gates'].items() if t>0])


def gate_pass(domains,paths,held,cfg):
    return (all(s.passes(domains[d],cfg['gates'],cfg['pair_gates']) for d in ('in_support','activation_boundary','state_extrema'))
        and all(s.passes(row,cfg['held_gates'],cfg['pair_gates']) for row in held.values())
        and all(row['finite'] and all(row[k]<=t for k,t in cfg['path_gates'].items())
            and all(all(row['per_channel'][c][k]<=t for k,t in cfg['path_channel_gates'].items()) for c in s.CHANNELS) for row in paths.values()))


def frozen_attribution(root,output,seeds,torch,device):
    source=root/'experiments/results/task24_kaggle_4fad2c6'
    freeze=json.loads((source/'selection_freeze.json').read_text());selected=freeze['selection']['independent']
    cp=source/selected['checkpoint'];assert sha(cp)==freeze['checkpoint_hashes'][cp.name]
    model=s.model_factory(torch,selected['width'],'independent',seeds).to(device)
    model.load_state_dict(torch.load(cp,map_location=device,weights_only=True))
    v=np.random.default_rng(240405).uniform(-135,75,64)
    values=np.c_[np.repeat(v,2),np.tile([[0.]*6,[1.]*6],(64,1)),np.ones(128)]
    ti,tt,_=s.targets(values);rows=[]
    original_x=torch.tensor(values,device=device,dtype=torch.float32)[None].expand(len(seeds),-1,-1)
    with torch.no_grad():_,fixed_inf,fixed_tau=model(original_x)
    for precision in ('float32','float64'):
        dtype=getattr(torch,precision)
        x=torch.tensor(values,device=device,dtype=dtype)[None].expand(len(seeds),-1,-1)
        pi,pt=fixed_inf.to(dtype),fixed_tau.to(dtype)
        for replacement in ('none','inf','tau','both'):
            i=torch.tensor(ti,device=device,dtype=dtype)[None].expand(len(seeds),-1,-1,-1) if replacement in ('inf','both') else pi
            t=torch.tensor(tt,device=device,dtype=dtype)[None].expand(len(seeds),-1,-1,-1) if replacement in ('tau','both') else pt
            state=x[...,1:7].reshape(len(seeds),128,3,2).transpose(1,2).clone();z=-torch.expm1(-1/t)
            with torch.no_grad():
                for step in range(1,10001):
                    state=(1-z)*state+z*i
                    if step in (1000,10000):
                        val=values.copy();val[:,7]=step;truth=s.targets(val)[2]
                        pred=state.cpu().double().numpy()
                        for n,seed in enumerate(seeds):
                            rows.append(dict(seed=seed,precision=precision,replacement=replacement,horizon_ms=step,Nap_h_rmse=float(np.sqrt(np.mean((pred[n,2,:,1]-truth[2,:,1])**2)))))
    write(output/'frozen_rate_precision_attribution.json',dict(selectable=False,old_fresh_diagnostic_only=True,identical_learned_rates_across_precisions=True,rows=rows))
    return rows


def run(root,output,cfg,revision):
    import torch
    from .gpu_baseline_runtime import environment_manifest
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['require_cuda'] and device!='cuda':raise RuntimeError('CUDA required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    seeds=cfg['seeds'];ds=cfg['data_seeds'];fit=s.data(ds['fit'],cfg['pool_size']);r=np.random.default_rng(ds['fit_regions']);n=len(fit)//4
    fit[:n,0]=r.uniform(-75,-25,n);fit[n:2*n,0]=r.uniform(-65,-35,n)
    dev=dict(uniform=s.data(ds['development'][0],2048),activation_boundary=s.data(ds['development'][1],2048,(-75,-25)),state_extrema=s.state_grid())
    np.savez_compressed(output/'fit_development.npz',fit=fit,**dev)
    ti,tt,ty=s.targets(fit);x,ti,tt,ty=[torch.tensor(a,dtype=torch.float32,device=device) for a in (fit,ti,tt,ty)]
    models={(a,w):s.model_factory(torch,w,'independent',seeds).to(device) for a in cfg['arms'] for w in cfg['widths']}
    eq=[s.equivalence(torch,w,'independent',seeds,device) for w in cfg['widths']]
    eq.extend(channel_step_equivalence(torch,w,seeds,device) for w in cfg['widths'])
    write(output/'equivalence_preflight.json',dict(valid=all(e['valid'] for e in eq),rows=eq))
    if not all(e['valid'] for e in eq):raise RuntimeError('Vectorized equivalence failed')
    opts={k:torch.optim.Adam(m.parameters(),lr=cfg['learning_rate']) for k,m in models.items()}
    norms={k:torch.zeros((len(seeds),3),device=device) for k in models};hits={k:torch.zeros((len(seeds),3),device=device) for k in models};bundle_hits={k:torch.zeros(len(seeds),device=device) for k in models}
    ladder=[];best={};one_step_best={};start=time.monotonic();rng=np.random.default_rng(ds['minibatches'])
    for step in range(cfg['checkpoints'][-1]+1):
        if step in cfg['checkpoints']:
            for (arm,width),model in models.items():
                mm={d:s.evaluate(model,v,seeds,torch,device) for d,v in dev.items()}
                held=composed_hold(model,seeds,torch,device,cfg['held_development_seed'])
                cp=output/f'checkpoint_{arm}_w{width}_step{step}.pt';torch.save(model.state_dict(),cp)
                scores=[];direct=[]
                for index,seed in enumerate(seeds):
                    domains={d:rows[index] for d,rows in mm.items()};sc=s.score(domains,cfg);combined=max(sc,max(held_score(a,cfg) for a in held[index].values()))
                    scores.append(combined);direct.append(sc)
                    ladder.append(dict(arm=arm,width=width,step=step,seed=seed,metrics=domains,composed_hold=held[index],score=combined,one_step_score=sc))
                candidate=dict(arm=arm,width=width,step=step,score=max(scores),checkpoint=cp.name)
                if arm not in best or candidate['score']<best[arm]['score']:best[arm]=candidate
                c={**candidate,'score':max(direct)}
                if arm not in one_step_best or c['score']<one_step_best[arm]['score']:one_step_best[arm]=c
            write(output/'development_ladder.json',ladder)
            print(f'[GIADA Task24b] checkpoint {step}/{cfg["checkpoints"][-1]} elapsed={(time.monotonic()-start)/60:.1f}min',flush=True)
        if step==cfg['checkpoints'][-1]:break
        idx=torch.tensor(rng.integers(0,len(fit),cfg['batch_size']),device=device);batch=x[idx][None].expand(len(seeds),-1,-1)
        for (arm,width),model in models.items():
            opt=opts[arm,width];opt.param_groups[0]['lr']=learning_rate(arm,step,cfg);opt.zero_grad(set_to_none=True)
            p,i,t=model(batch)
            loss=(p-ty[None,:,idx]).square().mean((1,2,3))+cfg['inf_weight']*(i-ti[None,:,idx]).square().mean((1,2,3))+cfg['log_tau_weight']*(t.log()-tt[None,:,idx].log()).square().mean((1,2,3))
            loss.sum().backward();gn=clip(model,torch,arm.split('_')[1]);norms[arm,width]+=gn.detach();hits[arm,width]+=(gn>1).detach();bundle_hits[arm,width]+=(gn.square().sum(1).sqrt()>1).detach();opt.step()
    write(output/'gradient_probes.json',[dict(arm=a,width=w,mean_channel_norm=(norms[a,w]/cfg['checkpoints'][-1]).cpu().tolist(),fraction_above_one=(hits[a,w]/cfg['checkpoints'][-1]).cpu().tolist(),bundle_fraction_above_one=(bundle_hits[a,w]/cfg['checkpoints'][-1]).cpu().tolist(),seed_order=seeds,channel_order=s.CHANNELS) for a,w in models])
    indexed={(r['arm'],r['width'],r['step'],r['seed']):r['score'] for r in ladder}
    contrasts=[]
    for width in cfg['widths']:
        for step in cfg['checkpoints'][1:]:
            for seed in seeds:
                for left,right,factor in [('constant_bundle','decay_bundle','schedule'),('constant_channel','decay_channel','schedule'),('constant_bundle','constant_channel','clipping'),('decay_bundle','decay_channel','clipping')]:
                    a,b=indexed[left,width,step,seed],indexed[right,width,step,seed]
                    contrasts.append(dict(factor=factor,reference=left,treatment=right,width=width,step=step,seed=seed,improvement_fraction=(a-b)/max(a,1e-12)))
    for arm in cfg['arms']:
        for seed in seeds:
            for width in cfg['widths']:
                if 30000 in cfg['checkpoints'] and 60000 in cfg['checkpoints']:
                    a,b=indexed[arm,width,30000,seed],indexed[arm,width,60000,seed]
                    contrasts.append(dict(factor='budget',arm=arm,width=width,seed=seed,improvement_fraction=(a-b)/max(a,1e-12)))
            for step in cfg['checkpoints'][1:]:
                if 16 in cfg['widths'] and 32 in cfg['widths']:
                    a,b=indexed[arm,16,step,seed],indexed[arm,32,step,seed]
                    contrasts.append(dict(factor='width',arm=arm,step=step,seed=seed,improvement_fraction=(a-b)/max(a,1e-12)))
    write(output/'paired_contrasts.json',contrasts)
    freeze=dict(selection=best,one_step_selection_diagnostic=one_step_best,fresh_accessed=False,config=cfg,code_revision=revision,checkpoint_hashes={p.name:sha(p) for p in output.glob('checkpoint_*.pt')})
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();write(output/'selection_freeze.json',freeze)
    fresh=dict(in_support=s.data(ds['fresh'][0],4096),activation_boundary=s.data(ds['fresh'][1],4096,(-75,-25)),state_extrema=s.state_grid(True,node_offset=.71))
    np.savez_compressed(output/'fresh.npz',**fresh)
    results=[]
    source=root/cfg['source_result'];old=json.loads((source/'selection_freeze.json').read_text())
    candidates={a:(models[a,c['width']],output/c['checkpoint'],c) for a,c in best.items()}
    for family in ('shared_heads','conditioned'):
        c=old['selection'][family];cp=source/c['checkpoint'];assert sha(cp)==old['checkpoint_hashes'][cp.name]
        candidates['frozen_'+family]=(s.model_factory(torch,c['width'],family,seeds).to(device),cp,c)
    for arm,(model,cp,selected) in candidates.items():
        model.load_state_dict(torch.load(cp,map_location=device,weights_only=True))
        mm={d:s.evaluate(model,v,seeds,torch,device) for d,v in fresh.items()}
        paths={name:s.path_rollout(model,seeds,torch,device,dt) for name,dt in cfg['path_dt_ms'].items()}
        held=s.held_rollout(model,seeds,torch,device,cfg['rollout_horizons_steps'],cfg['held_fresh_seed'])
        for i,seed in enumerate(seeds):
            domains={d:r[i] for d,r in mm.items()};pp={name:r[i] for name,r in paths.items()}
            results.append(dict(arm=arm,width=selected['width'],step=selected['step'],seed=seed,metrics=domains,path_rollout=pp,held_rollout=held[i],passed=gate_pass(domains,pp,held[i],cfg),parameter_count=sum(p.numel() for p in model.parameters())//len(seeds)))
        print('[GIADA Task24b] frozen evaluation '+arm,flush=True)
    attribution=frozen_attribution(root,output,seeds,torch,device)
    passing={a:all(r['passed'] for r in results if r['arm']==a) for a in candidates}
    primary=passing['decay_channel']
    counts={a:next(r['parameter_count'] for r in results if r['arm']==a) for a in candidates}
    sharing={a:primary and passing[a] and counts[a]<counts['decay_channel'] for a in ('frozen_shared_heads','frozen_conditioned')}
    report=dict(schema_version='giada-task24b-v1',valid=True,per_arm_passed=passing,primary_repair_passed=primary,task25_preparation_authorized=primary,sharing_against_repaired_reference=sharing,selected_parameter_counts=counts,selection=best,learned=results,fresh_used_for_selection=False,old_fresh_used_for_selection=False,oracle_probes_selectable=False,training_and_evaluation_seconds=time.monotonic()-start,environment=environment_manifest(torch),scope=cfg['limits'],original_task24_decision_unchanged=True)
    write(output/'final_report.json',report);return report
