"""Task27: paired current-loss factorial on the five-channel Task26 cell."""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import heterogeneous_mechanism_composition as h
from . import joint_heterogeneous_comparison as j
from .hh_family_transfer import sha, write
from .joint_gate_symmetric_confirmation import _clip_per_seed


def config(root):
    patch_path = root / 'experiments/task27_current_supervision.json'
    patch = json.loads(patch_path.read_text(encoding='utf-8'))
    prior_path = root / patch['parent_contract']
    inherited = j.config(root)
    return {**inherited, **patch, 'task27_contract_sha256': sha(patch_path),
            'parent_contract_sha256': sha(prior_path)}


def verify_parent(root, cfg):
    report_path = root / cfg['parent_run']
    freeze_path = root / cfg['parent_freeze']
    process_path = root / cfg['parent_process']
    report = json.loads(report_path.read_text(encoding='utf-8'))
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    claimed = freeze.pop('freeze_sha256')
    actual = hashlib.sha256(json.dumps(freeze, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    process = json.loads(process_path.read_text(encoding='utf-8'))
    if actual != claimed or freeze['fresh_accessed'] or process != {'phase':'gpu','returncode':0}:
        raise RuntimeError('Task26 freeze or worker status invalid')
    if not report['valid'] or not report['comparison_passed'] or not report['task27_preparation_authorized']:
        raise RuntimeError('Task26 scientific prerequisite failed')
    if report['fresh_used_for_selection'] or report['current_supervision_used']:
        raise RuntimeError('Task26 selection or loss contract changed')
    if freeze['selection'] != report['selection']:
        raise RuntimeError('Task26 freeze/report selection mismatch')
    if sorted(report['per_family_passed']) != sorted(cfg['families']) or not all(report['per_family_passed'].values()):
        raise RuntimeError('Task26 family support incomplete')
    widths = {family: int(report['selection'][family]['width']) for family in cfg['families']}
    hashes = {name:sha(root / cfg[key]) for name,key in (
        ('final_report.json','parent_run'),('selection_freeze.json','parent_freeze'),
        ('process_status.json','parent_process'))}
    if hashes != cfg['parent_result_hashes']:
        raise RuntimeError('Task26 result hashes changed')
    return widths, hashes


def panel_tensors(torch, device):
    gs = [np.ones(5), np.array([1,1,0,0,0]), np.array([0,0,1,1,1]),
          np.array([.25,.25,4,4,4]), np.array([4,4,.25,.25,.25])]
    gs += [np.eye(5)[k] for k in range(5)]
    reversals = [h.REVERSALS, np.array([40,160,25,55,85])]
    return (torch.tensor(np.repeat(gs,2,axis=0),device=device,dtype=torch.float32),
            torch.tensor(np.tile(reversals,(len(gs),1)),device=device,dtype=torch.float32),
            torch.tensor(h.POWERS,device=device,dtype=torch.float32).reshape(1,5,1))


def current_terms(pred, truth, voltage, panels, torch):
    """Return per-seed individual and total normalized analytic-current loss."""
    conductance, reversals, powers = panels
    opening = pred[...,0].pow(powers)*pred[...,1]
    target = truth[...,0].pow(powers)*truth[...,1]
    error = opening-target
    individual = error.square().mean((1,2))
    drive = (voltage[:,None,None,:]-reversals[None,:,:,None])*conductance[None,:,:,None]
    normalized_total = (error[:,None]*drive).sum(2)/drive.abs().sum(2).clamp_min(1e-12)
    total = normalized_total.square().mean((1,2))
    return individual, total


def arm_loss(model, arm, batch, inf, tau, target, cfg, panels, torch):
    pred, ip, tp = model(batch)
    base = ((pred-target).square().mean((2,3))
            + cfg['inf_weight']*(ip-inf).square().mean((2,3))
            + cfg['log_tau_weight']*(tp.log()-tau.log()).square().mean((2,3))).mean(1)
    individual, total = current_terms(pred,target,batch[...,0],panels,torch)
    if arm in ('individual','both'):
        base = base + cfg['individual_current_weight']*individual
    if arm in ('total','both'):
        base = base + cfg['total_current_weight']*total
    return base


def make_support(cfg):
    ds=cfg['data_seeds']
    fit=h.data(ds['fit'],cfg['pool_size'],cfg['dt_values_ms'])
    rng=np.random.default_rng(ds['fit_regions']);n=len(fit)//4
    fit[:n,0]=rng.uniform(-105,-65,n);fit[n:2*n,0]=rng.uniform(-75,-25,n)
    devcfg={**cfg,'extrema_nodes':61,'extrema_node_offset':.5}
    dev=dict(in_support=h.data(ds['development'][0],cfg['development_count'],cfg['dt_values_ms']),
             activation_boundary=h.data(ds['development'][1],cfg['development_count'],cfg['dt_values_ms'],(-105,5)),
             state_extrema=h.extrema(devcfg))
    hv=np.repeat(np.random.default_rng(ds['held_development']).uniform(-135,75,64),2)
    held=np.c_[hv,np.tile([[0.]*10,[1.]*10],(64,1)),np.ones(128)]
    return fit,dev,held


def run(root, output, cfg, revision):
    import torch
    from .gpu_baseline_runtime import environment_manifest
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['require_cuda'] and device!='cuda': raise RuntimeError('Task27 requires CUDA')
    if not json.loads((output/'native_audit.json').read_text())['valid']:
        raise RuntimeError('Task27 native oracle invalid')
    widths, prerequisite=verify_parent(root,cfg)
    seeds=cfg['seeds'];fit,dev,held=make_support(cfg)
    np.savez_compressed(output/'fit_development.npz',fit=fit,held=held,**dev)
    ti,tt,ty=h.targets(fit)
    x,ti,tt,ty=[torch.tensor(value,device=device,dtype=torch.float32) for value in (fit,ti,tt,ty)]
    panels=panel_tensors(torch,device)
    models={(family,arm):j.model_factory(torch,widths[family],family,seeds).to(device)
            for family in cfg['families'] for arm in cfg['supervision_arms']}
    paired_initial={family:max(float((a-b).detach().abs().max()) for arm in cfg['supervision_arms'][1:]
                 for a,b in zip(models[family,'none'].parameters(),models[family,arm].parameters()))
                 for family in cfg['families']}
    if any(v!=0 for v in paired_initial.values()):raise RuntimeError('Paired model initialization diverged')
    eq=[j.equivalence(torch,widths[family],family,seeds,device) for family in cfg['families']]
    write(output/'equivalence_preflight.json',dict(valid=all(e['valid'] for e in eq),rows=eq,paired_initial=paired_initial))
    if not all(e['valid'] for e in eq):raise RuntimeError('Task27 vectorized Adam equivalence failed')
    optimizer=torch.optim.Adam([p for model in models.values() for p in model.parameters()],lr=cfg['learning_rate'],foreach=True)
    rng=np.random.default_rng(cfg['data_seeds']['minibatches']);start=time.monotonic()
    ladder=[];best={};norms={k:torch.zeros(len(seeds),device=device) for k in models}
    hits={k:torch.zeros(len(seeds),device=device) for k in models}
    for step in range(cfg['checkpoints'][-1]+1):
        if step in cfg['checkpoints']:
            for (family,arm),model in models.items():
                mm={domain:j.evaluate(model,vals,seeds,torch,device) for domain,vals in dev.items()}
                hold={}
                for horizon in (1000,10000):
                    vals=held.copy();vals[:,-1]=horizon
                    hold[str(horizon)]=j.evaluate(model,vals,seeds,torch,device)
                cp=output/f'checkpoint_{family}_{arm}_step{step}.pt'
                torch.save(model.state_dict(),cp)
                scores=[]
                for s,seed in enumerate(seeds):
                    domains={d:rows[s] for d,rows in mm.items()};hs={d:rows[s] for d,rows in hold.items()}
                    sc=max([j.score(row,cfg['gates'],cfg['current_gates']) for row in domains.values()]
                           +[j.score(row,cfg['held_gates'],cfg['path_current_gates']) for row in hs.values()])
                    ladder.append(dict(family=family,arm=arm,seed=seed,step=step,score=sc,domains=domains,composed_hold=hs))
                    scores.append(sc)
                row=dict(family=family,arm=arm,width=widths[family],step=step,score=max(scores),checkpoint=cp.name)
                if (family,arm) not in best or (row['score'],step)<(best[family,arm]['score'],best[family,arm]['step']):best[family,arm]=row
            write(output/'development_ladder.json',ladder)
            print(f'[GIADA Task27] checkpoint {step}/{cfg["checkpoints"][-1]} elapsed={(time.monotonic()-start)/60:.1f}min',flush=True)
        if step==cfg['checkpoints'][-1]:break
        fraction=max(0.,min(1.,(step-cfg['decay_start'])/(cfg['checkpoints'][-1]-cfg['decay_start'])))
        optimizer.param_groups[0]['lr']=cfg['learning_rate_final']+.5*(cfg['learning_rate']-cfg['learning_rate_final'])*(1+math.cos(math.pi*fraction))
        idx=torch.tensor(rng.integers(0,len(fit),cfg['batch_size']),device=device)
        batch=x[idx][None].expand(len(seeds),-1,-1)
        optimizer.zero_grad(set_to_none=True)
        loss=sum(arm_loss(model,arm,batch,ti[None,:,idx],tt[None,:,idx],ty[None,:,idx],cfg,panels,torch).sum()
                 for (family,arm),model in models.items())
        loss.backward()
        for key,model in models.items():
            grad=_clip_per_seed(model,torch,cfg['clip_norm'])
            norms[key]+=grad.detach();hits[key]+=(grad>cfg['clip_norm']).detach()
        optimizer.step()
    write(output/'clipping_probes.json',[dict(family=f,arm=a,seed_order=seeds,
        mean_norm=(norms[f,a]/cfg['checkpoints'][-1]).cpu().tolist(),
        clipping_fraction=(hits[f,a]/cfg['checkpoints'][-1]).cpu().tolist()) for f,a in models])
    selection={f:{a:best[f,a] for a in cfg['supervision_arms']} for f in cfg['families']}
    freeze=dict(config=cfg,selection=selection,code_revision=revision,prerequisite_hashes=prerequisite,
                fresh_accessed=False,checkpoint_hashes={p.name:sha(p) for p in output.glob('checkpoint_*.pt')})
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    write(output/'selection_freeze.json',freeze)
    fresh=dict(in_support=h.data(cfg['fresh_seeds'][0],cfg['sample_count'],cfg['dt_values_ms']),
               activation_boundary=h.data(cfg['fresh_seeds'][1],cfg['sample_count'],cfg['dt_values_ms'],(-105,5)),
               state_extrema=h.extrema(cfg))
    np.savez_compressed(output/'fresh.npz',**fresh)
    results=[]
    for (family,arm),model in models.items():
        chosen=best[family,arm];cp=output/chosen['checkpoint']
        if sha(cp)!=freeze['checkpoint_hashes'][cp.name]:raise RuntimeError('Task27 selected checkpoint hash mismatch')
        model.load_state_dict(torch.load(cp,map_location=device,weights_only=True));model.eval()
        mm={d:j.evaluate(model,v,seeds,torch,device) for d,v in fresh.items()}
        pair=j.as_pair(model,torch)
        held_roll=h.rollout(pair,seeds,torch,device,1,True,cfg['held_horizons_ms'],cfg['fresh_seeds'][3])
        paths={label:h.rollout(pair,seeds,torch,device,dt,False,[],0) for label,dt in cfg['path_dt_ms'].items()}
        for s,seed in enumerate(seeds):
            domains={d:rows[s] for d,rows in mm.items()};pp={d:rows[s] for d,rows in paths.items()}
            passed=(all(h.passes(row,cfg['gates'],cfg['current_gates']) for row in domains.values())
                    and all(h.passes(row,cfg['held_gates'],cfg['path_current_gates']) for row in held_roll[s].values())
                    and all(h.passes(row,cfg['path_gates'],cfg['path_current_gates']) for path in pp.values() for row in path.values()))
            results.append(dict(family=family,arm=arm,seed=seed,width=widths[family],step=chosen['step'],
                                parameters=sum(p.numel() for p in model.parameters())//len(seeds),
                                metrics=domains,held_rollout=held_roll[s],path_rollout=pp,passed=passed))
        print(f'[GIADA Task27] frozen evaluation {family}/{arm}',flush=True)
    contrasts=[]
    for family in cfg['families']:
        for seed in seeds:
            rows={a:next(r for r in results if r['family']==family and r['arm']==a and r['seed']==seed) for a in cfg['supervision_arms']}
            for domain in fresh:
                for treatment in cfg['supervision_arms'][1:]:
                    baseline=rows['none']['metrics'][domain]['currents']
                    candidate=rows[treatment]['metrics'][domain]['currents']
                    for metric in cfg['current_gates']:
                        contrasts.append(dict(family=family,seed=seed,domain=domain,treatment=treatment,metric=metric,
                            baseline=baseline[metric],candidate=candidate[metric],
                            improvement_fraction=(baseline[metric]-candidate[metric])/max(baseline[metric],1e-12)))
    write(output/'fresh_paired_contrasts.json',contrasts)
    passing={family:{arm:all(r['passed'] for r in results if r['family']==family and r['arm']==arm)
                     for arm in cfg['supervision_arms']} for family in cfg['families']}
    primary={}
    for family in cfg['families']:
        paired=[]
        for seed in seeds:
            by_arm={arm:next(r for r in results if r['family']==family and r['seed']==seed and r['arm']==arm)
                    for arm in ('none','both')}
            worst={arm:max(row['metrics'][domain]['currents'][metric]/limit
                           for domain in fresh for metric,limit in cfg['current_gates'].items())
                   for arm,row in by_arm.items()}
            paired.append((worst['none']-worst['both'])/max(worst['none'],1e-12))
        primary[family]=float(np.median(paired))
    improvement={family:passing[family]['both'] and primary[family]>=.10 for family in cfg['families']}
    report=dict(schema_version='giada-task27-v1',valid=True,per_arm_passed=passing,
                task28_preparation_authorized=all(passing[f]['none'] and passing[f]['both'] for f in cfg['families']),
                current_supervision_material_benefit=improvement,primary_median_worst_current_gain=primary,
                gate_d_completed=False,selection=selection,learned=results,fresh_paired_contrasts=contrasts,
                fresh_used_for_selection=False,prerequisite_hashes=prerequisite,training_and_evaluation_seconds=time.monotonic()-start,
                environment=environment_manifest(torch),scope=cfg['limits'])
    write(output/'final_report.json',report)
    return report
