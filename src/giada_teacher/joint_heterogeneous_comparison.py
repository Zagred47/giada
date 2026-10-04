"""Original Task26: learnability of independent vs shared heterogeneous rates."""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from . import heterogeneous_mechanism_composition as h
from .hh_family_transfer import write, sha
from .joint_gate_symmetric_confirmation import _clip_per_seed


def config(root):
    amendment = root / 'experiments/task26_joint_vs_independent.json'
    patch = json.loads(amendment.read_text())
    base = root / patch['parent_contract']
    inherited = json.loads(base.read_text())
    keys = ('channels','sample_count','extrema_nodes','dt_values_ms','held_horizons_ms','path_dt_ms',
            'gates','held_gates','current_gates','path_gates','path_current_gates','current_semantics','extrema_support')
    return {**{key:inherited[key] for key in keys}, **patch,
            'parent_contract_sha256': sha(base), 'task26_contract_sha256': sha(amendment)}


def model_factory(torch, width, family, seeds):
    if family not in ('independent', 'shared_heads'):
        raise ValueError(family)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n, independent = len(seeds), family == 'independent'
            shapes = {'w1': (n,5,width,1) if independent else (n,width,1),
                      'b1': (n,5,width) if independent else (n,width),
                      'w2': (n,5,width,width) if independent else (n,width,width),
                      'b2': (n,5,width) if independent else (n,width),
                      'wo': (n,5,4,width), 'bo': (n,5,4)}
            for name, shape in shapes.items():
                value = torch.zeros(shape)
                if name.startswith('w'):
                    for s, seed in enumerate(seeds):
                        gen = torch.Generator().manual_seed(seed + {'w1':0,'w2':100,'wo':200}[name])
                        sample_shape = shape[2:] if independent or name == 'wo' else shape[1:]
                        sample = torch.randn(sample_shape, generator=gen) * .1
                        value[s] = sample[None].expand_as(value[s]) if independent or name == 'wo' else sample
                setattr(self, name, torch.nn.Parameter(value))

        def forward(self, x):
            n, b, _ = x.shape
            v = ((x[...,0] + 30) / 100).unsqueeze(-1)
            if family == 'independent':
                f = v[:,None].expand(-1,5,-1,-1)
                f = torch.nn.functional.silu(torch.einsum('scbi,scwi->scbw',f,self.w1)+self.b1[:,:,None])
                f = torch.nn.functional.silu(torch.einsum('scbi,scwi->scbw',f,self.w2)+self.b2[:,:,None])
                raw = torch.einsum('scbw,scow->scbo',f,self.wo)+self.bo[:,:,None]
            else:
                f = torch.nn.functional.silu(torch.bmm(v,self.w1.transpose(1,2))+self.b1[:,None])
                f = torch.nn.functional.silu(torch.bmm(f,self.w2.transpose(1,2))+self.b2[:,None])
                raw = torch.einsum('sbw,scow->scbo',f,self.wo)+self.bo[:,:,None]
            inf = torch.sigmoid(raw[...,:2])
            tau = torch.exp(raw[...,2:].clamp(-12,12))
            state = x[...,1:11].reshape(n,b,5,2).transpose(1,2)
            z = -torch.expm1(-x[:,None,:,-1,None]/tau)
            return (1-z)*state+z*inf, inf, tau
    return Model()


def as_pair(model, torch):
    """Reuse audited Task25 evaluation without changing its immutable sources.

    Rates depend on V only. Evaluation adapters fill irrelevant other-channel
    states with zero, returning only the requested channels. They make two rate
    forwards, so timing ALWAYS measures the direct full-model forward instead.
    """
    class Adapter:
        def __init__(self, start, count): self.start, self.count = start, count
        def __call__(self, x):
            full = torch.zeros((*x.shape[:-1],12), dtype=x.dtype, device=x.device)
            full[...,0], full[...,-1] = x[...,0], x[...,-1]
            a, b = 1+2*self.start, 1+2*(self.start+self.count)
            full[...,a:b] = x[...,1:-1]
            return tuple(value[:,self.start:self.start+self.count] for value in model(full))
    return Adapter(0,2), Adapter(2,3)


def evaluate(model, values, seeds, torch, device):
    return [h.measurements(values,*row) for row in h.predict(as_pair(model,torch),values,seeds,torch,device)]


def score(metrics, gates, current_gates):
    if not all(metrics[c]['finite'] and metrics[c]['occupancy_violations'] == 0 for c in h.CHANNELS):
        raise RuntimeError('Nonfinite or nonphysical development state')
    return max([metrics[c][key]/limit for c in h.CHANNELS for key,limit in gates.items() if limit>0]
               + [metrics['currents'][key]/limit for key,limit in current_gates.items() if limit>0])


def losses(model, batch, ti, tt, ty, cfg, torch):
    p, i, t = model(batch)
    return ((p-ty).square().mean((2,3)) + cfg['inf_weight']*(i-ti).square().mean((2,3))
            + cfg['log_tau_weight']*(t.log()-tt.log()).square().mean((2,3)))


def equivalence(torch, width, family, seeds, device):
    model = model_factory(torch,width,family,seeds).to(device)
    x = torch.tensor(h.data(269901,24,[.025,1,1000]),device=device,dtype=torch.float32)[None].expand(len(seeds),-1,-1)
    before = model(x)[0].detach()
    opt = torch.optim.Adam(model.parameters(),lr=.003,foreach=True)
    weights = torch.arange(1,len(seeds)+1,device=device)
    model(x)[0].square().mean((1,2,3)).mul(weights).sum().backward()
    _clip_per_seed(model,torch,1.)
    opt.step()
    prediction_error, adam_error = 0., 0.
    for s, seed in enumerate(seeds):
        single = model_factory(torch,width,family,[seed]).to(device)
        prediction_error = max(prediction_error,float((single(x[s:s+1])[0][0]-before[s]).detach().abs().max()))
        one = torch.optim.Adam(single.parameters(),lr=.003,foreach=True)
        single(x[s:s+1])[0].square().mean().mul(weights[s]).backward()
        _clip_per_seed(single,torch,1.)
        one.step()
        adam_error = max(adam_error,max(float((p[s:s+1]-single.state_dict()[key]).detach().abs().max()) for key,p in model.named_parameters()))
    return dict(family=family,width=width,prediction_error=prediction_error,adam_error=adam_error,
                valid=max(prediction_error,adam_error)<1e-5,independent_adam_moments=True,independent_seed_clipping=True)


def gradient_probe(model, x, ti, tt, ty, cfg, torch):
    """Five channel losses projected onto the same shared trunk parameters."""
    params = [getattr(model,key) for key in ('w1','b1','w2','b2')]
    task = losses(model,x,ti,tt,ty,cfg,torch)
    gradients = []
    for channel in range(5):
        g = torch.autograd.grad(task[:,channel].sum(),params,retain_graph=channel<4)
        gradients.append(torch.cat([v.flatten(1) for v in g],1))
    g = torch.stack(gradients,1)
    norm = g.norm(dim=-1)
    cosine = torch.bmm(g,g.transpose(1,2))/((norm[:,:,None]*norm[:,None,:]).clamp_min(1e-30))
    return dict(channel_order=h.CHANNELS,norms=norm.detach().cpu().tolist(),cosines=cosine.detach().cpu().tolist(),
                selection_eligible=False,interpretation='Local conflict probe; correlation is not a causal attribution.')


def timing(model, torch, device, cfg):
    if device != 'cuda': return dict(valid=False,reason='CPU preflight, not hardware evidence')
    result = {}
    with torch.no_grad():
        for batch in (1,256,8192):
            x = torch.tensor(h.data(269910,batch,[1.]),device=device,dtype=torch.float32)[None].expand(len(cfg['seeds']),-1,-1)
            for _ in range(20):model(x)
            times = []
            for _ in range(5):
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(100):model(x)
                torch.cuda.synchronize();times.append((time.perf_counter()-start)/100)
            result[str(batch)] = dict(median_seconds=float(np.median(times)),repetitions=100,blocks=5,
                                      includes='full ten-state update, all three vectorized seeds; excludes analytical current')
    return dict(valid=True,batches=result,selection_eligible=False)


def validate_prerequisite(root, cfg):
    folder = root / cfg['parent_result']
    report = json.loads((folder/'final_report.json').read_text())
    audit = json.loads((folder/'result_audit.json').read_text())
    assert report['valid'] and report['composition_passed'] and report['task26_preparation_authorized'] and audit['valid']
    assert all(sha(folder/name) == digest for name,digest in cfg['parent_result_hashes'].items())
    # No checkpoint warmstart or reuse of previous fresh arrays.
    return dict(cfg['parent_result_hashes'])


def run(root, output, cfg, revision):
    import torch
    from .gpu_baseline_runtime import environment_manifest
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['require_cuda'] and device != 'cuda':raise RuntimeError('CUDA required')
    assert json.loads((output/'native_audit.json').read_text())['valid']
    prerequisite = validate_prerequisite(root,cfg)
    seeds, ds = cfg['seeds'], cfg['data_seeds']
    fit = h.data(ds['fit'],cfg['pool_size'],cfg['dt_values_ms'])
    rng = np.random.default_rng(ds['fit_regions']);n=len(fit)//4
    fit[:n,0]=rng.uniform(-105,-65,n);fit[n:2*n,0]=rng.uniform(-75,-25,n)
    devcfg={**cfg,'extrema_nodes':61,'extrema_node_offset':.5}
    dev=dict(uniform=h.data(ds['development'][0],cfg['development_count'],cfg['dt_values_ms']),
             activation_boundary=h.data(ds['development'][1],cfg['development_count'],cfg['dt_values_ms'],(-105,5)),state_extrema=h.extrema(devcfg))
    hv=np.repeat(np.random.default_rng(ds['held_development']).uniform(-135,75,64),2)
    heldvals=np.c_[hv,np.tile([[0.]*10,[1.]*10],(64,1)),np.ones(128)]
    np.savez_compressed(output/'fit_development.npz',fit=fit,**dev,held_development=heldvals)
    ti,tt,ty=h.targets(fit)
    x,ti,tt,ty=[torch.tensor(value,device=device,dtype=torch.float32) for value in (fit,ti,tt,ty)]
    models={(family,width):model_factory(torch,width,family,seeds).to(device) for family in cfg['families'] for width in cfg['widths']}
    eq=[equivalence(torch,width,family,seeds,device) for family,width in models]
    write(output/'equivalence_preflight.json',dict(valid=all(row['valid'] for row in eq),rows=eq))
    if not all(row['valid'] for row in eq):raise RuntimeError('Vectorization equivalence failed')
    # One foreach optimizer dispatch, independent moment tensors for all parameters.
    optimizer=torch.optim.Adam([p for model in models.values() for p in model.parameters()],lr=cfg['learning_rate'],foreach=True)
    ladder, probes, best = [], [], {}
    gradient_norms={key:torch.zeros(len(seeds),device=device) for key in models}
    clipping_hits={key:torch.zeros(len(seeds),device=device) for key in models}
    rng=np.random.default_rng(ds['minibatches']);start=time.monotonic()
    for step in range(cfg['checkpoints'][-1]+1):
        if step in cfg['checkpoints']:
            for (family,width),model in models.items():
                mm={domain:evaluate(model,vals,seeds,torch,device) for domain,vals in dev.items()}
                held={}
                for horizon in (1000,10000):
                    vals=heldvals.copy();vals[:,-1]=horizon
                    held[str(horizon)]=evaluate(model,vals,seeds,torch,device)
                cp=output/f'checkpoint_{family}_w{width}_step{step}.pt';torch.save(model.state_dict(),cp)
                scores=[]
                for s,seed in enumerate(seeds):
                    domains={domain:rows[s] for domain,rows in mm.items()};hh={horizon:rows[s] for horizon,rows in held.items()}
                    sc=max([score(row,cfg['gates'],cfg['current_gates']) for row in domains.values()]
                           +[score(row,cfg['held_gates'],cfg['path_current_gates']) for row in hh.values()])
                    scores.append(sc);ladder.append(dict(family=family,width=width,step=step,seed=seed,metrics=domains,composed_hold=hh,score=sc))
                candidate=dict(family=family,width=width,step=step,score=max(scores),checkpoint=cp.name)
                if family not in best or (candidate['score'],width,step)<(best[family]['score'],best[family]['width'],best[family]['step']):best[family]=candidate
                if family=='shared_heads':
                    idx=torch.arange(min(256,len(fit)),device=device)
                    probe=gradient_probe(model,x[idx][None].expand(len(seeds),-1,-1),ti[None,:,idx],tt[None,:,idx],ty[None,:,idx],cfg,torch)
                    probes.append(dict(width=width,step=step,seed_order=seeds,**probe))
            write(output/'development_ladder.json',ladder);write(output/'gradient_probes.json',probes)
            print(f'[GIADA Task26] checkpoint {step}/{cfg["checkpoints"][-1]} elapsed={(time.monotonic()-start)/60:.1f}min',flush=True)
        if step==cfg['checkpoints'][-1]:break
        fraction=max(0.,min(1.,(step-cfg['decay_start'])/(cfg['checkpoints'][-1]-cfg['decay_start'])))
        optimizer.param_groups[0]['lr']=cfg['learning_rate_final']+.5*(cfg['learning_rate']-cfg['learning_rate_final'])*(1+math.cos(math.pi*fraction))
        idx=torch.tensor(rng.integers(0,len(fit),cfg['batch_size']),device=device)
        batch=x[idx][None].expand(len(seeds),-1,-1);optimizer.zero_grad(set_to_none=True)
        loss=sum(losses(model,batch,ti[None,:,idx],tt[None,:,idx],ty[None,:,idx],cfg,torch).mean(1).sum() for model in models.values())
        loss.backward()
        for key,model in models.items():
            norms=_clip_per_seed(model,torch,cfg['clip_norm'])
            gradient_norms[key]+=norms.detach();clipping_hits[key]+=(norms>cfg['clip_norm']).detach()
        optimizer.step()
    write(output/'clipping_probes.json',[dict(family=family,width=width,seed_order=seeds,
        mean_bundle_norm=(gradient_norms[family,width]/cfg['checkpoints'][-1]).cpu().tolist(),
        clipping_fraction=(clipping_hits[family,width]/cfg['checkpoints'][-1]).cpu().tolist()) for family,width in models])
    freeze=dict(config=cfg,selection=best,code_revision=revision,fresh_accessed=False,prerequisite_hashes=prerequisite,
                checkpoint_hashes={p.name:sha(p) for p in output.glob('checkpoint_*.pt')})
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    write(output/'selection_freeze.json',freeze)
    fresh=dict(in_support=h.data(cfg['fresh_seeds'][0],cfg['sample_count'],cfg['dt_values_ms']),
               activation_boundary=h.data(cfg['fresh_seeds'][1],cfg['sample_count'],cfg['dt_values_ms'],(-105,5)),state_extrema=h.extrema(cfg))
    np.savez_compressed(output/'fresh.npz',**fresh)
    results, timings = [], {}
    for family,selected in best.items():
        model=models[family,selected['width']];cp=output/selected['checkpoint'];assert sha(cp)==freeze['checkpoint_hashes'][cp.name]
        model.load_state_dict(torch.load(cp,map_location=device,weights_only=True));model.eval()
        pair=as_pair(model,torch)
        mm={domain:evaluate(model,vals,seeds,torch,device) for domain,vals in fresh.items()}
        held=h.rollout(pair,seeds,torch,device,1,True,cfg['held_horizons_ms'],cfg['fresh_seeds'][3])
        held64=h.rollout(pair,seeds,torch,device,1,True,cfg['held_horizons_ms'],cfg['fresh_seeds'][3],'float64')
        paths={label:h.rollout(pair,seeds,torch,device,dt,False,[],0) for label,dt in cfg['path_dt_ms'].items()}
        wrong=[]
        vals=fresh['in_support'];state=vals[:,1:11].reshape(-1,5,2).transpose(1,0,2)
        for p,i,t in h.predict(pair,vals,seeds,torch,device):
            order=[1,2,3,4,0];wrong.append(h.measurements(vals,h.update(state,i[order],t[order],vals[:,-1]),i[order],t[order]))
        for s,seed in enumerate(seeds):
            domains={domain:rows[s] for domain,rows in mm.items()};pp={label:rows[s] for label,rows in paths.items()}
            passed=(all(h.passes(row,cfg['gates'],cfg['current_gates']) for row in domains.values())
                    and all(h.passes(row,cfg['held_gates'],cfg['path_current_gates']) for row in held[s].values())
                    and all(h.passes(row,cfg['path_gates'],cfg['path_current_gates']) for path in pp.values() for row in path.values()))
            results.append(dict(arm=family,seed=seed,width=selected['width'],step=selected['step'],parameter_count=sum(p.numel() for p in model.parameters())//len(seeds),
                                metrics=domains,held_rollout=held[s],held_fp64_diagnostic=held64[s],path_rollout=pp,swapped_identity=wrong[s],passed=passed))
        timings[family]=timing(model,torch,device,cfg)
        print('[GIADA Task26] frozen evaluation '+family,flush=True)
    indexed={(r['family'],r['width'],r['step'],r['seed']):r['score'] for r in ladder};contrasts=[]
    for width in cfg['widths']:
        for step in cfg['checkpoints'][1:]:
            for seed in seeds:
                a,b=[indexed[f,width,step,seed] for f in cfg['families']]
                contrasts.append(dict(factor='sharing',width=width,step=step,seed=seed,improvement_fraction=(a-b)/max(a,1e-12)))
    for family in cfg['families']:
        for seed in seeds:
            if 16 in cfg['widths'] and 32 in cfg['widths']:
                for step in cfg['checkpoints'][1:]:
                    a,b=indexed[family,16,step,seed],indexed[family,32,step,seed]
                    contrasts.append(dict(factor='capacity',family=family,seed=seed,step=step,improvement_fraction=(a-b)/max(a,1e-12)))
            for width in cfg['widths']:
                if 30000 in cfg['checkpoints'] and 60000 in cfg['checkpoints']:
                    a,b=indexed[family,width,30000,seed],indexed[family,width,60000,seed]
                    contrasts.append(dict(factor='budget',family=family,seed=seed,width=width,improvement_fraction=(a-b)/max(a,1e-12)))
    if 16 in cfg['widths'] and 32 in cfg['widths']:
        for seed in seeds:
            for step in cfg['checkpoints'][1:]:
                a,b=indexed['independent',16,step,seed],indexed['shared_heads',32,step,seed]
                contrasts.append(dict(factor='approx_parameter_matched',seed=seed,step=step,
                    reference='independent_w16',treatment='shared_heads_w32',reference_parameters=1860,treatment_parameters=1780,
                    improvement_fraction=(a-b)/max(a,1e-12),exact_parameter_match=False))
    write(output/'paired_contrasts.json',contrasts);write(output/'timing_probes.json',timings)
    costs=[dict(family=family,width=width,parameters=sum(p.numel() for p in model.parameters())//len(seeds),
                rate_mlp_macs_per_voltage=5*(width*width+5*width) if family=='independent' else width*width+21*width,
                excludes='Nonlinearities, gate update, current algebra, memory traffic; not measured speedup') for (family,width),model in models.items()]
    write(output/'cost_probes.json',costs)
    passing={family:all(row['passed'] for row in results if row['arm']==family) for family in cfg['families']}
    counts={family:next(row['parameter_count'] for row in results if row['arm']==family) for family in cfg['families']}
    fresh_contrasts=[]
    for seed in seeds:
        a=next(row for row in results if row['arm']=='independent' and row['seed']==seed)
        b=next(row for row in results if row['arm']=='shared_heads' and row['seed']==seed)
        for domain in fresh:
            for channel in h.CHANNELS:
                fresh_contrasts.append(dict(seed=seed,domain=domain,channel=channel,
                    independent_gate_rmse=a['metrics'][domain][channel]['gate_rmse'],shared_gate_rmse=b['metrics'][domain][channel]['gate_rmse'],
                    shared_over_independent=b['metrics'][domain][channel]['gate_rmse']/max(a['metrics'][domain][channel]['gate_rmse'],1e-12),selection_eligible=False))
    write(output/'fresh_paired_contrasts.json',fresh_contrasts)
    report=dict(schema_version='giada-task26-v1',valid=True,per_family_passed=passing,comparison_passed=all(passing.values()),
                task27_preparation_authorized=all(passing.values()),compression_valid=all(passing.values()) and counts['shared_heads']<counts['independent'],
                gate_d_completed=False,selection=best,learned=results,fresh_used_for_selection=False,prerequisite_hashes=prerequisite,
                paired_contrasts=contrasts,fresh_paired_contrasts=fresh_contrasts,selected_parameter_counts=counts,timing=timings,training_and_evaluation_seconds=time.monotonic()-start,
                environment=environment_manifest(torch),scope=cfg['limits'],current_supervision_used=False)
    write(output/'final_report.json',report)
    return report
