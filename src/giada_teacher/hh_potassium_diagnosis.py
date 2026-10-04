"""Task18b: preregistered K_Pst tail x tau-supervision x capacity diagnosis."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from .hh_family_transfer import data,rates,update,model_factory,measurements,passes,write,sha

def pools(count):
    uniform=data(181101,count)
    tail=uniform.copy()
    tail[:count//4,0]=np.random.default_rng(181102).uniform(-135,-115,count//4)
    return {'uniform':uniform,'tail_enriched':tail}

def descriptors(seeds):
    return [('K_Pst',f'{sampling}/tau{weight}',seed) for sampling in ('uniform','tail_enriched') for weight in (.01,.1) for seed in seeds]

def metric_rows(model,values,ds,torch,device):
    x=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(len(ds),-1,-1)
    with torch.no_grad():result=[a.cpu().double().numpy() for a in model(x)]
    return [measurements(values,*(a[i] for a in result),'K_Pst') for i in range(len(ds))]

def score(metrics):
    return max(m['gate_rmse']/.001+m['gate_max_error']/.01+m['inf_rmse']/.002+m['log_tau_rmse']/.02 for m in metrics.values())

def run(output,config,revision):
    from .gpu_baseline_runtime import configure_torch_runtime,environment_manifest
    from .joint_gate_symmetric_confirmation import _clip_per_seed
    torch=configure_torch_runtime(17);device='cuda' if torch.cuda.is_available() else 'cpu'
    if config.get('require_cuda',True) and device!='cuda':raise RuntimeError('CUDA required')
    output=Path(output);ds=descriptors(config['seeds']);pool=pools(config['pool_size'])
    if config.get('require_cuda',True):
        audit=json.loads((output/'native_audit.json').read_text())
        if not audit['valid']:raise RuntimeError('native audit is not valid')
    dev={'uniform':data(181201,2048),'negative_tail':data(181202,2048,(-135,-115))}
    np.savez_compressed(output/'fit_development.npz',**{f'fit_{k}':v for k,v in pool.items()},**{f'dev_{k}':v for k,v in dev.items()})
    x=np.stack([pool[o.split('/')[0]] for _,o,_ in ds])
    ti,tt=rates('K_Pst',x[...,0]);ty=update(x[...,1:3],ti,tt,x[...,3])
    x,ti,tt,ty=[torch.tensor(a,dtype=torch.float32,device=device) for a in (x,ti,tt,ty)]
    tw=torch.tensor([float(o.split('tau')[1]) for _,o,_ in ds],device=device)
    models={w:model_factory(torch,w,ds).to(device) for w in config['widths']}
    optim={w:torch.optim.Adam(m.parameters(),lr=.003) for w,m in models.items()}
    eq=[]
    for w,m in models.items():
        a=model_factory(torch,w,ds).to(device);a.load_state_dict(m.state_dict())
        b=model_factory(torch,w,[ds[0]]).to(device);b.load_state_dict({k:v[:1].clone() for k,v in m.state_dict().items()})
        xx=x[:,:32];pa=a(xx)[0];pb=b(xx[:1])[0]
        err=float((pa[:1]-pb).detach().abs().max())
        aa=torch.optim.Adam(a.parameters(),lr=.003);ab=torch.optim.Adam(b.parameters(),lr=.003)
        pa.square().mean((1,2)).sum().backward();pb.square().mean().backward()
        _clip_per_seed(a,torch,1);_clip_per_seed(b,torch,1);aa.step();ab.step()
        pe=max(float((v[:1]-b.state_dict()[k]).abs().max()) for k,v in a.state_dict().items())
        eq.append({'width':w,'prediction_error':err,'adam_error':pe})
    write(output/'equivalence_preflight.json',{'valid':all(max(r['prediction_error'],r['adam_error'])<1e-5 for r in eq),'rows':eq})
    if not all(max(r['prediction_error'],r['adam_error'])<1e-5 for r in eq):raise RuntimeError('vectorization mismatch')
    ladder=[];best=None;rng=np.random.default_rng(181301);start=time.monotonic()
    for step in range(config['checkpoints'][-1]+1):
        if step in config['checkpoints']:
            for w,m in models.items():
                dm={k:metric_rows(m,v,ds,torch,device) for k,v in dev.items()}
                torch.save(m.state_dict(),output/f'checkpoint_w{w}_step{step}.pt')
                for n,(_,o,s) in enumerate(ds):
                    mm={k:v[n] for k,v in dm.items()}
                    ladder.append({'width':w,'step':step,'arm':o,'seed':s,'metrics':mm,'score':score(mm)})
                for arm in dict.fromkeys(o for _,o,_ in ds):
                    indices=[n for n,(_,o,_) in enumerate(ds) if o==arm]
                    candidate={'width':w,'step':step,'arm':arm,'indices':indices,'max_seed_score':max(score({k:v[n] for k,v in dm.items()}) for n in indices)}
                    if best is None or candidate['max_seed_score']<best['max_seed_score']:best=candidate
            write(output/'development_ladder.json',ladder)
            print(f'[GIADA Task18b] checkpoint {step}/{config["checkpoints"][-1]} elapsed={(time.monotonic()-start)/60:.1f}min',flush=True)
        if step==config['checkpoints'][-1]:break
        idx=torch.tensor(rng.integers(0,config['pool_size'],config['batch_size']),device=device)
        for w,m in models.items():
            optim[w].zero_grad(set_to_none=True);p,i,t=m(x[:,idx])
            loss=(p-ty[:,idx]).square().mean((1,2))+.1*(i-ti[:,idx]).square().mean((1,2))+tw*(t.log()-tt[:,idx].log()).square().mean((1,2))
            loss.sum().backward();_clip_per_seed(m,torch,1);optim[w].step()
    # Fixed-budget contrasts, before fresh access; never compare differently selected seeds as a pure factor effect.
    by={(r['width'],r['step'],r['arm'],r['seed']):r for r in ladder};contrasts=[]
    for r in ladder:
        sampling,weight=r['arm'].split('/tau')
        variants={'sampling':(r['width'],r['step'],'tail_enriched/tau'+weight,r['seed']),
                  'tau_weight':(r['width'],r['step'],sampling+'/tau0.1',r['seed']),
                  'capacity':(config['widths'][-1],r['step'],r['arm'],r['seed'])}
        later=[s for s in config['checkpoints'] if s>r['step']]
        if r['step']>0 and later:variants['budget']=(r['width'],later[0],r['arm'],r['seed'])
        for factor,k in variants.items():
            if (factor=='sampling' and sampling!='uniform') or (factor=='tau_weight' and weight!='0.01') or (factor=='capacity' and r['width']==config['widths'][-1]):continue
            other=by[k]
            contrasts.append({'factor':factor,'seed':r['seed'],'step':r['step'],'treated_step':other['step'],'base_arm':r['arm'],'base_width':r['width'],'treated_arm':other['arm'],'treated_width':other['width'],'score_difference':other['score']-r['score'],'gate_rmse_difference':{d:other['metrics'][d]['gate_rmse']-r['metrics'][d]['gate_rmse'] for d in dev}})
    write(output/'paired_development_contrasts.json',contrasts)
    freeze={'selected_common_configuration':best,'fresh_accessed':False,'code_revision':revision,'config':config,'checkpoint_hashes':{p.name:sha(p) for p in output.glob('checkpoint_*.pt')}}
    freeze['freeze_sha256']=__import__('hashlib').sha256(json.dumps(freeze,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write(output/'selection_freeze.json',freeze)
    fresh={'in_support':data(181401,4096),'negative_tail':data(181402,4096,(-135,-115)),'ood_negative':data(181403,2048,(-155,-135))}
    np.savez_compressed(output/'fresh.npz',**fresh)
    selected=models[best['width']];cp=output/f'checkpoint_w{best["width"]}_step{best["step"]}.pt'
    if sha(cp)!=freeze['checkpoint_hashes'][cp.name]:raise RuntimeError('checkpoint hash mismatch')
    selected.load_state_dict(torch.load(cp,map_location=device,weights_only=True))
    fm={k:metric_rows(selected,v,ds,torch,device) for k,v in fresh.items()}
    results=[]
    for n in best['indices']:
        mm={k:v[n] for k,v in fm.items()}
        results.append({'seed':ds[n][2],'metrics':mm,'passed':all(passes(mm[k]) for k in ('in_support','negative_tail'))})
    # Test boundary and gate/time strata are diagnostics, not new tuning inputs.
    probe=np.c_[np.repeat(np.linspace(-135,-110,251),7),np.full(251*7,.31),np.full(251*7,.79),np.tile([.025,.1,.5,1,5,25,100],251)]
    pm=metric_rows(selected,probe,ds,torch,device)
    write(output/'tail_probe_metrics.json',[{'seed':ds[n][2],'metrics':pm[n]} for n in best['indices']])
    grid=np.linspace(-135,75,2049);ii,tt=rates('K_Pst',grid);numeric={}
    for name in ('in_support','negative_tail'):
        values=fresh[name];inf=np.stack([np.interp(values[:,0],grid,ii[:,g]) for g in range(2)],-1);tau=np.stack([np.interp(values[:,0],grid,tt[:,g]) for g in range(2)],-1)
        numeric[name]=measurements(values,update(values[:,1:3],inf,tau,values[:,3]),inf,tau,'K_Pst')
    passed=all(r['passed'] for r in results)
    report={'schema_version':'giada-task18b-potassium-v1','valid':True,'potassium_passed':passed,'task19_authorized':passed,'fresh_used_for_selection':False,'selected':best,'results':results,'lut_control':numeric,'training_seconds':time.monotonic()-start,'environment':environment_manifest(torch),'scope':'isolated K_Pst rates and held-voltage transitions; Ca/Na prior Task18 confirmation retained; no embedded Na/K substitution'}
    write(output/'final_report.json',report);return report
