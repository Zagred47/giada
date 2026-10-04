"""Task18c paired local-coverage x robust-loss x learning-rate refinement."""
import hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
from .hh_family_transfer import data,rates,update,model_factory,measurements,passes,write,sha
from .hh_potassium_diagnosis import score,metric_rows

def arms(config):
    return [(s,k,lr) for s in config['sampling'] for k in config['topk_weights'] for lr in config['learning_rates']]

def pools(count):
    a=data(182101,count);a[:count//4,0]=np.random.default_rng(182102).uniform(-135,-115,count//4)
    b=a.copy();b[count//4:count//2,0]=np.random.default_rng(182103).uniform(-65,-55,count//4)
    return {'tail_only':a,'tail_and_kink':b}

def initialize(torch,config,source,device):
    if sha(source)!=config['parent_fixture_sha256']:raise RuntimeError('parent fixture SHA mismatch')
    aa=arms(config);ds=[('K_Pst',f'{s}/top{k}/lr{lr}',seed) for s,k,lr in aa for seed in config['seeds']]
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None:raise RuntimeError('parent archive CRC')
        f=json.loads(z.read('selection_freeze.json'));claimed=f.pop('freeze_sha256')
        if hashlib.sha256(json.dumps(f,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=claimed:raise RuntimeError('parent freeze SHA')
        best=f['selected_common_configuration']
        if best['width']!=32 or best['step']!=60000 or best['arm']!='tail_enriched/tau0.1':raise RuntimeError('wrong parent configuration')
        name='checkpoint_w32_step60000.pt';raw=z.read(name)
        if hashlib.sha256(raw).hexdigest()!=f['checkpoint_hashes'][name]:raise RuntimeError('parent checkpoint SHA')
        state=torch.load(io.BytesIO(raw),map_location=device,weights_only=True)
    model=model_factory(torch,32,ds).to(device)
    selected={k:v[best['indices']] for k,v in state.items()}
    model.load_state_dict({k:v.repeat((len(aa),)+(1,)*(v.ndim-1)) for k,v in selected.items()})
    parent=model_factory(torch,32,[('K_Pst','parent',s) for s in config['seeds']]).to(device);parent.load_state_dict(selected)
    return model,parent,ds

def run(output,config,revision,source):
    from .gpu_baseline_runtime import configure_torch_runtime,environment_manifest
    from .joint_gate_symmetric_confirmation import _clip_per_seed
    torch=configure_torch_runtime(17);device='cuda' if torch.cuda.is_available() else 'cpu';output=Path(output)
    if config.get('require_cuda',True):
        if device!='cuda' or not json.loads((output/'native_audit.json').read_text())['valid']:raise RuntimeError('CUDA and native audit required')
    model,parent,ds=initialize(torch,config,source,device);aa=arms(config);pool=pools(config['pool_size'])
    dev={'uniform':data(182201,2048),'negative_tail':data(182202,2048,(-135,-115)),'kink':data(182203,2048,(-65,-55))}
    np.savez_compressed(output/'fit_development.npz',**{f'fit_{k}':v for k,v in pool.items()},**{f'dev_{k}':v for k,v in dev.items()})
    xx=np.stack([pool[s] for s,k,lr in aa for seed in config['seeds']]);ti,tt=rates('K_Pst',xx[...,0]);ty=update(xx[...,1:3],ti,tt,xx[...,3])
    x,ti,tt,ty=[torch.tensor(v,dtype=torch.float32,device=device) for v in (xx,ti,tt,ty)]
    # Adam moments are independent on the leading axis; lr scales its update, not gradients.
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    lrvec=torch.tensor([lr for s,k,lr in aa for seed in config['seeds']],device=device)
    kw=torch.tensor([k for s,k,lr in aa for seed in config['seeds']],device=device)
    single=model_factory(torch,32,[ds[0]]).to(device);single.load_state_dict({k:v[:1].clone() for k,v in model.state_dict().items()})
    clone=model_factory(torch,32,ds).to(device);clone.load_state_dict(model.state_dict())
    oa=torch.optim.Adam(clone.parameters(),lr=.003);ob=torch.optim.Adam(single.parameters(),lr=float(lrvec[0]))
    p=clone(x[:,:32])[0];q=single(x[:1,:32])[0];pe=float((p[:1]-q).detach().abs().max())
    p.square().mean((1,2)).sum().backward();q.square().mean().backward();_clip_per_seed(clone,torch,1);_clip_per_seed(single,torch,1)
    before={k:v.detach().clone() for k,v in clone.named_parameters()};oa.step();ob.step()
    with torch.no_grad():
        for k,v in clone.named_parameters():v.copy_(before[k]+(lrvec/.003).reshape((-1,)+(1,)*(v.ndim-1))*(v-before[k]))
    ae=max(float((v[:1]-single.state_dict()[k]).abs().max()) for k,v in clone.state_dict().items())
    low=next(n for n,(_,a,_) in enumerate(ds) if a.endswith('lr0.0003'))
    single_low=model_factory(torch,32,[ds[low]]).to(device);single_low.load_state_dict({k:v[low:low+1].clone() for k,v in model.state_dict().items()})
    lowopt=torch.optim.Adam(single_low.parameters(),lr=.0003);single_low(x[low:low+1,:32])[0].square().mean().backward();_clip_per_seed(single_low,torch,1);lowopt.step()
    ae=max(ae,max(float((v[low:low+1]-single_low.state_dict()[k]).abs().max()) for k,v in clone.state_dict().items()))
    write(output/'equivalence_preflight.json',{'valid':max(pe,ae)<1e-5,'prediction_error':pe,'adam_error':ae,'independent_moments_and_clipping':True})
    if max(pe,ae)>=1e-5:raise RuntimeError('vectorized Adam mismatch')
    ladder=[];best=None;start=time.monotonic();rng=np.random.default_rng(182301)
    for step in range(config['checkpoints'][-1]+1):
        if step in config['checkpoints']:
            dm={k:metric_rows(model,v,ds,torch,device) for k,v in dev.items()};torch.save(model.state_dict(),output/f'checkpoint_step{step}.pt')
            for n,(_,arm,seed) in enumerate(ds):
                mm={k:v[n] for k,v in dm.items()};ladder.append(dict(step=step,arm=arm,seed=seed,metrics=mm,score=score(mm)))
            for arm in dict.fromkeys(a for _,a,_ in ds):
                indices=[n for n,(_,a,_) in enumerate(ds) if a==arm];candidate=dict(step=step,arm=arm,indices=indices,score=max(score({k:v[n] for k,v in dm.items()}) for n in indices))
                if best is None or candidate['score']<best['score']:best=candidate
            write(output/'development_ladder.json',ladder);print(f'[GIADA Task18c] checkpoint{step}/{config["checkpoints"][-1]} elapsed={(time.monotonic()-start)/60:.1f}min',flush=True)
        if step==config['checkpoints'][-1]:break
        idx=torch.tensor(rng.integers(0,config['pool_size'],config['batch_size']),device=device);opt.zero_grad(set_to_none=True);p,i,t=model(x[:,idx]);e=(p-ty[:,idx]).square()
        loss=e.mean((1,2))+.1*(i-ti[:,idx]).square().mean((1,2))+.1*(t.log()-tt[:,idx].log()).square().mean((1,2))+kw*e.flatten(1).topk(max(1,int(e.shape[1]*e.shape[2]*.05)),dim=1).values.mean(1)
        loss.sum().backward();_clip_per_seed(model,torch,1);before={k:v.detach().clone() for k,v in model.named_parameters()};opt.step()
        with torch.no_grad():
            for k,v in model.named_parameters():v.copy_(before[k]+(lrvec/.003).reshape((-1,)+(1,)*(v.ndim-1))*(v-before[k]))
    by={(r['step'],r['arm'],r['seed']):r for r in ladder};contrasts=[]
    for row in ladder:
        sampling,top,lr=row['arm'].split('/');variants={}
        if sampling=='tail_only':variants['coverage']=(row['step'],'tail_and_kink/'+top+'/'+lr,row['seed'])
        if top=='top0':variants['robust_loss']=(row['step'],sampling+'/top1/'+lr,row['seed'])
        if lr=='lr0.003':variants['learning_rate']=(row['step'],sampling+'/'+top+'/lr0.0003',row['seed'])
        later=[s for s in config['checkpoints'] if s>row['step']]
        if later:variants['budget']=(later[0],row['arm'],row['seed'])
        for factor,key in variants.items():
            other=by[key];contrasts.append(dict(factor=factor,seed=row['seed'],base_step=row['step'],treated_step=other['step'],base_arm=row['arm'],treated_arm=other['arm'],score_difference=other['score']-row['score'],gate_rmse_difference={d:other['metrics'][d]['gate_rmse']-row['metrics'][d]['gate_rmse'] for d in dev},gate_max_difference={d:other['metrics'][d]['gate_max_error']-row['metrics'][d]['gate_max_error'] for d in dev}))
    write(output/'paired_development_contrasts.json',contrasts)
    freeze=dict(selected=best,config=config,code_revision=revision,fresh_accessed=False,parent_archive_sha256=sha(source),checkpoint_hashes={p.name:sha(p) for p in output.glob('checkpoint_*.pt')});freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':')).encode()).hexdigest();write(output/'selection_freeze.json',freeze)
    fresh={'in_support':data(182401,4096),'negative_tail':data(182402,4096,(-135,-115)),'kink':data(182403,4096,(-65,-55)),'ood_negative':data(182404,2048,(-155,-135))};np.savez_compressed(output/'fresh.npz',**fresh)
    cp=output/f'checkpoint_step{best["step"]}.pt'
    if sha(cp)!=freeze['checkpoint_hashes'][cp.name]:raise RuntimeError('checkpoint changed')
    model.load_state_dict(torch.load(cp,map_location=device,weights_only=True));fm={k:metric_rows(model,v,ds,torch,device) for k,v in fresh.items()};pm={k:metric_rows(parent,v,[('K_Pst','parent',s) for s in config['seeds']],torch,device) for k,v in fresh.items()}
    results=[dict(seed=ds[n][2],metrics={k:v[n] for k,v in fm.items()},passed=all(passes(fm[k][n]) for k in ('in_support','negative_tail','kink'))) for n in best['indices']]
    report=dict(schema_version='giada-task18c-v1',valid=True,potassium_passed=all(r['passed'] for r in results),task19_authorized=all(r['passed'] for r in results),fresh_used_for_selection=False,selected=best,results=results,parent_same_fresh=pm,training_seconds=time.monotonic()-start,environment=environment_manifest(torch),scope=config['scope']);write(output/'final_report.json',report);return report
