"""IE-P6 paired repair matrix and directly learned local coefficient maps."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import ie_accuracy_execution as p5

p4=p5.p4
p3=p5.p3
CONFIG='experiments/ie_p6_preregistration_v1.json'
ARMS=('A_uniform','B_targeted','C_temporal','D_combined')
MEMBERS=tuple((a,s) for a in ARMS for s in (17,29,43))

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

class Progress:
    def write(self,value): return sys.__stdout__.write(value.replace('[GIADA IE-P4]','[GIADA IE-P6]'))
    def flush(self): return sys.__stdout__.flush()

def dense(torch,parent,device):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ('a','b','c','d'):
                self.register_parameter(name,torch.nn.Parameter(parent[name][[0,1,2]*4].to(device).clone()))
            self.register_buffer('scale',parent['scale'].to(device).clone())
        def forward(self,v):
            raw=torch.bmm(torch.nn.functional.silu(torch.bmm((v[...,None]+40)/80,self.a)+self.b),self.c)+self.d
            values=torch.sigmoid(raw)
            return values[...,:17],values[...,17:]*self.scale
    return Model()

def interpolate(torch,table,v,spacing):
    position=(v.clamp(-100,60)+100)/spacing
    idx=position.floor().long().clamp(0,table.shape[-2]-2)
    f=(position-idx)[...,None]
    if table.ndim==2: return table[idx]*(1-f)+table[idx+1]*f
    member=torch.arange(table.shape[0],device=v.device).reshape(-1,*([1]*(v.ndim-1)))
    return table[member,idx]*(1-f)+table[member,idx+1]*f

def local(torch,parent,knots,device):
    initial=dense(torch,parent,device)
    v=torch.linspace(-100,60,knots,device=device)
    with torch.no_grad():
        u,z=initial(v[None].expand(12,-1));values=torch.cat((u[:3],z[:3]/initial.scale),-1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.table=torch.nn.Parameter(values.clone())
            self.register_buffer('scale',initial.scale.clone());self.spacing=160/(knots-1)
        def forward(self,v):
            y=interpolate(torch,self.table,v,self.spacing)
            return y[...,:17],y[...,17:]*self.scale
        def project(self):
            with torch.no_grad(): self.table[...,:17].clamp_(0,1);self.table[...,17:].clamp_(1e-7,1)
    return Model()

def local_operator(torch,model,index):
    table=model.table[index].detach().clone();scale=model.scale.detach().clone();spacing=model.spacing
    def op(v):
        y=interpolate(torch,table,v,spacing)
        return y[...,:17],y[...,17:]*scale
    return op

def clip(torch,model):
    count=next(model.parameters()).shape[0]
    norm=torch.stack([p.grad.flatten(1).square().sum(-1) for p in model.parameters()]).sum(0).sqrt()
    for p in model.parameters():p.grad.mul_((1/norm.clamp_min(1)).reshape(count,*([1]*(p.ndim-1))))

def base_loss(torch,u,z,x,ut,zt,scale):
    target=(1-zt)*x+zt*ut
    err=(((1-z)*x+z*u-target)/scale).square()
    return err.mean((-1,-2))+(u-ut).square().mean((-1,-2))+.01*(torch.log(z/scale+1e-8)-torch.log(zt/scale+1e-8)).square().mean((-1,-2))

def rollout(torch,u,z,x):
    values=[]
    for i in range(u.shape[-3]):
        x=(1-z[...,i,:,:])*x+z[...,i,:,:]*u[...,i,:,:];values.append(x)
    return torch.stack(values,-3)

def sequences(torch,count,generator,device):
    start=torch.rand(1,count,device=device,generator=generator)*160-100
    walk=(start+torch.randn(8,count,device=device,generator=generator).cumsum(0)*8).clamp(-100,60)
    jumps=torch.rand(8,count,device=device,generator=generator)*160-100
    v=torch.where(torch.arange(count,device=device)[None,:]<count//2,walk,jumps)
    return v,torch.rand(count,17,device=device,generator=generator)

def temporal_loss(torch,model,v,x,truth=None):
    count=next(model.parameters()).shape[0]
    u,z=model(v.flatten()[None].expand(count,-1));u=u.reshape(count,8,-1,17);z=z.reshape_as(u)
    if truth is None:
        ut,zt=p4.coefficients(torch,v);truth=rollout(torch,ut,zt,x)
    return ((rollout(torch,u,z,x)-truth)/model.scale).square().mean((1,2,3))

def targeting(torch,model,fit):
    grid=torch.linspace(-100,60,2049,device=fit.device)
    with torch.no_grad():
        ut,zt=p4.coefficients(torch,grid);u,z=model(grid[None].expand(12,-1))
        e=torch.maximum((z*u-zt*ut).abs(),((1-z)+z*u-(1-zt)-zt*ut).abs())/model.scale
        scores=e[:3].amax((0,2));bins=[]
        ids=((grid+100)/2.5).long().clamp(0,63)
        for k in range(64):bins.append(scores[ids==k].max())
        bins=torch.stack(bins);bins=bins+.2*bins.mean().clamp_min(1e-12)
        weights=bins[((fit+100)/2.5).long().clamp(0,63)]
    return weights,{'source':'training-only fixed grid; no development/fresh/history','bin_weights':bins.cpu().tolist(),'grid_count':2049}

def dev_score(torch,model,v,x,seq,sx):
    count=next(model.parameters()).shape[0]
    with torch.no_grad():
        u,z=model(v[None].expand(count,-1));ut,zt=p4.coefficients(torch,v)
        diff=(1-z)*x+z*u-((1-zt)*x+zt*ut)
        long=(p5.endpoints(torch,u,z)-p5.endpoints(torch,ut,zt)).abs().amax((1,2,3))
        su,sz=model(seq.flatten()[None].expand(count,-1));su=su.reshape(count,8,-1,17);sz=sz.reshape_as(su)
        tu,tz=p4.coefficients(torch,seq)
        short=(rollout(torch,su,sz,sx)-rollout(torch,tu,tz,sx)).abs().amax((1,2,3))
        rmse=diff.square().mean((1,2)).sqrt();maximum=diff.abs().amax((1,2))
        score=torch.stack((rmse/.001,maximum/.01,long/.03,short/.03)).amax(0)
    return score.tolist(),{'rmse':rmse.tolist(),'max_abs':maximum.tolist(),'long_max':long.tolist(),'short_max':short.tolist()}

def gradient_probe(torch,parent,device):
    model=dense(torch,parent,device);v=torch.linspace(-100,60,257,device=device)
    u,z=model(v[None].expand(12,-1));ut,zt=p4.coefficients(torch,v);x=torch.full_like(ut,.5)
    base=base_loss(torch,u,z,x,ut,zt,model.scale)
    logits=(torch.bmm(torch.nn.functional.silu(torch.bmm(((v[None,:,None]+40)/80).expand(12,-1,-1),model.a)+model.b),model.c)+model.d)[...,:17]
    target=torch.logit(ut.clamp(1e-6,1-1e-6))
    normalized_error=(((1-z)*x+z*u-((1-zt)*x+zt*ut))/model.scale).square()
    components={
        'logit':.01*(logits-target).square().mean((1,2)),
        'constant400':(p5.endpoints(torch,u,z)-p5.endpoints(torch,ut,zt)).square().mean((1,2,3)),
        'worst_gate':.1*normalized_error.mean(1).amax(-1)}
    def grads(loss): return torch.cat([g.flatten(1) for g in torch.autograd.grad(loss.sum(),tuple(model.parameters()),retain_graph=True)],1)[:3]
    gb=grads(base);result={}
    for name,loss in components.items():
        gg=grads(loss)
        result[name]={'cosine_to_base':((gb*gg).sum(1)/(gb.norm(dim=1)*gg.norm(dim=1)).clamp_min(1e-30)).detach().cpu().tolist(),
                      'gradient_norm':gg.norm(dim=1).detach().cpu().tolist(),'base_gradient_norm':gb.norm(dim=1).detach().cpu().tolist()}
    return {'training_only':True,'causal_proof':False,'note':'Exact P5 balanced component definitions on current parent training probes.', 'components':result}

def preflight(torch,parent,device):
    gen=torch.Generator(device=device).manual_seed(81900);v=torch.rand(12,31,device=device,generator=gen)*160-100
    maximum=0.
    for constructor in (lambda:dense(torch,parent,device),lambda:local(torch,parent,81,device)):
        model=constructor();n=next(model.parameters()).shape[0];vv=v[:n]
        initial={k:t.detach().clone() for k,t in model.named_parameters()}
        target=torch.rand(n,31,17,device=device,generator=gen)
        opt=torch.optim.Adam(model.parameters(),lr=.001);u,z=model(vv)
        ((u-target).square().mean((1,2))+z.square().mean((1,2))).sum().backward();clip(torch,model);opt.step()
        if hasattr(model,'project'):model.project()
        for k in range(n):
            iso=constructor()
            with torch.no_grad():
                for name,p in iso.named_parameters():p.copy_(initial[name])
            oi=torch.optim.Adam(iso.parameters(),lr=.001);u,z=iso(vv)
            ((u[k]-target[k]).square().mean()+z[k].square().mean()).backward();clip(torch,iso);oi.step()
            if hasattr(iso,'project'):iso.project()
            for name,p in iso.named_parameters():maximum=max(maximum,float((p[k]-dict(model.named_parameters())[name][k]).abs().max()))
    if maximum>1e-5:raise RuntimeError(f'Batched Adam mismatch {maximum}')
    return {'valid':True,'independent_clipped_adam_max_error':maximum,'atol':1e-5}

def train(torch,parent,spec,output,device='cuda'):
    model=dense(torch,parent,device);locals_={k:local(torch,parent,k,device) for k in (81,161)}
    models={'dense':model,**{f'local{k}':m for k,m in locals_.items()}}
    optim={key:torch.optim.Adam(m.parameters(),lr=spec['learning_rate']) for key,m in models.items()}
    gen=torch.Generator(device=device).manual_seed(spec['data_seed'])
    fit=torch.rand(spec['fit_count'],device=device,generator=gen)*160-100;xf=torch.rand(len(fit),17,device=device,generator=gen)
    dev=torch.rand(spec['development_count'],device=device,generator=gen)*160-100;xd=torch.rand(len(dev),17,device=device,generator=gen)
    ds,dx=sequences(torch,128,gen,device);weights,probe=targeting(torch,model,fit);write(output/'targeting.json',probe)
    uf,zf=p4.coefficients(torch,fit);mask=torch.tensor([0]*6+[1]*6,device=device)
    pool,px=sequences(torch,4096,gen,device)
    with torch.no_grad(): pool_truth=rollout(torch,*p4.coefficients(torch,pool),px)
    best={key:[float('inf')]*next(m.parameters()).shape[0] for key,m in models.items()};saved={key:[None]*len(scores) for key,scores in best.items()};chosen={key:[None]*len(scores) for key,scores in best.items()}
    curves=[];started=time.perf_counter()
    for step in range(spec['training_steps']+1):
        if step in spec['checkpoints']:
            row={'step':step,'elapsed_seconds':time.perf_counter()-started,'models':{}}
            for key,m in models.items():
                scores,metrics=dev_score(torch,m,dev,xd,ds,dx);row['models'][key]={'score':scores,**metrics}
                for k,score in enumerate(scores):
                    if not np.isfinite(score):raise RuntimeError('Nonfinite dev score')
                    if score<best[key][k]:
                        best[key][k]=score;chosen[key][k]=step;saved[key][k]={n:p[k].detach().cpu().clone() for n,p in m.named_parameters()}
            curves.append(row);torch.save({key:m.state_dict() for key,m in models.items()},output/f'checkpoint_{step}.pt');write(output/'training_curves.json',curves)
            print(f'[GIADA IE-P6] training {step}/{spec["training_steps"]} dev median={np.median(row["models"]["dense"]["score"]):.5g}',flush=True)
        if step==spec['training_steps']:break
        ids=torch.randint(len(fit),(spec['batch_size'],),device=device,generator=gen)
        half=spec['batch_size']//2
        ti=torch.cat((ids[:half],torch.multinomial(weights,spec['batch_size']-half,replacement=True,generator=gen)))
        matrix=torch.stack([ti if a in ('B_targeted','D_combined') else ids for a,_ in MEMBERS])
        u,z=model(fit[matrix]);loss=base_loss(torch,u,z,xf[matrix],uf[matrix],zf[matrix],model.scale)
        if step%4==0:
            si=torch.randint(4096,(128,),device=device,generator=gen)
            loss=loss+mask*temporal_loss(torch,model,pool[:,si],px[si],pool_truth[:,si])
        optim['dense'].zero_grad();loss.sum().backward();clip(torch,model);optim['dense'].step()
        for key,m in models.items():
            if key=='dense':continue
            u,z=m(fit[ids][None].expand(3,-1));loss=base_loss(torch,u,z,xf[ids],uf[ids],zf[ids],m.scale)
            optim[key].zero_grad();loss.sum().backward();clip(torch,m);optim[key].step();m.project()
    for key,m in models.items():
        with torch.no_grad():
            for k,state in enumerate(saved[key]):
                for name,p in m.named_parameters():p[k].copy_(state[name])
    torch.save({key:m.state_dict() for key,m in models.items()},output/'frozen_checkpoint.pt')
    freeze={'members':MEMBERS,'selected_steps':chosen,'development_scores':best,'fresh_accessed':False,'checkpoint_sha256':hashlib.sha256((output/'frozen_checkpoint.pt').read_bytes()).hexdigest()}
    write(output/'selection_freeze.json',freeze)
    return models,freeze

def path_data(seed):
    rng=np.random.default_rng(seed);t=np.arange(400)*.1;paths=[];families=[]
    for f in ('constant','steps','sinusoid','chirp','pulse_recovery','random_walk'):
        for _ in range(4):
            if f=='constant':v=np.full(400,rng.uniform(-100,60))
            elif f=='steps':v=np.repeat(rng.uniform(-100,60,8),50)
            elif f=='sinusoid':v=-20+75*np.sin(2*np.pi*rng.uniform(.025,.2)*t+rng.uniform(0,6))
            elif f=='chirp':v=-20+75*np.sin(2*np.pi*(.02*t+.005*t*t)+rng.uniform(0,6))
            elif f=='pulse_recovery':v=np.where((t>10)&(t<18),rng.uniform(0,60),rng.uniform(-100,-65))
            else:v=np.clip(np.cumsum(rng.normal(0,4,400))-65,-100,60)
            paths.append(v);families.append(f)
    return np.array(paths).T,rng.uniform(0,1,(24,17)),families

def fresh(torch,ops,spec,output):
    gen=torch.Generator(device='cuda').manual_seed(spec['data_seed']+10)
    v=torch.rand(spec['fresh_count'],device='cuda',generator=gen)*160-100;x=torch.rand(len(v),17,device='cuda',generator=gen)
    ut,zt=p4.coefficients(torch,v);truth=(1-zt)*x+zt*ut
    pv,x0,families=path_data(spec['path_seed']);vv=torch.tensor(pv,device='cuda',dtype=torch.float32);initial=torch.tensor(x0,device='cuda',dtype=torch.float32)
    reference=rollout(torch,*p4.coefficients(torch,vv),initial);result={}
    np.savez_compressed(output/'fresh_imposed_paths.npz',voltage_mv=pv,initial_gates=x0,families=families)
    for name,op in ops.items():
        u,z=p4.coefficients(torch,v) if op is None else op(v);pred=rollout(torch,*(p4.coefficients(torch,vv) if op is None else op(vv)),initial)
        full=p4.restore(torch,pred,torch.zeros(400,24,18,device='cuda'));rf=p4.restore(torch,reference,torch.zeros_like(full))
        scale=torch.tensor(p4.ionic.GBAR,device='cuda')*(vv[...,None]-torch.tensor(p4.ionic.REVERSALS,device='cuda'))
        current=p3.exact_apertures(torch,full)*scale;rc=p3.exact_apertures(torch,rf)*scale
        result[name]={'fresh_gate':p3.errors(torch,(1-z)*x+z*u,truth),'forced_path_gate':p3.errors(torch,pred,reference),
            'families':{f:p3.errors(torch,pred[:,[i for i,g in enumerate(families) if g==f]],reference[:,[i for i,g in enumerate(families) if g==f]]) for f in sorted(set(families))},
            'gate_rmse_by_state':(pred-reference).square().mean((0,1)).sqrt().tolist(),'current_rmse_by_channel':(current-rc).square().mean((0,1)).sqrt().tolist(),
            'charge_max_abs_by_channel':((current-rc).sum(0)*.1).abs().max(0).values.tolist(),'inf_probe':p3.errors(torch,u,ut),'log_z_probe':p3.errors(torch,torch.log(z+1e-12),torch.log(zt+1e-12)),
            'occupancy_violations':int(((pred<0)|(pred>1)|~torch.isfinite(pred)).sum())}
    return result

def profiles(torch,ops):
    v=torch.linspace(-100,60,2049,device='cuda');ut,zt=p4.coefficients(torch,v);report={}
    for name,op in ops.items():
        u,z=p4.coefficients(torch,v) if op is None else op(v)
        error=torch.maximum((z*u-zt*ut).abs(),((1-z)+z*u-(1-zt)-zt*ut).abs())
        bins=((v+100)/2.5).long().clamp(0,63)
        report[name]={'gate_max':[float(t) for t in error.amax(0)],'bin_gate_max':torch.stack([error[bins==k].amax(0) for k in range(64)]).cpu().tolist()}
    return {'source':'post-freeze diagnostic grid; not selection', 'profiles':report}

def qualify(f,h,q,conversion=True):
    return bool(f['fresh_gate']['rmse']<=q['fresh_gate_rmse'] and f['fresh_gate']['max_abs']<=q['fresh_gate_max_abs'] and
        all(m['rmse']<=q['forced_path_gate_rmse'] and m['max_abs']<=q['forced_path_gate_max_abs'] for m in f['families'].values()) and
        h['voltage']['rmse']<=q['historical_voltage_rmse_mv'] and h['voltage']['max_abs']<=q['historical_voltage_max_abs_mv'] and h['calcium']['max_abs']<=q['historical_calcium_max_abs_mm'] and
        h['voltage_domain_exit_count']==0 and f['occupancy_violations']==0 and conversion)

def run(root,output,revision):
    import torch
    root,output=Path(root),Path(output);spec=json.loads((root/CONFIG).read_text());parent_path=root/spec['parent_checkpoint']
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==spec['parent_sha256'];assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch._dynamo.config.cache_size_limit=max(256,torch._dynamo.config.cache_size_limit)
    parent=torch.load(parent_path,map_location='cpu',weights_only=True);_,cfg,calcium,_,base=p5.n1.contract(root);assert cfg['dt_ms']==spec['dt_ms']
    validation=preflight(torch,parent,'cuda');write(output/'validation.json',validation)
    write(output/'gradient_probe.json',gradient_probe(torch,parent,'cuda'))
    original=dense(torch,parent,'cuda');init={}
    grid=torch.linspace(-100,60,65537,device='cuda')
    with torch.no_grad():
        for knots in (81,161):
            lm=local(torch,parent,knots,'cuda')
            for k,s in enumerate(spec['seeds']):
                u,z=local_operator(torch,lm,k)(grid);a,b=p5.operator(torch,original,k)(grid)
                init[f'local{knots}_seed{s}']={'parameter_bytes':lm.table[k].numel()*4+17*4,'maximum_initial_endpoint_delta':max(float((z*u-b*a).abs().max()),float(((1-z)+z*u-(1-b)-b*a).abs().max()))}
    write(output/'local_initialization.json',init)
    models,freeze=train(torch,parent,spec,output)
    ops={'formula_vectorized':None,'lut025':p4.lut_operator(torch)};frozen={f'parent_seed{s}':p5.operator(torch,original,k) for k,s in enumerate(spec['seeds'])};ops.update(frozen)
    conversions={};tables={}
    with torch.inference_mode():
        for k,(arm,s) in enumerate(MEMBERS):
            name=f'{arm}_seed{s}';op=p5.operator(torch,models['dense'],k);ops[name]=op
            top,table=p5.tabulate(torch,op,.25,'cuda');key=name+'_table025';ops[key]=top;tables[key]=table.cpu()
            u,z=op(grid);a,b=top(grid);delta=max(float((a*b-u*z).abs().max()),float(((1-b)+a*b-(1-z)-u*z).abs().max()))
            conversions[key]={'parent':name,'table_bytes':table.numel()*4,'maximum_endpoint_map_delta':delta,'conversion_passed':delta<=spec['conversion_gate_max_delta']}
        for knots in (81,161):
            for k,s in enumerate(spec['seeds']):ops[f'local{knots}_seed{s}']=local_operator(torch,models[f'local{knots}'],k)
        torch.save(tables,output/'neural_deployment_tables.pt');write(output/'conversion_report.json',conversions)
        write(output/'error_profiles.json',profiles(torch,ops))
        evaluation=fresh(torch,ops,spec,output);write(output/'fresh_evaluation.json',evaluation)
        with contextlib.redirect_stdout(Progress()):
            history=p4.historical(torch,root,ops,cfg,calcium,base);write(output/'historical_evaluation.json',history)
            interventions={name+'_'+kind:p5.swap(torch,op,kind) for name,op in frozen.items() for kind in ('u_all','z_all',*p5.GROUPS)}
            attribution=p4.historical(torch,root,interventions,cfg,calcium,base);write(output/'diagnostic_oracle_attribution.json',{'parent':'IE-P5 standard','selection_eligible':False,'metrics':attribution})
            rows,checks,failures=p4.timing(torch,root,ops,spec,cfg,calcium,base,output)
    qualified={name:qualify(f,history[name],spec['quality'],conversions.get(name,{}).get('conversion_passed',True)) for name,f in evaluation.items()}
    def score(name):
        f=evaluation[name]
        return max(f['fresh_gate']['rmse']/.001,f['fresh_gate']['max_abs']/.01,max(max(m['rmse']/.005,m['max_abs']/.03) for m in f['families'].values()))
    contrasts=[]
    for s in spec['seeds']:
        a,b,c,d=[score(f'{arm}_seed{s}') for arm in ARMS]
        contrasts.append({'seed':s,'scores':dict(zip(ARMS,(a,b,c,d))),'B_minus_A':b-a,'C_minus_A':c-a,'D_minus_B':d-b,'D_minus_C':d-c,'interaction':(d-c)-(b-a),'A_vs_parent_reduction':1-a/score(f'parent_seed{s}')})
    speed={}
    for row in rows:
        if row['batch']==spec['execution']['primary_batch']:
            saving=row['saving_vs_formula']
            speed[f"{row['mode']}/{row['name']}"]={'speed_passed':saving['median_saved_fraction']>=spec['execution']['minimum_paired_saving'] and saving['bootstrap_95pct'][0]>0,'quality_passed':qualified[row['name']]}
    bytes_={name:9420 for name in ops if name.startswith(tuple(ARMS)+('parent_',)) and not name.endswith('table025')}
    bytes_.update({name:item['table_bytes'] for name,item in conversions.items()});bytes_.update({name:item['parameter_bytes'] for name,item in init.items()})
    report={'valid':not failures and len(checks)==210,'scientific_no_go':False,'schema_version':spec['schema_version'],'code_revision':revision,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'selection':freeze,'validation':validation,
        'fresh':evaluation,'historical':history,'quality_qualified':qualified,'conversions':conversions,'local_initialization':init,'paired_contrasts':contrasts,'rows':rows,'preflights':checks,'graph_failures':failures,
        'primary_cost_quality':speed,'representation_bytes':bytes_,'candidate_promoted':False,'deployment_authorized':False,'exclusions':spec['exclusions']}
    write(output/'final_report.json',report);print('[GIADA IE-P6] complete; inspect all quality and paired cost contrasts',flush=True)
    return report
