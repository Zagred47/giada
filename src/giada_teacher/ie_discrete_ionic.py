"""IE-P4 joint17gate discrete coefficients, explicit state and exact readout."""
import hashlib
import json
from pathlib import Path
import numpy as np
from . import ie_residual_cost as p2
from . import ie_readout_comparison as p3
from . import ionic_block_teacher_forced as ionic
from . import cc_n1_contract_memory as n1
from .gate_d_compute_audit import _rates
from .ie_ionic_cost import paired_summary

CONFIG='experiments/ie_p4_discrete_ionic_preregistration_v1.json'
IDS=tuple(i for i in range(18) if i!=16)
MEMBERS=tuple((loss,width,seed) for loss in ('transition','internal') for width in (32,64) for seed in (17,29,43))


def coefficients(torch,v,dt=.1):
    uu=[];zz=[]
    for k,name in enumerate(ionic.CHANNELS):
        if name=='SK_E2':continue
        u,t=_rates(name,v,torch.full_like(v,1e-4),torch)
        if len(ionic.STATE_NAMES[k])==1:u,t=u[...,None],t[...,None]
        uu.append(u);zz.append(-torch.expm1(-dt/t))
    return torch.cat(uu,-1),torch.cat(zz,-1)


def restore(torch,new,state):
    return torch.cat((new[...,:16],state[...,16:17],new[...,16:]),-1)


def lut_operator(torch,spacing=.25,device='cuda'):
    grid=torch.linspace(-100,60,round(160/spacing)+1,device=device)
    u,z=coefficients(torch,grid)
    table=torch.cat((u,z),-1).detach()
    def op(v):
        position=((v.clamp(-100,60)+100)/spacing)
        index=position.floor().long().clamp(0,len(grid)-2)
        fraction=(position-index)[...,None]
        value=table[index]*(1-fraction)+table[index+1]*fraction
        return value[...,:17],value[...,17:]
    return op


def core(torch,v,ca,state,mult,inj,syn,area,cfg,calcium,base,operator=None):
    if operator is None:
        nc,ng,_,_=p2.ionic_prefix(torch,v,ca,state,mult,cfg,calcium)
    else:
        u,z=operator(v)
        old=state[...,list(IDS)]
        raw=restore(torch,(1-z)*old+z*u,state)
        nc,ng,_,_=p2.ionic_prefix(torch,v,ca,state,mult,cfg,calcium,gate_cache=raw)
    gs,ge=p3.readout(torch,ng,mult)
    nv=p2.membrane_solve(torch,v,gs,ge,inj,syn,area,cfg,base)
    return nv,nc,ng,gs,ge


def ensemble(torch,zscale,device):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a=torch.nn.Parameter(torch.empty(12,1,64,device=device))
            self.b=torch.nn.Parameter(torch.zeros(12,1,64,device=device))
            self.c=torch.nn.Parameter(torch.empty(12,64,34,device=device))
            self.d=torch.nn.Parameter(torch.zeros(12,1,34,device=device))
            for k,(_,width,seed) in enumerate(MEMBERS):
                gen=torch.Generator(device=device).manual_seed(seed)
                self.a.data[k].normal_(0,.8,generator=gen);self.c.data[k].normal_(0,.1,generator=gen)
            self.register_buffer('mask',(torch.arange(64,device=device)[None,:]<torch.tensor([m[1] for m in MEMBERS],device=device)[:,None]).float()[:,None,:])
            self.register_buffer('scale',zscale)
        def forward(self,v):
            h=torch.nn.functional.silu(torch.bmm((v[...,None]+40)/80,self.a)+self.b)*self.mask
            y=torch.sigmoid(torch.bmm(h,self.c)+self.d)
            return y[...,:17],y[...,17:]*self.scale
    return Model()


def operator(torch,model,index):
    width=MEMBERS[index][1]
    a=model.a[index,:,:width].detach().clone();b=model.b[index,:,:width].detach().clone()
    c=model.c[index,:width].detach().clone();d=model.d[index].detach().clone();scale=model.scale.detach().clone()
    def op(v):
        h=torch.nn.functional.silu(((v[...,None]+40)/80)@a+b)
        y=torch.sigmoid(h@c+d)
        return y[...,:17],y[...,17:]*scale
    return op


def clip_members(torch,model):
    norms=torch.stack([p.grad.flatten(1).square().sum(-1) for p in model.parameters()]).sum(0).sqrt()
    for p in model.parameters():p.grad.mul_((1/norms.clamp_min(1)).reshape(12,*([1]*(p.ndim-1))))


def preflight(torch,device):
    """Runtime equivalence: batched versus isolated member Adam updates."""
    scale=torch.ones(17,device=device)*.1
    model=ensemble(torch,scale,device)
    gen=torch.Generator(device=device).manual_seed(81700)
    v=torch.rand(12,71,device=device,generator=gen)*160-100
    target=torch.rand(12,71,17,device=device,generator=gen)
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    u,z=model(v);((u-target).square().mean((1,2))+z.square().mean((1,2))).sum().backward()
    clip_members(torch,model);opt.step();maximum=0.
    for k in range(12):
        iso=ensemble(torch,scale,device)
        with torch.no_grad():
            for n,p in iso.named_parameters():p.copy_(initial[n])
        opt_i=torch.optim.Adam(iso.parameters(),lr=.003)
        u,z=iso(v);((u[k]-target[k]).square().mean()+z[k].square().mean()).backward()
        clip_members(torch,iso);opt_i.step()
        for n,p in iso.named_parameters():maximum=max(maximum,float((p[k]-dict(model.named_parameters())[n][k]).abs().max()))
        expected=model(v)[0][k];actual=operator(torch,model,k)(v[k])[0]
        maximum=max(maximum,float((expected-actual).abs().max()))
    if maximum>1e-5:raise RuntimeError('Batched Adam/prediction preflight failed')
    return {'valid':True,'maximum_error':maximum,'atol':1e-5,'independent_adam_and_clipping':True}


def oracle(torch):
    values=np.unique(np.concatenate((np.linspace(-100,60,1025),[-66,-60,-38,-32,-27,-17,-64.4])))
    g=np.random.default_rng(81700).uniform(0,1,(len(values),18))
    expected=ionic.exact_step(values,np.full(len(values),1e-4),g,np.full(len(values),.1))[:,IDS]
    v=torch.tensor(values,dtype=torch.float64);u,z=coefficients(torch,v)
    actual=(1-z)*torch.tensor(g[:,IDS])+z*u
    error=float((actual-torch.tensor(expected)).abs().max())
    if not np.isfinite(error) or error>2e-5:raise RuntimeError(f'NumPy64 oracle mismatch {error}')
    return {'valid':True,'maximum_gate_error':error,'native_confirmation':False,'sample_count':len(values)}


def train(torch,spec,output):
    gen=torch.Generator(device='cuda').manual_seed(spec['data_seed'])
    fit=torch.rand(spec['fit_count'],device='cuda',generator=gen)*160-100
    dev=torch.rand(spec['development_count'],device='cuda',generator=gen)*160-100
    xf=torch.rand(len(fit),17,device='cuda',generator=gen);xd=torch.rand(len(dev),17,device='cuda',generator=gen)
    uf,zf=coefficients(torch,fit);ud,zd=coefficients(torch,dev)
    _,zg=coefficients(torch,torch.linspace(-100,60,4097,device='cuda'))
    scale=(zg.max(0).values*1.001).clamp(max=1).detach()
    model=ensemble(torch,scale,'cuda');opt=torch.optim.Adam(model.parameters(),lr=spec['learning_rate'])
    internal=torch.tensor([m[0]=='internal' for m in MEMBERS],device='cuda').float()
    best=[float('inf')]*12;states=[None]*12;selected=[None]*12;curves=[]
    for step in range(spec['training_steps']+1):
        if step in spec['checkpoints']:
            with torch.no_grad():
                u,z=model(dev[None].expand(12,-1));pred=(1-z)*xd+z*u;target=(1-zd)*xd+zd*ud
                scores=((pred-target)/scale).square().mean((1,2)).sqrt().tolist()
            curves.append({'step':step,'development_normalized_transition_rmse':scores})
            torch.save(model.state_dict(),output/f'checkpoint_{step}.pt')
            (output/'training_curves.json').write_text(json.dumps(curves,indent=2))
            for k,score in enumerate(scores):
                if not np.isfinite(score):raise RuntimeError('Nonfinite training trajectory')
                if score<best[k]:
                    best[k]=score;selected[k]=step;states[k]={n:p[k].detach().clone() for n,p in model.named_parameters()}
            print(f'[GIADA IE-P4] training {step}/{spec["training_steps"]} dev median={np.median(scores):.6g}',flush=True)
        if step==spec['training_steps']:break
        ids=torch.randint(len(fit),(spec['batch_size'],),device='cuda',generator=gen)
        u,z=model(fit[ids][None].expand(12,-1));x=xf[ids];target=(1-zf[ids])*x+zf[ids]*uf[ids]
        loss=(((1-z)*x+z*u-target)/scale).square().mean((1,2))
        aux=(u-uf[ids]).square().mean((1,2))+.01*(torch.log(z/scale+1e-8)-torch.log(zf[ids]/scale+1e-8)).square().mean((1,2))
        opt.zero_grad();(loss+internal*aux).sum().backward();clip_members(torch,model);opt.step()
    with torch.no_grad():
        for k in range(12):
            for n,p in model.named_parameters():p[k].copy_(states[k][n])
    torch.save(model.state_dict(),output/'frozen_checkpoint.pt')
    freeze={'selected_steps':selected,'development_scores':best,'fresh_accessed':False,
            'checkpoint_sha256':hashlib.sha256((output/'frozen_checkpoint.pt').read_bytes()).hexdigest(),
            'members':MEMBERS,'parameter_counts':[m[1]*36+34 for m in MEMBERS]}
    (output/'selection_freeze.json').write_text(json.dumps(freeze,indent=2))
    return model,freeze


def path_data():
    rng=np.random.default_rng(81702);t=np.arange(400)*.1;paths=[];families=[]
    for family in ('constant','steps','sinusoid','chirp','pulse_recovery','random_walk'):
        for _ in range(4):
            if family=='constant':v=np.full(400,rng.uniform(-100,60))
            elif family=='steps':v=np.repeat(rng.uniform(-100,60,8),50)
            elif family=='sinusoid':v=-20+75*np.sin(2*np.pi*rng.uniform(.025,.2)*t+rng.uniform(0,6))
            elif family=='chirp':v=-20+75*np.sin(2*np.pi*(.02*t+.005*t*t)+rng.uniform(0,6))
            elif family=='pulse_recovery':v=np.where((t>10)&(t<18),rng.uniform(0,60),rng.uniform(-100,-65))
            else:v=np.clip(np.cumsum(rng.normal(0,4,400))-65,-100,60)
            paths.append(v);families.append(family)
    return np.array(paths).T,rng.uniform(0,1,(24,17)),families


def fresh_evaluation(torch,ops,spec,output):
    gen=torch.Generator(device='cuda').manual_seed(spec['data_seed']+10)
    v=torch.rand(spec['fresh_count'],device='cuda',generator=gen)*160-100
    x=torch.rand(len(v),17,device='cuda',generator=gen);ut,zt=coefficients(torch,v);target=(1-zt)*x+zt*ut
    pv,x0,families=path_data();vv=torch.tensor(pv,device='cuda',dtype=torch.float32)
    uu,zz=coefficients(torch,vv);initial=torch.tensor(x0,device='cuda',dtype=torch.float32)
    def rollout(u,z):
        g=initial;values=[]
        for t in range(400):g=(1-z[t])*g+z[t]*u[t];values.append(g)
        return torch.stack(values)
    reference=rollout(uu,zz);report={}
    np.savez_compressed(output/'fresh_imposed_paths.npz',voltage_mv=pv,initial_gates=x0,families=families)
    for name,op in ops.items():
        u,z=coefficients(torch,v) if op is None else op(v)
        pu,pz=coefficients(torch,vv) if op is None else op(vv)
        trajectory=rollout(pu,pz)
        by_family={f:p3.errors(torch,trajectory[:,[i for i,g in enumerate(families) if g==f]],reference[:,[i for i,g in enumerate(families) if g==f]]) for f in sorted(set(families))}
        full=restore(torch,trajectory,torch.zeros(400,24,18,device='cuda'))
        ref_full=restore(torch,reference,torch.zeros_like(full))
        current=(p3.exact_apertures(torch,full)*torch.tensor(ionic.GBAR,dtype=full.dtype,device='cuda')*(vv[...,None]-torch.tensor(ionic.REVERSALS,dtype=full.dtype,device='cuda')))
        refcurrent=(p3.exact_apertures(torch,ref_full)*torch.tensor(ionic.GBAR,dtype=full.dtype,device='cuda')*(vv[...,None]-torch.tensor(ionic.REVERSALS,dtype=full.dtype,device='cuda')))
        report[name]={'fresh_gate':p3.errors(torch,(1-z)*x+z*u,target),'inf_probe':p3.errors(torch,u,ut),
            'log_z_probe':p3.errors(torch,torch.log(z+1e-12),torch.log(zt+1e-12)),
            'forced_path_gate':p3.errors(torch,trajectory,reference),'families':by_family,
            'current':p3.errors(torch,current,refcurrent),'charge_max_abs_ma_ms_cm2':float(((current-refcurrent).sum(0)*.1).abs().max()),
            'gate_rmse_by_state':(trajectory-reference).square().mean((0,1)).sqrt().tolist(),
            'current_rmse_by_channel_ma_cm2':(current-refcurrent).square().mean((0,1)).sqrt().tolist(),
            'charge_max_abs_by_channel_ma_ms_cm2':((current-refcurrent).sum(0)*.1).abs().max(0).values.tolist(),
            'occupancy_violations':int(((trajectory<0)|(trajectory>1)|~torch.isfinite(trajectory)).sum())}
    return report


def historical(torch,root,ops,cfg,calcium,base):
    traces=np.load(root/'experiments/results/task35_native_shadow_traces.npz');report={}
    def t(x):return torch.tensor(x,device='cuda',dtype=torch.float32)
    injections=t(traces['injection']);synapses=t(traces['base_g_us']);mult=t(traces['multipliers'])
    for name,op in ops.items():
        v=t(traces['voltage'][:,0]);ca=t(traces['calcium'][:,0]);g=t(traces['gates'][:,0]);vs=[];cs=[];outside=0
        for tick in range(400):
            outside+=int(((v<-100)|(v>60)|~torch.isfinite(v)).sum())
            v,ca,g,*_=core(torch,v,ca,g,mult,injections[:,tick],synapses[:,tick],float(traces['area_um2']),cfg,calcium,base,op)
            vs.append(v);cs.append(ca)
        report[name]={'voltage':p3.errors(torch,torch.stack(vs,1),t(traces['voltage'][:,1:])),
                      'calcium':p3.errors(torch,torch.stack(cs,1),t(traces['calcium'][:,1:])),'voltage_domain_exit_count':outside}
        print(f'[GIADA IE-P4] historical {name} V={report[name]["voltage"]["rmse"]:.6g}',flush=True)
    return report


def timing(torch,root,ops,spec,cfg,calcium,base,output):
    traces=np.load(root/'experiments/results/task35_native_shadow_traces.npz');rng=np.random.default_rng(spec['data_seed'])
    rows=[];checks=[];failures=[]
    for batch in spec['execution']['batches']:
        ep=rng.integers(0,32,batch);tick=rng.integers(0,400,batch)
        def t(x):return torch.tensor(x,device='cuda',dtype=torch.float32)
        args=(t(traces['voltage'][ep,tick]),t(traces['calcium'][ep,tick]),t(traces['gates'][ep,tick]),t(traces['multipliers'][ep]),t(traces['injection'][ep,tick]),t(traces['base_g_us'][ep,tick]),float(traces['area_um2']),cfg,calcium,base)
        funcs={};held_graphs=[]
        for name,op in ops.items():
            def fn(op=op):return core(torch,*args,operator=op)
            eager=fn();compiled=torch.compile(fn,fullgraph=True);actual=compiled();torch.cuda.synchronize()
            eq=p2.differences(torch,actual,eager,spec['numeric_limits']);checks.append({'name':name,'batch':batch,'mode':'compiled',**eq})
            if not eq['valid']:raise RuntimeError(f'Compiled preflight {name}: {eq}')
            funcs['compiled',name]=compiled
            for _ in range(spec['execution']['warmup']):compiled()
            torch.cuda.synchronize()
            try:
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):compiled()
                stream.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):held=compiled()
                graph.replay();torch.cuda.synchronize()
                eq=p2.differences(torch,held,eager,spec['numeric_limits']);checks.append({'name':name,'batch':batch,'mode':'cuda_graph',**eq})
                if not eq['valid']:raise RuntimeError(f'Graph mismatch {eq}')
                held_graphs.append((graph,held));funcs['cuda_graph',name]=graph.replay
            except Exception as error:failures.append({'name':name,'batch':batch,'error':str(error)})
        samples=p2.benchmark(torch,funcs,spec['execution'])
        for (mode,name),times in samples.items():
            reference=samples.get((mode,'formula_vectorized'))
            rows.append({'batch':batch,'mode':mode,'name':name,'timings':times,
                'saving_vs_formula':paired_summary(reference['wall_ms'],times['wall_ms']) if reference else None})
        (output/'timing_progress.json').write_text(json.dumps({'rows':rows,'preflights':checks,'graph_failures':failures},indent=2))
        print(f'[GIADA IE-P4] benchmark batch={batch} complete',flush=True)
    return rows,checks,failures


def run(root,output,revision):
    import torch
    root,output=Path(root),Path(output);spec=json.loads((root/CONFIG).read_text())
    _,cfg,calcium,_,base=n1.contract(root);assert cfg['dt_ms']==spec['dt_ms'] and torch.cuda.is_available()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch._dynamo.config.cache_size_limit=max(256,torch._dynamo.config.cache_size_limit)
    validation={'oracle':oracle(torch),'batched_adam':preflight(torch,'cuda')}
    (output/'validation.json').write_text(json.dumps(validation,indent=2))
    model,freeze=train(torch,spec,output)
    ops={'formula_vectorized':None,'lut025':lut_operator(torch,spec['lut_spacing_mv'])}
    for k,(loss,width,seed) in enumerate(MEMBERS):ops[f'{loss}_w{width}_seed{seed}']=operator(torch,model,k)
    with torch.inference_mode():
        fresh=fresh_evaluation(torch,ops,spec,output)
        (output/'fresh_evaluation.json').write_text(json.dumps(fresh,indent=2,allow_nan=False))
        history=historical(torch,root,ops,cfg,calcium,base)
        (output/'historical_evaluation.json').write_text(json.dumps(history,indent=2,allow_nan=False))
        rows,checks,failures=timing(torch,root,ops,spec,cfg,calcium,base,output)
    q=spec['quality'];qualified={}
    for name in ops:
        f=fresh[name];h=history[name]
        qualified[name]=bool(f['fresh_gate']['rmse']<=q['fresh_gate_rmse'] and f['fresh_gate']['max_abs']<=q['fresh_gate_max_abs'] and
            all(x['rmse']<=q['forced_path_gate_rmse'] and x['max_abs']<=q['forced_path_gate_max_abs'] for x in f['families'].values()) and
            h['voltage']['rmse']<=q['historical_voltage_rmse_mv'] and h['voltage']['max_abs']<=q['historical_voltage_max_abs_mv'] and
            h['calcium']['max_abs']<=q['historical_calcium_max_abs_mm'] and h['voltage_domain_exit_count']==0 and f['occupancy_violations']==0)
    report={'valid':True,'schema_version':spec['schema_version'],'code_revision':revision,'device':torch.cuda.get_device_name(),'torch':torch.__version__,
        'validation':validation,'selection':freeze,'fresh':fresh,'historical':history,'quality_qualified':qualified,
        'rows':rows,'preflights':checks,'graph_failures':failures,'candidate_promoted':False,'deployment_authorized':False,'exclusions':spec['exclusions']}
    (output/'final_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('[GIADA IE-P4] complete; inspect quality and cost separately',flush=True)
    return report
