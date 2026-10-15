"""IE-P3: exact algebra versus learned apertures, with common state dynamics."""
import json
import hashlib
import time
from pathlib import Path
import numpy as np
from . import ie_residual_cost as p2
from . import cc_n1_contract_memory as n1
from . import ionic_block_teacher_forced as ionic
from .gate_d_compute_audit import OFFSETS, POWERS, GBAR_VALUES, REVERSAL_VALUES
from .ie_ionic_cost import paired_summary

CONFIG='experiments/ie_p3_readout_preregistration_v1.json'


def features(torch,g):
    first=torch.tensor(OFFSETS[:-1],device=g.device)
    second=torch.tensor([OFFSETS[k]+1 if len(s)==2 else OFFSETS[k] for k,s in enumerate(ionic.STATE_NAMES)],device=g.device)
    has=torch.tensor([len(s)==2 for s in ionic.STATE_NAMES],device=g.device)
    m=g[...,first];h=torch.where(has,g[...,second],torch.ones_like(m))
    power=torch.tensor(POWERS,dtype=g.dtype,device=g.device).expand_as(m)
    return torch.stack((m,h,power/4,has.to(g.dtype).expand_as(m)),-1)


def exact_apertures(torch,g):
    x=features(torch,g)
    m,h=x[...,0],x[...,1]
    powers=torch.tensor(POWERS,device=g.device)
    # Integer multiplications, not a floating-exponent approximation.
    m2=m*m
    return torch.where(powers==1,m,torch.where(powers==2,m2,torch.where(powers==3,m2*m,m2*m2)))*h


def readout(torch,g,mult,operator=None):
    if operator is None:
        opening=exact_apertures(torch,g)
    else:
        opening=operator(features(torch,g))
    conductance=opening*torch.tensor(GBAR_VALUES,dtype=g.dtype,device=g.device)*mult
    reverse=torch.tensor(REVERSAL_VALUES,dtype=g.dtype,device=g.device)
    return conductance.sum(-1),(conductance*reverse).sum(-1)


def core(torch,v,ca,g,mult,inj,syn,area,cfg,calcium,base,operator=None,legacy=False):
    # Shared verified phase: updated Ca gates -> Ca -> updated SK -> readout.
    nc,ng,gs,ge=p2.ionic_prefix(torch,v,ca,g,mult,cfg,calcium)
    if not legacy:gs,ge=readout(torch,ng,mult,operator)
    nv=p2.membrane_solve(torch,v,gs,ge,inj,syn,area,cfg,base)
    return nv,nc,ng,gs,ge


def make_ensemble(torch,device):
    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a=torch.nn.Parameter(torch.empty(6,4,32,device=device))
            self.b=torch.nn.Parameter(torch.zeros(6,1,32,device=device))
            self.c=torch.nn.Parameter(torch.empty(6,32,1,device=device))
            self.d=torch.nn.Parameter(torch.zeros(6,1,1,device=device))
            for k,seed in enumerate((17,29,43,17,29,43)):
                gen=torch.Generator(device=device).manual_seed(seed)
                self.a.data[k].normal_(0,.4,generator=gen)
                self.c.data[k].normal_(0,.1,generator=gen)
            self.register_buffer('mask',(torch.arange(32,device=device)[None,:]<torch.tensor([16]*3+[32]*3,device=device)[:,None]).float()[:,None,:])
        def forward(self,x):
            z=torch.nn.functional.silu(torch.bmm(x,self.a)+self.b)*self.mask
            return torch.sigmoid(torch.bmm(z,self.c)+self.d).squeeze(-1)
    return Ensemble()


def independent_operator(torch,model,index,width):
    a=model.a[index,:,:width].detach().clone();b=model.b[index,:,:width].detach().clone()
    c=model.c[index,:width].detach().clone();d=model.d[index].detach().clone()
    def op(x):return torch.sigmoid(torch.nn.functional.silu(x@a+b)@c+d).squeeze(-1)
    return op


def errors(torch,pred,target):
    diff=pred-target
    return {'rmse':float(diff.square().mean().sqrt()),'max_abs':float(diff.abs().max()),'finite':bool(torch.isfinite(pred).all())}


def train(torch,spec,output):
    device='cuda';gen=torch.Generator(device=device).manual_seed(spec['data_seed'])
    fit=torch.rand(spec['fit_count'],18,device=device,generator=gen)
    dev=torch.rand(spec['development_count'],18,device=device,generator=gen)
    fresh=torch.rand(spec['fresh_count'],18,device=device,generator=gen)
    xf=features(torch,fit).reshape(-1,4);yf=exact_apertures(torch,fit).reshape(-1)
    xd=features(torch,dev).reshape(-1,4);yd=exact_apertures(torch,dev).reshape(-1)
    model=make_ensemble(torch,device);optimizer=torch.optim.Adam(model.parameters(),lr=spec['learning_rate'])
    best=[float('inf')]*6;frozen=[None]*6;curves=[];selected=[None]*6
    for step in range(spec['training_steps']+1):
        if step in spec['checkpoints']:
            with torch.no_grad():scores=(model(xd[None].expand(6,-1,-1))-yd).square().mean(-1).sqrt().tolist()
            curves.append({'step':step,'development_rmse':scores})
            torch.save(model.state_dict(),output/f'checkpoint_{step}.pt')
            (output/'training_curves.json').write_text(json.dumps(curves,indent=2))
            for k,score in enumerate(scores):
                if score<best[k]:
                    best[k]=score;selected[k]=step
                    frozen[k]={n:p[k].detach().clone() for n,p in model.named_parameters()}
            print(f'[GIADA IE-P3] training {step}/{spec["training_steps"]} dev median={np.median(scores):.6g}',flush=True)
        if step==spec['training_steps']:break
        indices=torch.randint(len(xf),(spec['batch_size'],),device=device,generator=gen)
        pred=model(xf[indices][None].expand(6,-1,-1));loss=(pred-yf[indices]).square().mean(-1)
        optimizer.zero_grad();loss.sum().backward()
        norms=torch.stack([p.grad.flatten(1).square().sum(1) for p in model.parameters()]).sum(0).sqrt()
        scale=(1/norms.clamp_min(1)).detach()
        for p in model.parameters():p.grad.mul_(scale.reshape(6,*([1]*(p.ndim-1))))
        optimizer.step()
    freeze={'selected_steps':selected,'development_rmse':best,'fresh_accessed':False}
    with torch.no_grad():
        for k in range(6):
            for n,p in model.named_parameters():p[k].copy_(frozen[k][n])
    torch.save(model.state_dict(),output/'frozen_checkpoint.pt')
    freeze['checkpoint_sha256']=hashlib.sha256((output/'frozen_checkpoint.pt').read_bytes()).hexdigest()
    (output/'selection_freeze.json').write_text(json.dumps(freeze,indent=2))
    fresh_report=[]
    with torch.no_grad():
        for k in range(6):
            op=independent_operator(torch,model,k,16 if k<3 else 32)
            fresh_report.append(errors(torch,op(features(torch,fresh)),exact_apertures(torch,fresh)))
    (output/'training_curves.json').write_text(json.dumps(curves,indent=2))
    return model,freeze,fresh_report


def run(root,output,revision):
    import torch
    root,output=Path(root),Path(output)
    spec=json.loads((root/CONFIG).read_text());_,cfg,calcium,_,base=n1.contract(root)
    assert torch.cuda.is_available() and cfg['dt_ms']==spec['dt_ms']
    torch.set_num_threads(1)
    torch._dynamo.config.cache_size_limit=max(128,torch._dynamo.config.cache_size_limit)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model,freeze,fresh=train(torch,spec,output)
    operators={'vectorized_algebra':None,'legacy_algebra':None}
    for k in range(6):operators[f'learned_width{16 if k<3 else 32}_seed{spec["seeds"][k%3]}']=independent_operator(torch,model,k,16 if k<3 else 32)
    traces=np.load(root/'experiments/results/task35_native_shadow_traces.npz')
    rng=np.random.default_rng(spec['data_seed']);rows=[];checks=[];failures=[];diagnostics={}
    # Historical closed-loop diagnostics: all candidates start from the same physical state.
    with torch.no_grad():
        for name,op in operators.items():
            v=torch.tensor(traces['voltage'][:,0],device='cuda',dtype=torch.float32)
            ca=torch.tensor(traces['calcium'][:,0],device='cuda',dtype=torch.float32)
            g=torch.tensor(traces['gates'][:,0],device='cuda',dtype=torch.float32)
            mult=torch.tensor(traces['multipliers'],device='cuda',dtype=torch.float32)
            vs=[];cs=[]
            for t in range(traces['injection'].shape[1]):
                inj=torch.tensor(traces['injection'][:,t],device='cuda',dtype=torch.float32)
                syn=torch.tensor(traces['base_g_us'][:,t],device='cuda',dtype=torch.float32)
                v,ca,g,*_=core(torch,v,ca,g,mult,inj,syn,float(traces['area_um2']),cfg,calcium,base,op,name=='legacy_algebra')
                vs.append(v);cs.append(ca)
            vv=torch.stack(vs,1);cc=torch.stack(cs,1)
            diagnostics[name]={'voltage':errors(torch,vv,torch.tensor(traces['voltage'][:,1:],device='cuda',dtype=torch.float32)),
                               'calcium':errors(torch,cc,torch.tensor(traces['calcium'][:,1:],device='cuda',dtype=torch.float32))}
            print(f'[GIADA IE-P3] diagnostic {name} V={diagnostics[name]["voltage"]["rmse"]:.6g}',flush=True)
        for batch in spec['execution']['batches']:
            ep=rng.integers(0,32,batch);tick=rng.integers(0,400,batch)
            def tensor(x):return torch.tensor(x,device='cuda',dtype=torch.float32)
            args=(tensor(traces['voltage'][ep,tick]),tensor(traces['calcium'][ep,tick]),tensor(traces['gates'][ep,tick]),
                  tensor(traces['multipliers'][ep]),tensor(traces['injection'][ep,tick]),tensor(traces['base_g_us'][ep,tick]),float(traces['area_um2']),cfg,calcium,base)
            parity=p2.differences(torch,core(torch,*args),core(torch,*args,legacy=True),(.002,1e-8,2e-5,1e-10,1e-8))
            checks.append({'name':'exact_recomposition','batch':batch,**parity})
            if not parity['valid']:raise RuntimeError('Exact vectorized algebra changed the reference')
            funcs={};graphs=[]
            for name,op in operators.items():
                def fn(op=op,name=name):return core(torch,*args,operator=op,legacy=name=='legacy_algebra')
                eager=fn();compiled=torch.compile(fn,fullgraph=True)
                actual=compiled();torch.cuda.synchronize()
                check=p2.differences(torch,actual,eager,(.002,1e-8,2e-5,1e-10,1e-8));check.update(name=name,batch=batch,mode='compiled');checks.append(check)
                if not check['valid']:raise RuntimeError(f'Compiled mismatch {name} b{batch}')
                funcs[('compiled',name)]=compiled
                for _ in range(spec['execution']['warmup']):compiled()
                torch.cuda.synchronize()
                try:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):compiled()
                    stream.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):captured=compiled()
                    graph.replay();torch.cuda.synchronize()
                    check=p2.differences(torch,captured,eager,(.002,1e-8,2e-5,1e-10,1e-8));check.update(name=name,batch=batch,mode='cuda_graph');checks.append(check)
                    if not check['valid']:raise RuntimeError('Graph mismatch')
                    graphs.append((graph,captured));funcs[('cuda_graph',name)]=graph.replay
                except Exception as e:failures.append({'name':name,'batch':batch,'error':str(e)})
            times=p2.benchmark(torch,funcs,spec['execution'])
            for (mode,name),timing in times.items():
                control=times.get((mode,'vectorized_algebra'))
                rows.append({'batch':batch,'mode':mode,'name':name,'timings':timing,
                             'saving_vs_vectorized':paired_summary(control['wall_ms'],timing['wall_ms']) if control else None})
            print(f'[GIADA IE-P3] benchmark batch={batch} complete',flush=True)
    qualified={}
    for k in range(6):
        name=f'learned_width{16 if k<3 else 32}_seed{spec["seeds"][k%3]}'
        d=diagnostics[name];q=spec['quality'];f=fresh[k]
        qualified[name]=bool(f['finite'] and f['rmse']<=q['aperture_rmse'] and f['max_abs']<=q['aperture_max_abs'] and
            d['voltage']['rmse']<=q['diagnostic_voltage_rmse_mv'] and d['voltage']['max_abs']<=q['diagnostic_voltage_max_abs_mv'] and
            d['calcium']['max_abs']<=q['diagnostic_calcium_max_abs_mm'])
    report={'valid':True,'code_revision':revision,'scope':spec['scope'],'selection':freeze,'fresh_atomic':fresh,
            'quality_qualified':qualified,'device':torch.cuda.get_device_name(),'torch':torch.__version__,
            'historical_diagnostics':diagnostics,'rows':rows,'preflights':checks,'graph_failures':failures,
            'candidate_promoted':False,'deployment_authorized':False,'exclusions':spec['exclusions']}
    (output/'final_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
