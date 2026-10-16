"""IE-P5: matched accuracy repair, coefficient attribution and neural tabulation."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from . import ie_discrete_ionic as p4
from . import ie_readout_comparison as p3
from . import cc_n1_contract_memory as n1

CONFIG='experiments/ie_p5_accuracy_execution_preregistration_v1.json'
MEMBERS=tuple((loss,seed) for loss in ('internal','balanced') for seed in (17,29,43))
GROUPS={'calcium':(0,1,2,3),'sodium':(4,5,6,7,8,9),'potassium':(12,13,14,15,16),'ih_im':(10,11)}


class Progress:
    def __init__(self,target):self.target=target
    def write(self,value):return self.target.write(value.replace('[GIADA IE-P4]','[GIADA IE-P5]'))
    def flush(self):return self.target.flush()


def ensemble(torch,parent,device):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ('a','b','c','d'):
                self.register_parameter(name,torch.nn.Parameter(parent[name][[9,10,11,9,10,11]].to(device).clone()))
            self.register_buffer('scale',parent['scale'].to(device).clone())
        def forward(self,v,logits=False):
            raw=torch.bmm(torch.nn.functional.silu(torch.bmm((v[...,None]+40)/80,self.a)+self.b),self.c)+self.d
            values=torch.sigmoid(raw);result=(values[...,:17],values[...,17:]*self.scale)
            return (*result,raw[...,:17]) if logits else result
    return Model()


def operator(torch,model,k):
    a,b,c,d=[getattr(model,n)[k].detach().clone() for n in ('a','b','c','d')];scale=model.scale.detach().clone()
    def op(v):
        raw=torch.nn.functional.silu(((v[...,None]+40)/80)@a+b)@c+d
        values=torch.sigmoid(raw)
        return values[...,:17],values[...,17:]*scale
    return op


def tabulate(torch,op,spacing,device):
    knots=torch.linspace(-100,60,round(160/spacing)+1,device=device)
    with torch.no_grad():table=torch.cat(op(knots),-1).detach().clone()
    def compiled(v):
        position=(v.clamp(-100,60)+100)/spacing
        index=position.floor().long().clamp(0,len(knots)-2)
        f=(position-index)[...,None]
        value=(1-f)*table[index]+f*table[index+1]
        return value[...,:17],value[...,17:]
    return compiled,table


def swap(torch,op,kind):
    def intervention(v):
        u,z=op(v);ut,zt=p4.coefficients(torch,v)
        if kind=='u_all':return ut,z
        if kind=='z_all':return u,zt
        mask=torch.zeros(17,dtype=torch.bool,device=v.device);mask[list(GROUPS[kind])]=True
        return torch.where(mask,ut,u),torch.where(mask,zt,z)
    return intervention


def clip(torch,model):
    norms=torch.stack([p.grad.flatten(1).square().sum(-1) for p in model.parameters()]).sum(0).sqrt()
    for p in model.parameters():p.grad.mul_((1/norms.clamp_min(1)).reshape(6,*([1]*(p.ndim-1))))


def endpoints(torch,u,z):
    persistence=(1-z).pow(400)
    zero=(1-persistence)*u
    return torch.stack((zero,persistence+zero),-1)


def preflight(torch,parent,device):
    model=ensemble(torch,parent,device);gen=torch.Generator(device=device).manual_seed(81800)
    v=torch.rand(6,47,device=device,generator=gen)*160-100
    reference=p4.ensemble(torch,parent['scale'].to(device),device);reference.load_state_dict({k:x.to(device) for k,x in parent.items()})
    maximum=0.
    for k in range(6):
        source=p4.operator(torch,reference,9+k%3)(v[k]);actual=operator(torch,model,k)(v[k])
        maximum=max(maximum,max(float((a-b).abs().max()) for a,b in zip(source,actual)))
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    target=torch.rand(6,47,17,device=device,generator=gen)
    opt=torch.optim.Adam(model.parameters(),lr=.003);u,z=model(v)
    ((u-target).square().mean((1,2))+z.square().mean((1,2))).sum().backward();clip(torch,model);opt.step()
    for k in range(6):
        iso=ensemble(torch,parent,device)
        with torch.no_grad():
            for n,p in iso.named_parameters():p.copy_(initial[n])
        oi=torch.optim.Adam(iso.parameters(),lr=.003);u,z=iso(v)
        ((u[k]-target[k]).square().mean()+z[k].square().mean()).backward();clip(torch,iso);oi.step()
        for n,p in iso.named_parameters():maximum=max(maximum,float((p[k]-dict(model.named_parameters())[n][k]).abs().max()))
    if maximum>1e-5:raise RuntimeError(f'Warm-start/Adam mismatch {maximum}')
    return {'valid':True,'warm_start_and_adam_max_error':maximum,'atol':1e-5,'adam_reset_in_both_arms':True}


def train(torch,parent,spec,output):
    model=ensemble(torch,parent,'cuda');gen=torch.Generator(device='cuda').manual_seed(spec['data_seed'])
    fit=torch.rand(spec['fit_count'],device='cuda',generator=gen)*160-100
    dev=torch.rand(spec['development_count'],device='cuda',generator=gen)*160-100
    xf=torch.rand(len(fit),17,device='cuda',generator=gen);xd=torch.rand(len(dev),17,device='cuda',generator=gen)
    uf,zf=p4.coefficients(torch,fit);ud,zd=p4.coefficients(torch,dev);scale=model.scale
    truth=(1-zd)*xd+zd*ud;long_truth=endpoints(torch,ud,zd)
    opt=torch.optim.Adam(model.parameters(),lr=spec['learning_rate'])
    balanced=torch.tensor([0,0,0,1,1,1],device='cuda')
    best=[float('inf')]*6;states=[None]*6;chosen=[None]*6;curves=[]
    for step in range(spec['training_steps']+1):
        if step in spec['checkpoints']:
            with torch.no_grad():
                u,z=model(dev[None].expand(6,-1));diff=(1-z)*xd+z*u-truth
                rmse=diff.square().mean((1,2)).sqrt();maxerr=diff.abs().amax((1,2))
                longerr=(endpoints(torch,u,z)-long_truth).abs().amax((1,2,3))
                scores=torch.stack((rmse/.001,maxerr/.01,longerr/.03)).amax(0).tolist()
            curves.append({'step':step,'development_score':scores,'transition_rmse':rmse.tolist(),'transition_max':maxerr.tolist(),'constant400_max':longerr.tolist()})
            torch.save(model.state_dict(),output/f'checkpoint_{step}.pt')
            (output/'training_curves.json').write_text(json.dumps(curves,indent=2))
            for k,s in enumerate(scores):
                if not np.isfinite(s):raise RuntimeError('Nonfinite development score')
                if s<best[k]:best[k]=s;chosen[k]=step;states[k]={n:p[k].detach().clone() for n,p in model.named_parameters()}
            print(f'[GIADA IE-P5] refit {step}/{spec["training_steps"]} dev score median={np.median(scores):.6g}',flush=True)
        if step==spec['training_steps']:break
        ids=torch.randint(len(fit),(spec['batch_size'],),device='cuda',generator=gen)
        u,z,logits=model(fit[ids][None].expand(6,-1),True);x=xf[ids];target=(1-zf[ids])*x+zf[ids]*uf[ids]
        err=(((1-z)*x+z*u-target)/scale).square()
        base=err.mean((1,2))+(u-uf[ids]).square().mean((1,2))+.01*(torch.log(z/scale+1e-8)-torch.log(zf[ids]/scale+1e-8)).square().mean((1,2))
        logit_target=torch.logit(uf[ids].clamp(1e-6,1-1e-6))
        extra=.01*(logits-logit_target).square().mean((1,2))+(endpoints(torch,u,z)-endpoints(torch,uf[ids],zf[ids])).square().mean((1,2,3))+.1*err.mean(1).amax(-1)
        opt.zero_grad();(base+balanced*extra).sum().backward();clip(torch,model);opt.step()
    with torch.no_grad():
        for k in range(6):
            for n,p in model.named_parameters():p[k].copy_(states[k][n])
    torch.save(model.state_dict(),output/'frozen_checkpoint.pt')
    freeze={'members':MEMBERS,'selected_steps':chosen,'development_scores':best,'fresh_accessed':False,
            'checkpoint_sha256':hashlib.sha256((output/'frozen_checkpoint.pt').read_bytes()).hexdigest()}
    (output/'selection_freeze.json').write_text(json.dumps(freeze,indent=2))
    return model,freeze


def path_data():
    rng=np.random.default_rng(81802);t=np.arange(400)*.1;paths=[];families=[]
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
    v=torch.rand(spec['fresh_count'],device='cuda',generator=gen)*160-100
    x=torch.rand(len(v),17,device='cuda',generator=gen);ut,zt=p4.coefficients(torch,v);target=(1-zt)*x+zt*ut
    pv,x0,families=path_data();vv=torch.tensor(pv,device='cuda',dtype=torch.float32);initial=torch.tensor(x0,device='cuda',dtype=torch.float32)
    def rollout(u,z):
        state=initial;values=[]
        for k in range(400):state=(1-z[k])*state+z[k]*u[k];values.append(state)
        return torch.stack(values)
    reference=rollout(*p4.coefficients(torch,vv));report={}
    np.savez_compressed(output/'fresh_imposed_paths.npz',voltage_mv=pv,initial_gates=x0,families=families)
    for name,op in ops.items():
        u,z=p4.coefficients(torch,v) if op is None else op(v)
        prediction=rollout(*(p4.coefficients(torch,vv) if op is None else op(vv)))
        by_family={f:p3.errors(torch,prediction[:,[i for i,g in enumerate(families) if g==f]],reference[:,[i for i,g in enumerate(families) if g==f]]) for f in sorted(set(families))}
        full=p4.restore(torch,prediction,torch.zeros(400,24,18,device='cuda'));rf=p4.restore(torch,reference,torch.zeros_like(full))
        gbar=torch.tensor(p4.ionic.GBAR,dtype=full.dtype,device='cuda');rev=torch.tensor(p4.ionic.REVERSALS,dtype=full.dtype,device='cuda')
        current=p3.exact_apertures(torch,full)*gbar*(vv[...,None]-rev);rc=p3.exact_apertures(torch,rf)*gbar*(vv[...,None]-rev)
        report[name]={'fresh_gate':p3.errors(torch,(1-z)*x+z*u,target),'forced_path_gate':p3.errors(torch,prediction,reference),'families':by_family,
            'gate_rmse_by_state':(prediction-reference).square().mean((0,1)).sqrt().tolist(),
            'current_rmse_by_channel':(current-rc).square().mean((0,1)).sqrt().tolist(),
            'charge_max_abs_by_channel':((current-rc).sum(0)*.1).abs().max(0).values.tolist(),
            'inf_probe':p3.errors(torch,u,ut),'log_z_probe':p3.errors(torch,torch.log(z+1e-12),torch.log(zt+1e-12)),
            'occupancy_violations':int(((prediction<0)|(prediction>1)|~torch.isfinite(prediction)).sum())}
    return report


def run(root,output,revision):
    import torch
    root,output=Path(root),Path(output);spec=json.loads((root/CONFIG).read_text())
    assert hashlib.sha256((root/spec['parent_checkpoint']).read_bytes()).hexdigest()==spec['parent_sha256']
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch._dynamo.config.cache_size_limit=max(256,torch._dynamo.config.cache_size_limit)
    parent=torch.load(root/spec['parent_checkpoint'],map_location='cpu',weights_only=True)
    _,cfg,calcium,_,base=n1.contract(root);assert cfg['dt_ms']==spec['dt_ms']
    validation=preflight(torch,parent,'cuda');(output/'validation.json').write_text(json.dumps(validation,indent=2))
    original=ensemble(torch,parent,'cuda');model,freeze=train(torch,parent,spec,output)
    ops={'formula_vectorized':None,'lut025':p4.lut_operator(torch)};frozen={}
    for k,seed in enumerate(spec['seeds']):
        frozen[f'frozen64_seed{seed}']=operator(torch,original,k)
    ops.update(frozen);conversions={};compiled_tables={};grid=torch.linspace(-100,60,65537,device='cuda')
    with torch.inference_mode():
        for k,(loss,seed) in enumerate(MEMBERS):
            name=f'refit_{loss}_seed{seed}';op=operator(torch,model,k);ops[name]=op
            u,z=op(grid)
            for spacing,label in ((.25,'table025'),(1.,'table1')):
                table_op,table=tabulate(torch,op,spacing,'cuda');key=name+'_'+label;ops[key]=table_op
                a,b=table_op(grid)
                delta=max(float((b*a-z*u).abs().max()),float(((1-b)+b*a-(1-z)-z*u).abs().max()))
                conversions[key]={'parent':name,'spacing_mv':spacing,'table_bytes':table.numel()*table.element_size(),
                    'dense_parameter_bytes':(64*36+34)*4+17*4,'maximum_endpoint_map_delta':delta,'conversion_passed':delta<=spec['conversion_gate_max_delta']}
                compiled_tables[key]=table.cpu()
        torch.save(compiled_tables,output/'neural_deployment_tables.pt')
        (output/'conversion_report.json').write_text(json.dumps(conversions,indent=2))
        f=fresh(torch,ops,spec,output);(output/'fresh_evaluation.json').write_text(json.dumps(f,indent=2,allow_nan=False))
        # Presentation-only renaming of reused, unchanged IE-P4 evaluation helpers.
        with contextlib.redirect_stdout(Progress(sys.stdout)):
            history=p4.historical(torch,root,ops,cfg,calcium,base)
            (output/'historical_evaluation.json').write_text(json.dumps(history,indent=2,allow_nan=False))
            interventions={name+'_'+kind:swap(torch,op,kind) for name,op in frozen.items() for kind in ('u_all','z_all',*GROUPS)}
            attribution=p4.historical(torch,root,interventions,cfg,calcium,base)
            (output/'diagnostic_oracle_attribution.json').write_text(json.dumps({'selection_eligible':False,'metrics':attribution},indent=2))
            rows,checks,failures=p4.timing(torch,root,ops,spec,cfg,calcium,base,output)
    qualified={};q=spec['quality']
    for name,a in f.items():
        h=history[name]
        qualified[name]=bool(a['fresh_gate']['rmse']<=q['fresh_gate_rmse'] and a['fresh_gate']['max_abs']<=q['fresh_gate_max_abs'] and
            all(m['rmse']<=q['forced_path_gate_rmse'] and m['max_abs']<=q['forced_path_gate_max_abs'] for m in a['families'].values()) and
            h['voltage']['rmse']<=q['historical_voltage_rmse_mv'] and h['voltage']['max_abs']<=q['historical_voltage_max_abs_mv'] and
            h['calcium']['max_abs']<=q['historical_calcium_max_abs_mm'] and h['voltage_domain_exit_count']==0 and a['occupancy_violations']==0 and
            (name not in conversions or conversions[name]['conversion_passed']))
    report={'valid':True,'schema_version':spec['schema_version'],'code_revision':revision,'device':torch.cuda.get_device_name(),'torch':torch.__version__,
        'validation':validation,'selection':freeze,'fresh':f,'historical':history,'quality_qualified':qualified,'conversions':conversions,
        'diagnostic_oracles_selection_eligible':False,'rows':rows,'preflights':checks,'graph_failures':failures,
        'candidate_promoted':False,'deployment_authorized':False,'exclusions':spec['exclusions']}
    (output/'final_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('[GIADA IE-P5] complete; tabulation is numerical deployment, not a new neural architecture',flush=True)
    return report
