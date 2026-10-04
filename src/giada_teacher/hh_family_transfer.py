"""Original roadmap Task18: independent two-gate HH mechanisms, no weight sharing."""
from __future__ import annotations
import hashlib
import json
import subprocess
import shutil
import time
import sys
from pathlib import Path
import numpy as np

CHANNELS = ('Ca_HVA', 'NaTa_t', 'K_Pst')
TEACHER_REVISION = '074c4666300a8ad246601dab179a97a6942f0f29'

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rates(channel, voltage):
    """Independent transcription; canonical singular perturbations and fixed qt preserved."""
    v = np.asarray(voltage, dtype=np.float64).copy()
    qt = 2.3 ** 1.3
    if channel == 'Ca_HVA':
        v = np.where(v == -27, v + .0001, v)
        a = .055 * (-27-v) / np.expm1((-27-v)/3.8)
        b = .94*np.exp((-75-v)/17)
        ha = .000457*np.exp((-13-v)/50)
        hb = .0065/(np.exp((-v-15)/28)+1)
        inf = np.stack((a/(a+b),ha/(ha+hb)), -1)
        tau = np.stack((1/(a+b),1/(ha+hb)), -1)
    elif channel == 'NaTa_t':
        v = np.where(v == -38, v+.0001, v)
        a = .182*(v+38)/(-np.expm1(-(v+38)/6))
        b = .124*(-v-38)/(-np.expm1(-(-v-38)/6))
        mi, mt = a/(a+b), 1/(a+b)/qt
        v = np.where(v == -66, v+.0001, v)
        a = -.015*(v+66)/(-np.expm1((v+66)/6))
        b = -.015*(-v-66)/(-np.expm1((-v-66)/6))
        inf = np.stack((mi,a/(a+b)), -1)
        tau = np.stack((mt,1/(a+b)/qt), -1)
    elif channel == 'K_Pst':
        v += 10
        mi = 1/(1+np.exp(-(v+1)/12))
        mt = np.where(v < -50,1.25+175.03*np.exp(v*.026),1.25+13*np.exp(-v*.026))/qt
        hi = 1/(1+np.exp((v+54)/11))
        ht = (360+(1010+24*(v+55))*np.exp(-((v+75)/48)**2))/qt
        inf,tau = np.stack((mi,hi),-1),np.stack((mt,ht),-1)
    else:
        raise ValueError(channel)
    if not np.isfinite(inf).all() or not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError('invalid rates')
    return inf,tau

def update(state, inf, tau, dt):
    z = -np.expm1(-np.asarray(dt)[...,None]/tau)
    return (1-z)*state+z*inf

def native_audit(teacher, output):
    """Only diagnostic RANGE exposure changes; original equations remain byte-for-byte."""
    from neuron import h, load_mechanisms
    import neuron
    if neuron.__version__.split('+')[0] != '8.2.7':
        raise RuntimeError('NEURON version mismatch')
    revision=subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'],text=True).strip()
    if revision!=TEACHER_REVISION:raise RuntimeError('canonical teacher revision mismatch')
    root = Path(output)/'native'; root.mkdir()
    hashes = {}
    for channel in CHANNELS:
        source = Path(teacher)/'L5PC_NEURON_simulation/mods'/f'{channel}.mod'
        raw = source.read_text()
        instrumented = raw.replace('SUFFIX '+channel, 'SUFFIX '+channel+'\n RANGE mInf, mTau, hInf, hTau',1)
        if instrumented == raw: raise RuntimeError('instrumentation failed')
        (root/source.name).write_text(instrumented)
        hashes[channel] = {'canonical_mod_sha256':sha(source),'instrumented_mod_sha256':sha(root/source.name)}
    command = shutil.which('nrnivmodl') or str(Path(sys.executable).parent/'nrnivmodl')
    if not Path(command).is_file(): raise RuntimeError('nrnivmodl missing')
    with (root/'compile.log').open('w') as log:
        subprocess.run([command,str(root.resolve())],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    load_mechanisms(str(root.resolve()))
    h.CVode().active(0); h.secondorder=0
    rows=[]
    for channel in CHANNELS:
        sec=h.Section(name='audit_'+channel);sec.L=sec.diam=10;sec.insert(channel)
        mech=getattr(sec(.5),channel)
        setattr(mech,{'Ca_HVA':'gCa_HVAbar','NaTa_t':'gNaTa_tbar','K_Pst':'gK_Pstbar'}[channel],0)
        worst_rate=worst_step=worst_v=0.
        for v in sorted(set(np.linspace(-135,75,85).tolist()+[-66,-60,-38,-27])):
            inf,tau=rates(channel,v)
            h.finitialize(v)
            observed=np.array([mech.mInf,mech.hInf,mech.mTau,mech.hTau])
            expected=np.r_[inf,tau]
            worst_rate=max(worst_rate,float(np.max(abs(observed-expected)/(1+abs(expected)))))
            for dt in (.025,.1,1,25,100):
                h.dt=dt;h.finitialize(v);mech.m=.31;mech.h=.79
                h.fadvance()
                target=update(np.array([.31,.79]),inf,tau,dt)
                worst_step=max(worst_step,float(np.max(abs(np.array([mech.m,mech.h])-target))))
                worst_v=max(worst_v,abs(float(sec(.5).v)-v))
        rows.append({'channel':channel,'rate_scaled_max_error':worst_rate,'gate_max_error':worst_step,'voltage_drift_mv':worst_v})
        h.delete_section(sec=sec)
    report={'valid':all(r['rate_scaled_max_error']<1e-7 and r['gate_max_error']<1e-7 and r['voltage_drift_mv']<1e-8 for r in rows),'rows':rows,'source_hashes':hashes,'temperature_policy':'canonical fixed 34C qt for Na/K; Ca_HVA unchanged'}
    write(Path(output)/'native_audit.json',report)
    if not report['valid']: raise RuntimeError('independent native/formula oracle failed')
    return report

def data(seed,count,domain=(-135,75)):
    rng=np.random.default_rng(seed)
    v=rng.uniform(*domain,count); state=rng.uniform(0,1,(count,2))
    dt=rng.choice([.025,.1,.5,1,5,25,100],count)
    return np.c_[v,state,dt]

def targets(x):
    rr=[rates(c,x[:,0]) for c in CHANNELS]
    return np.stack([r[0] for r in rr]),np.stack([r[1] for r in rr]),np.stack([update(x[:,1:3],*r,x[:,3]) for r in rr])

def model_factory(torch,width,descriptors):
    """Leading axis is independent channel/objective/seed, never shared weights."""
    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__();n=len(descriptors)
            for name,shape in [('w1',(n,width,1)),('b1',(n,width)),('w2',(n,width,width)),('b2',(n,width)),('wo',(n,4,width)),('bo',(n,4))]:
                value=torch.zeros(shape)
                if name.startswith('w'):
                    for i,(_,_,seed) in enumerate(descriptors):
                        generator=torch.Generator().manual_seed(seed+{'w1':0,'w2':100,'wo':200}[name])
                        value[i].normal_(0,.1,generator=generator)
                setattr(self,name,torch.nn.Parameter(value))
        def forward(self,x):
            f=((x[...,0]+30)/100).unsqueeze(-1)
            f=torch.nn.functional.silu(torch.bmm(f,self.w1.transpose(1,2))+self.b1[:,None])
            f=torch.nn.functional.silu(torch.bmm(f,self.w2.transpose(1,2))+self.b2[:,None])
            raw=torch.bmm(f,self.wo.transpose(1,2))+self.bo[:,None]
            inf=torch.sigmoid(raw[...,:2]);tau=torch.exp(raw[...,2:].clamp(-12,12))
            z=-torch.expm1(-x[...,3,None]/tau)
            return (1-z)*x[...,1:3]+z*inf,inf,tau
    return Ensemble()

def measurements(x,pred,inf,tau,channel):
    ti,tt=rates(channel,x[:,0]); truth=update(x[:,1:3],ti,tt,x[:,3]); power=3 if channel=='NaTa_t' else 2
    return {'gate_rmse':float(np.sqrt(np.mean((pred-truth)**2))), 'gate_max_error':float(abs(pred-truth).max()),
            'inf_rmse':float(np.sqrt(np.mean((inf-ti)**2))),'log_tau_rmse':float(np.sqrt(np.mean((np.log(tau)-np.log(tt))**2))),
            'open_rmse':float(np.sqrt(np.mean((pred[:,0]**power*pred[:,1]-truth[:,0]**power*truth[:,1])**2))),
            'occupancy_violations':int(np.sum((pred<0)|(pred>1))), 'finite':bool(np.isfinite(pred).all() and np.isfinite(tau).all())}

def passes(m):
    return m['finite'] and m['occupancy_violations']==0 and m['gate_rmse']<=.001 and m['gate_max_error']<=.01 and m['inf_rmse']<=.002 and m['log_tau_rmse']<=.02 and m['open_rmse']<=.002

def run_training(output,config,revision):
    from .gpu_baseline_runtime import configure_torch_runtime,environment_manifest
    from .joint_gate_symmetric_confirmation import _clip_per_seed
    torch=configure_torch_runtime(17)
    if config.get('require_cuda',True) and not torch.cuda.is_available():raise RuntimeError('registered Task18 requires CUDA')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    output=Path(output)
    fit=data(180101,config['pool_size']);dev=data(180201,2048)
    np.savez_compressed(output/'fit_development.npz',fit=fit,development=dev)
    fi,ft,fy=targets(fit)
    descriptors=[(c,o,s) for c in CHANNELS for o in ('transition_only','rate_supervised') for s in config['seeds']]
    ci=torch.tensor([CHANNELS.index(c) for c,_,_ in descriptors],device=device)
    x=torch.tensor(fit,dtype=torch.float32,device=device)
    ti=torch.tensor(fi,dtype=torch.float32,device=device)[ci];tt=torch.tensor(ft,dtype=torch.float32,device=device)[ci];ty=torch.tensor(fy,dtype=torch.float32,device=device)[ci]
    objective=torch.tensor([o=='rate_supervised' for _,o,_ in descriptors],device=device,dtype=torch.float32)
    models={w:model_factory(torch,w,descriptors).to(device) for w in config['widths']}
    optim={w:torch.optim.Adam(m.parameters(),lr=.003) for w,m in models.items()}
    # Mathematical vectorization equivalence includes Adam and independent clipping.
    preflight=[]
    probe=x[:32][None].expand(len(descriptors),-1,-1)
    for w,model in models.items():
        single=model_factory(torch,w,[descriptors[0]]).to(device)
        single.load_state_dict({k:v[:1].detach().clone() for k,v in model.state_dict().items()})
        a=model(probe)[0][0];b=single(probe[:1])[0][0]
        error=float((a-b).detach().abs().max())
        ma=model_factory(torch,w,descriptors).to(device);ma.load_state_dict(model.state_dict())
        mb=model_factory(torch,w,[descriptors[0]]).to(device);mb.load_state_dict(single.state_dict())
        oa=torch.optim.Adam(ma.parameters(),lr=.003);ob=torch.optim.Adam(mb.parameters(),lr=.003)
        ma(probe)[0].square().mean((1,2)).sum().backward();mb(probe[:1])[0].square().mean().backward()
        _clip_per_seed(ma,torch,1.);_clip_per_seed(mb,torch,1.);oa.step();ob.step()
        pe=max(float((v[0]-mb.state_dict()[k][0]).detach().abs().max()) for k,v in ma.state_dict().items())
        preflight.append({'width':w,'prediction_error':error,'adam_step_error':pe})
    if any(max(r['prediction_error'],r['adam_step_error'])>1e-5 for r in preflight):raise RuntimeError('ensemble equivalence failed')
    write(output/'equivalence_preflight.json',{'valid':True,'rows':preflight,'independent_adam_and_clipping':True})
    best={};ladder=[];start=time.monotonic();rng=np.random.default_rng(180301)
    def evaluate(model,values):
        inp=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(len(descriptors),-1,-1)
        with torch.no_grad(): result=[v.detach().cpu().double().numpy() for v in model(inp)]
        return [measurements(values,*(r[i] for r in result),c) for i,(c,_,_) in enumerate(descriptors)]
    for step in range(config['checkpoints'][-1]+1):
        if step in config['checkpoints']:
            for w,model in models.items():
                metrics=evaluate(model,dev)
                torch.save(model.state_dict(),output/f'checkpoint_w{w}_step{step}.pt')
                for i,(c,o,s) in enumerate(descriptors):
                    m=metrics[i];score=m['gate_rmse']/.001+m['inf_rmse']/.002+m['log_tau_rmse']/.02
                    row={'channel':c,'objective':o,'seed':s,'width':w,'step':step,'score':score,'metrics':m};ladder.append(row)
                    key=f'{c}/{o}/{s}'
                    if key not in best or score<best[key]['score']:best[key]=dict(row,index=i)
            write(output/'development_ladder.json',ladder)
            print(f'[GIADA Task18] checkpoint {step}/{config["checkpoints"][-1]} elapsed={(time.monotonic()-start)/60:.1f}min',flush=True)
        if step==config['checkpoints'][-1]:break
        idx=torch.tensor(rng.integers(0,len(fit),config['batch_size']),device=device)
        batch=x[idx][None].expand(len(descriptors),-1,-1)
        for w,model in models.items():
            optim[w].zero_grad(set_to_none=True)
            p,i,t=model(batch)
            losses=(p-ty[:,idx]).square().mean((1,2))
            losses=losses+objective*(.1*(i-ti[:,idx]).square().mean((1,2))+.01*(t.log()-tt[:,idx].log()).square().mean((1,2)))
            losses.sum().backward();_clip_per_seed(model,torch,1.);optim[w].step()
    freeze={'selection':best,'fresh_accessed':False,'config':config,'code_revision':revision,'checkpoints':{p.name:sha(p) for p in output.glob('checkpoint_*.pt')}}
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    write(output/'selection_freeze.json',freeze)
    fresh={'in_support':data(180401,4096),'voltage_extrapolation':data(180402,2048,(-155,-135)), 'positive_extrapolation':data(180403,2048,(75,95))}
    np.savez_compressed(output/'fresh.npz',**fresh)
    results=[];cache={}
    for key,row in best.items():
        w,step=row['width'],row['step'];ck=(w,step)
        if ck not in cache:
            path=output/f'checkpoint_w{w}_step{step}.pt'
            if sha(path)!=freeze['checkpoints'][path.name]:raise RuntimeError('frozen checkpoint changed')
            model=models[w];model.load_state_dict(torch.load(output/f'checkpoint_w{w}_step{step}.pt',map_location=device,weights_only=True))
            cache[ck]={name:evaluate(model,values) for name,values in fresh.items()}
        metrics={name:rr[row['index']] for name,rr in cache[ck].items()}
        results.append({'channel':row['channel'],'objective':row['objective'],'seed':row['seed'],'width':w,'step':step,'metrics':metrics,'in_support_passed':passes(metrics['in_support'])})
    # Numeric baselines: same input tuples; LUT interpolation never extrapolates silently.
    numerical=[]
    grid=np.linspace(-135,75,2049)
    for c in CHANNELS:
        ii,tt=rates(c,grid)
        for name,values in fresh.items():
            if name!='in_support':continue
            inf=np.stack([np.interp(values[:,0],grid,ii[:,g]) for g in range(2)],-1)
            tau=np.stack([np.interp(values[:,0],grid,tt[:,g]) for g in range(2)],-1)
            m=measurements(values,update(values[:,1:3],inf,tau,values[:,3]),inf,tau,c)
            numerical.append({'channel':c,'family':'linear_lut_2049_f64','domain':name,'metrics':m,'passed':passes(m)})
    write(output/'fresh_metrics.json',{'learned':results,'numerical':numerical})
    supported={c:all(r['in_support_passed'] for r in results if r['channel']==c and r['objective']=='rate_supervised') for c in CHANNELS}
    report={'schema_version':'giada-roadmap-task18-v1','valid':True,'transfer_passed':all(supported.values()),'per_channel_transfer':supported,'task19_authorized':all(supported.values()),'fresh_used_for_selection':False,'learned':results,'numerical':numerical,'environment':environment_manifest(torch),'training_seconds':time.monotonic()-start,'scope':'isolated held-voltage two-gate kinetics, not embedded causal substitution or voltage-path sufficiency','hardware_speedup_claimed':False,'shared_weights':False}
    write(output/'final_report.json',report)
    return report
