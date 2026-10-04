"""Original Task24: three canonical sodium channels at imposed voltage."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from itertools import product
import numpy as np
from .hh_family_transfer import rates as hh_rates, update, write, sha, TEACHER_REVISION
from .joint_gate_symmetric_confirmation import _clip_per_seed

CHANNELS = ('NaTa_t', 'NaTs2_t', 'Nap_Et2')
FAMILIES = ('independent', 'shared_heads', 'conditioned')
DT = (.025, .1, .5, 1., 5., 25., 100., 500., 1000., 5000., 10000.)
SEAMS = (-38., -66., -32., -60., -17., -64.4)


def rates(channel, voltage):
    if channel == 'NaTa_t':
        return hh_rates(channel, voltage)
    if channel not in CHANNELS:
        raise ValueError(channel)
    v = np.asarray(voltage, dtype=np.float64).copy()
    qt = 2.3 ** 1.3
    if channel == 'NaTs2_t':
        v = np.where(v == -32., v + .0001, v)
        a = .182*(v+32)/(-np.expm1(-(v+32)/6))
        b = .124*(-v-32)/(-np.expm1(-(-v-32)/6))
        mi, mt = a/(a+b), 1/(a+b)/qt
        v = np.where(v == -60., v + .0001, v)
        a = -.015*(v+60)/(-np.expm1((v+60)/6))
        b = -.015*(-v-60)/(-np.expm1((-v-60)/6))
        hi, ht = a/(a+b), 1/(a+b)/qt
    else:
        # Canonical ordering matters: mInf precedes the -38mV perturbation.
        mi = 1/(1+np.exp((v+52.6)/-4.6))
        v = np.where(v == -38., v + .0001, v)
        a = .182*(v+38)/(-np.expm1(-(v+38)/6))
        b = .124*(-v-38)/(-np.expm1(-(-v-38)/6))
        mt = 6/(a+b)/qt  # preserve actual MOD, not the ambiguous comment
        v = np.where((v == -17.) | (v == -64.4), v + .0001, v)
        hi = 1/(1+np.exp((v+48.8)/10))
        a = -2.88e-6*(v+17)/(-np.expm1((v+17)/4.63))
        b = 6.94e-6*(v+64.4)/(-np.expm1(-(v+64.4)/2.63))
        ht = 1/(a+b)/qt
    return np.stack((mi,hi),-1), np.stack((mt,ht),-1)


def data(seed, count, domain=(-135., 75.)):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(*domain, count), rng.uniform(0, 1, (count, 6)), rng.choice(DT, count)]


def state_grid(fresh=False):
    size, offset = (121, .37) if fresh else (61, .5)
    v = -135 + 210*(np.arange(size)+offset)/size
    if fresh:
        v = np.r_[v, -135., 75., SEAMS]
    states = np.array(list(product((0.,1.), repeat=6)))
    vv, ss, dd = np.meshgrid(v, np.arange(64), DT, indexing='ij')
    return np.c_[vv.ravel(), states[ss.ravel()], dd.ravel()]


def targets(values):
    rr = [rates(c, values[:,0]) for c in CHANNELS]
    inf, tau = np.stack([r[0] for r in rr]), np.stack([r[1] for r in rr])
    state = values[:,1:7].reshape(-1,3,2).transpose(1,0,2)
    return inf, tau, update(state, inf, tau, values[:,7])


def model_factory(torch, width, family, seeds):
    if family not in FAMILIES:
        raise ValueError(family)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n, ind = len(seeds), family == 'independent'
            shapes = {'w1': (n,3,width,1) if ind else (n,width,4 if family=='conditioned' else 1),
                      'b1': (n,3,width) if ind else (n,width),
                      'w2': (n,3,width,width) if ind else (n,width,width),
                      'b2': (n,3,width) if ind else (n,width),
                      'wo': (n,3,4,width) if family!='conditioned' else (n,4,width),
                      'bo': (n,3,4) if family!='conditioned' else (n,4)}
            for name, shape in shapes.items():
                value = torch.zeros(shape)
                if name.startswith('w'):
                    for s, seed in enumerate(seeds):
                        gen = torch.Generator().manual_seed(seed+{'w1':0,'w2':100,'wo':200}[name])
                        if ind:
                            sample = torch.randn(shape[2:],generator=gen)*.1
                            value[s] = sample[None].expand(3,*sample.shape)
                        elif name=='wo' and family=='shared_heads':
                            sample=torch.randn((4,width),generator=gen)*.1
                            value[s]=sample[None].expand(3,4,width)
                        elif name=='w1' and family=='conditioned':
                            value[s,:,0]=torch.randn((width,),generator=gen)*.1
                            value[s,:,1:]=torch.randn((width,3),generator=gen)*.1
                        else:
                            value[s].normal_(0,.1,generator=gen)
                setattr(self,name,torch.nn.Parameter(value))

        def forward(self,x,swap=False):
            n,b,_=x.shape
            v=((x[...,0]+30)/100).unsqueeze(-1)
            if family=='independent':
                f=v[:,None].expand(-1,3,-1,-1)
                f=torch.nn.functional.silu(torch.einsum('scbi,scwi->scbw',f,self.w1)+self.b1[:,:,None])
                f=torch.nn.functional.silu(torch.einsum('scbi,scwi->scbw',f,self.w2)+self.b2[:,:,None])
                raw=torch.einsum('scbw,scow->scbo',f,self.wo)+self.bo[:,:,None]
            elif family=='shared_heads':
                f=torch.nn.functional.silu(torch.bmm(v,self.w1.transpose(1,2))+self.b1[:,None])
                f=torch.nn.functional.silu(torch.bmm(f,self.w2.transpose(1,2))+self.b2[:,None])
                raw=torch.einsum('sbw,scow->scbo',f,self.wo)+self.bo[:,:,None]
            else:
                ids=torch.eye(3,device=x.device,dtype=x.dtype)
                if swap: ids=ids[[1,2,0]]
                f=torch.cat((v[:,None].expand(-1,3,-1,-1),ids[None,:,None].expand(n,-1,b,-1)),-1).reshape(n,3*b,4)
                f=torch.nn.functional.silu(torch.bmm(f,self.w1.transpose(1,2))+self.b1[:,None])
                f=torch.nn.functional.silu(torch.bmm(f,self.w2.transpose(1,2))+self.b2[:,None])
                raw=(torch.bmm(f,self.wo.transpose(1,2))+self.bo[:,None]).reshape(n,3,b,4)
            if swap and family!='conditioned':raw=raw[:,[1,2,0]]
            inf=torch.sigmoid(raw[...,:2]);tau=torch.exp(raw[...,2:].clamp(-12,12))
            state=x[...,1:7].reshape(n,b,3,2).transpose(1,2)
            z=-torch.expm1(-x[:,None,:,7,None]/tau)
            return (1-z)*state+z*inf,inf,tau
    return Model()


def current_metrics(values, predicted, truth):
    """Density mA/cm2, state at update endpoint, supplied endpoint V.

    Deterministic parameter panel: missing mechanisms and reversal crossings.
    Normalize by sum absolute fully-open driving currents, not teacher current.
    This avoids division by small opening probability or cancellation.
    """
    po=predicted[...,0]**3*predicted[...,1]
    to=truth[...,0]**3*truth[...,1]
    errors=[];normalized=[];individual=[];raw_individual=[]
    for multipliers in ((0.,1.,1.),(1.,0.,1.),(1.,1.,0.),(.25,1.,4.),(1.,1.,1.),(4.,1.,.25)):
        for reversals in ((55.,55.,55.),(25.,55.,85.),(85.,55.,25.)):
            driving=np.stack([1e-5*g*(values[:,0]-e) for g,e in zip(multipliers,reversals)])
            err=(po-to)*driving
            scale=np.maximum(abs(driving).sum(0),1e-12)
            errors.append(err.sum(0));normalized.append(err.sum(0)/scale)
            individual.append(err/np.maximum(abs(driving),1e-12))
            raw_individual.append(err)
    total=np.stack(errors);norm=np.stack(normalized);ind=np.stack(individual)
    return dict(individual_current_rmse_ma_cm2=float(np.sqrt(np.mean(np.stack(raw_individual)**2))),
                total_current_rmse_ma_cm2=float(np.sqrt(np.mean(total**2))),
                total_current_normalized_rmse=float(np.sqrt(np.mean(norm**2))),
                total_current_normalized_max=float(abs(norm).max()),
                individual_current_normalized_rmse=float(np.sqrt(np.mean(ind**2))))


def measurements(values,pred,inf,tau):
    ti,tt,truth=targets(values)
    out={}
    for k,c in enumerate(CHANNELS):
        e=pred[k]-truth[k];oi=pred[k,:,0]**3*pred[k,:,1];ot=truth[k,:,0]**3*truth[k,:,1]
        out[c]=dict(gate_rmse=float(np.sqrt(np.mean(e**2))),gate_max_error=float(abs(e).max()),
            m_rmse=float(np.sqrt(np.mean(e[:,0]**2))),h_rmse=float(np.sqrt(np.mean(e[:,1]**2))),
            inf_rmse=float(np.sqrt(np.mean((inf[k]-ti[k])**2))),
            log_tau_rmse=float(np.sqrt(np.mean((np.log(tau[k])-np.log(tt[k]))**2))),
            open_rmse=float(np.sqrt(np.mean((oi-ot)**2))),occupancy_violations=int(((pred[k]<0)|(pred[k]>1)).sum()),
            finite=bool(np.isfinite(pred[k]).all() and np.isfinite(inf[k]).all() and np.isfinite(tau[k]).all() and (tau[k]>0).all()))
    out['pair']=current_metrics(values,pred,truth)
    # All individual currents are measured explicitly as well as their sum.
    for k,c in enumerate(CHANNELS):
        e=(pred[k,:,0]**3*pred[k,:,1]-truth[k,:,0]**3*truth[k,:,1])*1e-5*(values[:,0]-55)
        out[c]['current_rmse_ma_cm2']=float(np.sqrt(np.mean(e**2)))
    return out


def evaluate(model,values,seeds,torch,device,swap=False):
    result=[[] for _ in seeds]
    # Chunk inference to bound device memory on state-extreme grids.
    for start in range(0,len(values),2048):
        x=torch.tensor(values[start:start+2048],dtype=torch.float32,device=device)[None].expand(len(seeds),-1,-1)
        with torch.no_grad(): out=[a.cpu().double().numpy() for a in model(x,swap)]
        for n in range(len(seeds)):result[n].append([a[n] for a in out])
    return [measurements(values,*(np.concatenate([r[k] for r in rows],axis=1) for k in range(3))) for rows in result]


def passes(metrics,gates,pair_gates):
    return all(metrics[c]['finite'] and all(metrics[c][k]<=t for k,t in gates.items()) for c in CHANNELS) and all(metrics['pair'][k]<=t for k,t in pair_gates.items())


def score(domains,config):
    scores = []
    for metrics in domains.values():
        scores.extend(metrics[c][k] / t for c in CHANNELS
                      for k, t in config['gates'].items() if t > 0)
        scores.extend(metrics['pair'][k] / t
                      for k, t in config['pair_gates'].items() if t > 0)
    return max(scores)


def equivalence(torch,width,family,seeds,device):
    model=model_factory(torch,width,family,seeds).to(device)
    x=torch.tensor(data(240001,24),dtype=torch.float32,device=device)[None].expand(len(seeds),-1,-1)
    opt=torch.optim.Adam(model.parameters(),lr=.003);singles=[];pe=ae=0.
    for n,seed in enumerate(seeds):
        single=model_factory(torch,width,family,[seed]).to(device)
        single.load_state_dict({k:v[n:n+1].clone() for k,v in model.state_dict().items()})
        singles.append((single,torch.optim.Adam(single.parameters(),lr=.003)))
        pe=max(pe,float((model(x)[0][n]-single(x[n:n+1])[0][0]).detach().abs().max()))
    weights=torch.arange(1,len(seeds)+1,device=device)
    model(x)[0].square().mean((1,2,3)).mul(weights).sum().backward();_clip_per_seed(model,torch,1.);opt.step()
    for n,(single,o) in enumerate(singles):
        single(x[n:n+1])[0].square().mean().mul(weights[n]).backward();_clip_per_seed(single,torch,1.);o.step()
        ae=max(ae,max(float((v[n:n+1]-single.state_dict()[k]).detach().abs().max()) for k,v in model.named_parameters()))
    return dict(family=family,width=width,prediction_error=pe,adam_error=ae,valid=max(pe,ae)<1e-5)


def imposed_paths():
    """Future imposed V is an experimental input, never generated by the model."""
    t=np.arange(1000)*.025
    return np.stack((np.full(1000,-70.),np.where(t<8,-90,np.where(t<16,20,-65)),
                     -100+140*t/25, -30+65*np.sin(2*np.pi*t/7),
                     -30+65*np.sin(2*np.pi*(t/15+t*t/150))))


def path_rollout(model,seeds,torch,device,dt=.025):
    paths=imposed_paths();npath=len(paths)
    state=np.broadcast_to(np.array([.1,.9,.8,.2,.4,.6]),(npath,6)).copy()
    truth=state.reshape(npath,3,2).transpose(1,0,2).copy()
    tensor=torch.tensor(state,dtype=torch.float32,device=device)[None].expand(len(seeds),-1,-1).clone()
    errors=[[] for _ in seeds];opens=[[] for _ in seeds];currents=[[] for _ in seeds];violations=[0 for _ in seeds]
    with torch.no_grad():
        for step in range(paths.shape[1]):
            v=paths[:,step]
            x=torch.cat((torch.tensor(v,dtype=torch.float32,device=device)[None,:,None].expand(len(seeds),-1,-1),tensor,
                         torch.full((len(seeds),npath,1),dt,device=device)),-1)
            pred=model(x)[0];tensor=pred.transpose(1,2).reshape(len(seeds),npath,6)
            rr=[rates(c,v) for c in CHANNELS];truth=update(truth,np.stack([r[0] for r in rr]),np.stack([r[1] for r in rr]),dt)
            p=pred.cpu().double().numpy()
            for s in range(len(seeds)):
                violations[s]+=int(((p[s]<0)|(p[s]>1)).sum())
                errors[s].append(p[s]-truth)
                op=p[s,...,0]**3*p[s,...,1]-truth[...,0]**3*truth[...,1]
                opens[s].append(op)
                # Equal Ena55/gbar1e-5: normalize by sum absolute driving currents.
                currents[s].append(op.sum(0)/3*np.sign(v-55))
    result=[]
    for s in range(len(seeds)):
        e=np.stack(errors[s]);op=np.stack(opens[s]);cur=np.stack(currents[s])
        result.append(dict(gate_rmse=float(np.sqrt(np.mean(e**2))),gate_max_error=float(abs(e).max()),
            open_rmse=float(np.sqrt(np.mean(op**2))),total_current_normalized_rmse=float(np.sqrt(np.mean(cur**2))),
            per_channel={c:dict(gate_rmse=float(np.sqrt(np.mean(e[:,k]**2))),
                m_rmse=float(np.sqrt(np.mean(e[:,k,:,0]**2))),h_rmse=float(np.sqrt(np.mean(e[:,k,:,1]**2))),
                gate_max_error=float(abs(e[:,k]).max()),open_rmse=float(np.sqrt(np.mean(op[:,k]**2)))) for k,c in enumerate(CHANNELS)},
            per_path_gate_rmse=[float(np.sqrt(np.mean(e[:,:,p]**2))) for p in range(npath)],
            occupancy_violations=violations[s],
            finite=bool(np.isfinite(e).all()),dt_ms=dt,duration_ms=paths.shape[1]*dt))
    return result


def held_rollout(model,seeds,torch,device,horizons):
    v=np.random.default_rng(240405).uniform(-135,75,64)
    values=np.c_[np.repeat(v,2),np.repeat(np.array([[0.]*6,[1.]*6]),64,axis=0),np.ones(128)]
    # Pair each V with both initial extremes, rather than grouping by state.
    values[:,1:7]=np.tile([[0.]*6,[1.]*6],(64,1))
    x=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(len(seeds),-1,-1)
    result=[{} for _ in seeds]
    with torch.no_grad():
        state,inf,tau=model(x);z=-torch.expm1(-1/tau)
        state=x[...,1:7].reshape(len(seeds),128,3,2).transpose(1,2).clone()
        for step in range(1,max(horizons)+1):
            state=(1-z)*state+z*inf
            if step in horizons:
                p=state.cpu().double().numpy();i=inf.cpu().double().numpy();t=tau.cpu().double().numpy()
                macro=values.copy();macro[:,7]=step
                for s in range(len(seeds)):result[s][str(step)]=measurements(macro,p[s],i[s],t[s])
    return result


def native_audit(teacher,output):
    """RANGE-only native instrumentation, no changed kinetics or current formula."""
    from neuron import h,load_mechanisms
    import neuron
    if neuron.__version__.split('+')[0]!='8.2.7':raise RuntimeError('NEURON version mismatch')
    if subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'],text=True).strip()!=TEACHER_REVISION:raise RuntimeError('teacher revision mismatch')
    root=Path(output)/'native';root.mkdir();hashes={}
    for c in CHANNELS:
        source=Path(teacher)/'L5PC_NEURON_simulation/mods'/f'{c}.mod'
        raw=source.read_text();modified=raw.replace('SUFFIX '+c,'SUFFIX '+c+'\n RANGE mInf, mTau, hInf, hTau',1)
        if raw==modified:raise RuntimeError('RANGE instrumentation failed')
        (root/source.name).write_text(modified)
        hashes[c]=dict(canonical_mod_sha256=sha(source),instrumented_mod_sha256=sha(root/source.name))
    command=shutil.which('nrnivmodl') or str(Path(sys.executable).parent/'nrnivmodl')
    with (root/'compile.log').open('w') as log:subprocess.run([command,str(root.resolve())],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    load_mechanisms(str(root.resolve()));h.CVode().active(0);h.secondorder=0;rows=[]
    for c in CHANNELS:
        sec=h.Section(name='audit_'+c);sec.L=sec.diam=10;sec.insert(c);mech=getattr(sec(.5),c)
        gname='g'+c+'bar';setattr(mech,gname,0.)
        er=es=ev=ec=0.
        for v in sorted(set(np.linspace(-135,75,85).tolist()+list(SEAMS))):
            i,t=rates(c,v);h.finitialize(v)
            expected=np.r_[i,t];actual=np.array([mech.mInf,mech.hInf,mech.mTau,mech.hTau])
            er=max(er,float(np.max(abs(actual-expected)/(1+abs(expected)))))
            for dt in DT:
                h.dt=dt;h.finitialize(v);mech.m=.31;mech.h=.79;h.fadvance()
                es=max(es,float(abs(np.array([mech.m,mech.h])-update(np.array([.31,.79]),i,t,dt)).max()))
                ev=max(ev,abs(float(sec(.5).v)-v))
            h.finitialize(v);mech.m=.31;mech.h=.79;sec(.5).ena=55;setattr(mech,gname,1e-5);h.fcurrent()
            expected_current=1e-5*.31**3*.79*(v-55)
            ec=max(ec,abs(float(mech.ina)-expected_current));setattr(mech,gname,0.)
        rows.append(dict(channel=c,rate_scaled_max_error=er,gate_max_error=es,voltage_drift_mv=ev,current_max_error_ma_cm2=ec))
        h.delete_section(sec=sec)
    report=dict(valid=all(r['rate_scaled_max_error']<1e-7 and r['gate_max_error']<1e-7 and r['voltage_drift_mv']<1e-8 and r['current_max_error_ma_cm2']<1e-10 for r in rows),rows=rows,source_hashes=hashes,
                temperature_policy='All sodium channels canonical fixed qt=2.3^1.3; exact singular perturbation order retained.',current_semantics='fcurrent algebra at supplied states, not a claim about embedded BREAKPOINT sampling')
    write(Path(output)/'native_audit.json',report)
    if not report['valid']:raise RuntimeError('Sodium family native oracle failed')
    return report


def inference_benchmark(models,best,seeds,torch,device):
    """Frozen forward latency only; not whole-neuron speedup or selection."""
    rows=[]
    with torch.no_grad():
        for batch_size in (1,128,1024):
            x=torch.tensor(data(240901,batch_size),dtype=torch.float32,device=device)[None].expand(len(seeds),-1,-1)
            for family,selection in best.items():
                model=models[family,selection['width']]
                for _ in range(20):model(x)
                timings=[]
                for _ in range(5):
                    if device=='cuda':
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(100):model(x)
                        end.record();torch.cuda.synchronize()
                        timings.append(start.elapsed_time(end)/100)
                    else:
                        start=time.perf_counter()
                        for _ in range(100):model(x)
                        timings.append((time.perf_counter()-start)*1000/100)
                rows.append(dict(family=family,batch_size=batch_size,median_forward_ms=float(np.median(timings)),repetitions_ms=timings,width=selection['width'],seed_bundle_count=len(seeds)))
    return dict(rows=rows,used_for_selection=False,scope='Frozen rate prediction plus exponential gate solver; vectorized3seed bundle. No analytic current, NEURON, whole-neuron or cross-device speedup claim.')


def run(output,config,revision):
    from .gpu_baseline_runtime import configure_torch_runtime,environment_manifest
    torch=configure_torch_runtime(17);device='cuda' if torch.cuda.is_available() else 'cpu';output=Path(output)
    if config['require_cuda'] and device!='cuda':raise RuntimeError('Task24 requires CUDA')
    if not json.loads((output/'native_audit.json').read_text())['valid']:raise RuntimeError('native preflight required')
    seeds=config['seeds'];ds=config['data_seeds']
    fit=data(ds['fit'],config['pool_size']);r=np.random.default_rng(ds['fit_regions']);n=len(fit)//4
    fit[:n,0]=r.uniform(-75,-25,n);fit[n:2*n,0]=r.uniform(-65,-35,n)
    dev=dict(uniform=data(ds['development'][0],2048),activation_boundary=data(ds['development'][1],2048,(-75,-25)),state_extrema=state_grid())
    np.savez_compressed(output/'fit_development.npz',fit=fit,**dev)
    ti,tt,ty=targets(fit)
    x,ti,tt,ty=[torch.tensor(a,dtype=torch.float32,device=device) for a in (fit,ti,tt,ty)]
    models={(f,w):model_factory(torch,w,f,seeds).to(device) for f in FAMILIES for w in config['widths']}
    pre=[equivalence(torch,w,f,seeds,device) for f,w in models]
    write(output/'equivalence_preflight.json',dict(valid=all(r['valid'] for r in pre),rows=pre))
    if not all(r['valid'] for r in pre):raise RuntimeError('vectorization equivalence failed')
    opts={k:torch.optim.Adam(m.parameters(),lr=config['learning_rate']) for k,m in models.items()}
    ladder=[];best={};start=time.monotonic();rng=np.random.default_rng(ds['minibatches'])
    for step in range(config['checkpoints'][-1]+1):
        if step in config['checkpoints']:
            for (family,width),model in models.items():
                mm={d:evaluate(model,v,seeds,torch,device) for d,v in dev.items()}
                path=output/f'checkpoint_{family}_w{width}_step{step}.pt';torch.save(model.state_dict(),path)
                scores=[]
                for s,seed in enumerate(seeds):
                    domains={d:a[s] for d,a in mm.items()};sc=score(domains,config);scores.append(sc)
                    ladder.append(dict(family=family,width=width,step=step,seed=seed,metrics=domains,score=sc))
                candidate=dict(family=family,width=width,step=step,score=max(scores),checkpoint=path.name)
                if family not in best or candidate['score']<best[family]['score']:best[family]=candidate
            write(output/'development_ladder.json',ladder)
            print(f'[GIADA Task24] checkpoint {step}/{config["checkpoints"][-1]} elapsed={(time.monotonic()-start)/60:.1f}min',flush=True)
        if step==config['checkpoints'][-1]:break
        idx=torch.tensor(rng.integers(0,len(fit),config['batch_size']),device=device)
        batch=x[idx][None].expand(len(seeds),-1,-1)
        for key,model in models.items():
            opts[key].zero_grad(set_to_none=True);p,i,t=model(batch)
            losses=(p-ty[None,:,idx]).square().mean((1,2,3))+config['inf_weight']*(i-ti[None,:,idx]).square().mean((1,2,3))+config['log_tau_weight']*(t.log()-tt[None,:,idx].log()).square().mean((1,2,3))
            losses.sum().backward();_clip_per_seed(model,torch,1.);opts[key].step()
    indexed={(r['family'],r['width'],r['step'],r['seed']):r for r in ladder}
    write(output/'paired_architecture_contrasts.json',[dict(family=f,width=r['width'],step=r['step'],seed=r['seed'],score_difference=indexed[f,r['width'],r['step'],r['seed']]['score']-r['score']) for r in ladder if r['family']=='independent' for f in ('shared_heads','conditioned')])
    scaling=[]
    for family in FAMILIES:
        for seed in seeds:
            for width in config['widths']:
                if 15000 in config['checkpoints'] and 30000 in config['checkpoints']:
                    old=indexed[family,width,15000,seed]['score'];new=indexed[family,width,30000,seed]['score']
                    scaling.append(dict(family=family,seed=seed,factor='budget',width=width,
                                        improvement_fraction=(old-new)/max(old,1e-12)))
            if 16 in config['widths'] and 32 in config['widths']:
                for step in config['checkpoints']:
                    old=indexed[family,16,step,seed]['score'];new=indexed[family,32,step,seed]['score']
                    scaling.append(dict(family=family,seed=seed,factor='width',step=step,
                                        improvement_fraction=(old-new)/max(old,1e-12)))
    write(output/'paired_scaling_contrasts.json',scaling)
    freeze=dict(selection=best,fresh_accessed=False,config=config,code_revision=revision,checkpoint_hashes={p.name:sha(p) for p in output.glob('checkpoint_*.pt')})
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();write(output/'selection_freeze.json',freeze)
    fresh=dict(in_support=data(ds['fresh'][0],4096),activation_boundary=data(ds['fresh'][1],4096,(-75,-25)),state_extrema=state_grid(True),ood_negative=data(ds['fresh'][2],2048,(-155,-135)),ood_positive=data(ds['fresh'][3],2048,(75,95)))
    np.savez_compressed(output/'fresh.npz',**fresh,imposed_voltage_paths=imposed_paths())
    results=[]
    for family,selection in best.items():
        model=models[family,selection['width']];cp=output/selection['checkpoint']
        if sha(cp)!=freeze['checkpoint_hashes'][cp.name]:raise RuntimeError('frozen checkpoint changed')
        model.load_state_dict(torch.load(cp,map_location=device,weights_only=True))
        mm={d:evaluate(model,v,seeds,torch,device) for d,v in fresh.items()}
        swapped=evaluate(model,fresh['in_support'],seeds,torch,device,True)
        paths={name:path_rollout(model,seeds,torch,device,dt) for name,dt in config['path_dt_ms'].items()}
        held=held_rollout(model,seeds,torch,device,config['rollout_horizons_steps'])
        for s,seed in enumerate(seeds):
            domains={d:a[s] for d,a in mm.items()}
            path_s={name:rows[s] for name,rows in paths.items()}
            path_passed=all(row['finite'] and all(row[k]<=t for k,t in config['path_gates'].items())
                and all(all(row['per_channel'][c][k]<=t for k,t in config['path_channel_gates'].items()) for c in CHANNELS) for row in path_s.values())
            passed=all(passes(domains[d],config['gates'],config['pair_gates']) for d in ('in_support','activation_boundary','state_extrema')) and path_passed and all(passes(m,config['held_gates'],config['pair_gates']) for m in held[s].values())
            ratio=np.median([swapped[s][c]['gate_rmse']/max(domains['in_support'][c]['gate_rmse'],1e-12) for c in CHANNELS])
            results.append(dict(family=family,width=selection['width'],step=selection['step'],seed=seed,metrics=domains,path_rollout=path_s,held_rollout=held[s],swapped_identity=swapped[s],swapped_identity_rmse_ratio=float(ratio),passed=passed,parameter_count=sum(p.numel() for p in model.parameters())//len(seeds)))
    numerical=[]
    for size in (513,2049):
        grid=np.linspace(-135,75,size);rr=[rates(c,grid) for c in CHANNELS]
        for d,v in fresh.items():
            if d.startswith('ood'):continue
            inf=np.stack([np.stack([np.interp(v[:,0],grid,r[0][:,g]) for g in range(2)],-1) for r in rr])
            tau=np.stack([np.stack([np.interp(v[:,0],grid,r[1][:,g]) for g in range(2)],-1) for r in rr])
            state=v[:,1:7].reshape(-1,3,2).transpose(1,0,2)
            m=measurements(v,update(state,inf,tau,v[:,7]),inf,tau)
            numerical.append(dict(family=f'lut_{size}_f64',domain=d,metrics=m,passed=passes(m,config['gates'],config['pair_gates'])))
    passing={f:all(r['passed'] for r in results if r['family']==f) for f in FAMILIES}
    counts={f:next(r['parameter_count'] for r in results if r['family']==f) for f in FAMILIES}
    sharing={f:passing['independent'] and passing[f] and counts[f]<counts['independent'] for f in ('shared_heads','conditioned')}
    benchmark=inference_benchmark(models,best,seeds,torch,device)
    write(output/'inference_benchmark.json',benchmark)
    write(output/'fresh_metrics.json',dict(learned=results,numerical=numerical))
    report=dict(schema_version='giada-roadmap-task24-v1',valid=True,per_family_passed=passing,composition_passed=passing['independent'],sharing_confirmed=any(sharing.values()),sharing_with_fewer_parameters=sharing,selected_parameter_counts=counts,
        task25_preparation_authorized=passing['independent'],selection=best,learned=results,numerical=numerical,inference_benchmark=benchmark,fresh_used_for_selection=False,training_and_evaluation_seconds=time.monotonic()-start,environment=environment_manifest(torch),scope=config['limits'],full_neuron_speedup_claimed=False)
    write(output/'final_report.json',report);return report
