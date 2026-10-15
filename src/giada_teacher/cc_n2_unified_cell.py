"""CC-05..08: native ionic playground and paired vectorized recurrent cells.

No synaptic compression, sealed confirmation or inference-speed claim.
Native and CUDA phases must execute in separate OS processes.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from . import cc_n1_contract_memory as n1
from . import ionic_block_teacher_forced as ionic
from . import task30_autonomous_voltage as t30
from . import task33_observable_synaptic_feedback as t33
from .hh_family_transfer import write

CONFIG = 'experiments/cc_n2_unified_cell_preregistration_v1.json'


def contract(root):
    spec = json.loads((root / CONFIG).read_text(encoding='utf-8'))
    if n1.sha(root / spec['parent_report']) != spec['parent_sha256']:
        raise ValueError('N2 parent SHA mismatch')
    parent = json.loads((root / spec['parent_report']).read_text())
    if not parent['valid'] or spec['dt_ms'] != .1:
        raise ValueError('N2 prerequisite or time contract invalid')
    assert spec['roadmap_ids'] == ['CC-05', 'CC-06', 'CC-07', 'CC-08']
    assert len(set(spec['role_seeds'].values())) == 3
    _, cfg, ca, _, base = n1.contract(root)
    return spec, cfg, ca, base


def design(spec):
    rows, initials, forcing = [], [], []
    ticks = round(spec['duration_ms']/spec['dt_ms'])
    memory = {'initial_cai_mM': spec['initial_cai_mM'],
              'initial_voltage_mv': spec['initial_voltage_mv']}
    for role, count in spec['role_counts'].items():
        rng = np.random.default_rng(spec['role_seeds'][role])
        for j in range(count):
            prep, duration, tail = (float(rng.uniform(*spec[key])) for key in (
                'preparation_voltage_range_mv', 'preparation_duration_range_ms', 'common_tail_range_ms'))
            row = dict(role=role, episode=j, panel=spec['panels'][j % 4],
                       preparation_voltage_mv=prep, preparation_ms=duration, tail_ms=tail)
            rows.append(row)
            initials.append(n1.conditioned_state(prep, duration, tail, memory))
            block = round(spec['current_block_ms']/spec['dt_ms'])
            forcing.append(np.repeat(rng.choice(spec['current_values_ma_cm2'],
                                      math.ceil(ticks/block)), block)[:ticks])
    return rows, np.stack(initials), np.stack(forcing, axis=1)


def native_phase(root, teacher, output):
    import subprocess
    import neuron
    from neuron import h, load_mechanisms
    spec, cfg, calcium, base = contract(root)
    if neuron.__version__.split('+')[0] != '8.2.7': raise ValueError('NEURON version')
    expected = n1.contract(root)[0]['teacher_revision']
    if subprocess.check_output(['git', '-C', str(teacher), 'rev-parse', 'HEAD'], text=True).strip() != expected:
        raise ValueError('Teacher revision')
    build = t33.compile_combined(root, teacher, output)
    load_mechanisms(str(build.resolve()))
    rows, initials, forcing = design(spec)
    ticks, count = forcing.shape
    arrays = {'voltage': np.empty((ticks+1, count)),
              'calcium': np.empty((ticks+1, count)), 'gates': np.empty((ticks+1, count, 18))}
    multipliers = np.stack([ionic.panel_multipliers()[r['panel']] for r in rows])
    for j, row in enumerate(rows):
        h.CVode().active(0); h.dt = spec['dt_ms']; h.secondorder = 0; h.celsius = cfg['temperature_c']
        sec = h.Section(name='cc_n2_native'); sec.L = sec.diam = 10.; sec.nseg = 1
        sec.cm = base['cm_uf_cm2']; sec.insert('pas')
        sec.g_pas, sec.e_pas = base['g_pas_s_cm2'], base['e_pas_mv']
        for name in (*ionic.CHANNELS, 'CaDynamics_E2'): sec.insert(name)
        h.ion_style('ca_ion', 3, 1, 0, 0, 0, sec=sec)
        seg = sec(.5); area = float(h.area(.5, sec=sec))
        clamp = h.IClamp(seg); clamp.delay = 0; clamp.dur = spec['duration_ms']+1
        h.finitialize(spec['initial_voltage_mv']); seg.cai = spec['initial_cai_mM']
        seg.eca, seg.ena, seg.ek = cfg['eca_mv'], 55., -85.
        cad = seg.CaDynamics_E2
        cad.decay, cad.gamma, cad.depth, cad.minCai = calcium['decay_ms'], calcium['gamma'], calcium['depth_um'], 1e-4
        for k, name in enumerate(ionic.CHANNELS):
            mech = getattr(seg, name)
            setattr(mech, 'g'+name+'bar', float(ionic.GBAR[k]*multipliers[j,k]))
            for q, state in enumerate(ionic.STATE_NAMES[k]):
                setattr(mech, state, float(initials[j,int(ionic.OFFSETS[k])+q]))
        for t in range(ticks+1):
            h.fcurrent()
            arrays['voltage'][t,j], arrays['calcium'][t,j] = seg.v, seg.cai
            arrays['gates'][t,j] = [float(getattr(getattr(seg, name), state))
                for name, names in zip(ionic.CHANNELS, ionic.STATE_NAMES) for state in names]
            if t < ticks:
                clamp.amp = float(forcing[t,j]*area*.01); h.fadvance()
        h.delete_section(sec=sec)
        if (j+1)%16 == 0: print(f'[GIADA CC-N2] native {j+1}/{count}', flush=True)
    initial_error = float(np.max(np.abs(arrays['gates'][0]-initials)))
    finite = all(np.isfinite(x).all() for x in arrays.values())
    valid = finite and initial_error <= spec['technical']['initial_state_atol'] and arrays['calcium'].min() >= 0
    np.savez_compressed(output/'native_dataset.npz', **arrays, injection=forcing, multipliers=multipliers)
    role_files={}
    for role in spec['role_counts']:
        ids=[i for i,r in enumerate(rows) if r['role']==role]
        filename=f'native_{role}.npz'
        np.savez_compressed(output/filename,**{k:a[:,ids] for k,a in arrays.items()},
                            injection=forcing[:,ids],multipliers=multipliers[ids])
        role_files[role]={'file':filename,'sha256':n1.sha(output/filename)}
    write(output/'dataset_manifest.json', {'valid': bool(valid), 'rows': rows,
        'initial_gate_max_abs_error': initial_error, 'role_counts': spec['role_counts'],
        'target_source': 'canonical_NEURON', 'synapses_present': False,
        'dataset_sha256': n1.sha(output/'native_dataset.npz'),
        'role_files':role_files,
        'role_split_unit': 'independent_seed_episode_and_continuous_preparation_history',
        'finite': finite})
    if not valid: raise RuntimeError('N2 native data audit failed')


def make_model(torch, spec, calcium, base, members=None):
    """Batched independent MLPs; no batch reductions or cross-member normalization."""
    choices = list(itertools.product(spec['arms'], spec['seeds'])) if members is None else members
    class Cell(torch.nn.Module):
        def __init__(self):
            super().__init__()
            n, w = len(choices), spec['width']
            self.w1 = torch.nn.Parameter(torch.empty(n,32,w)); self.b1 = torch.nn.Parameter(torch.zeros(n,1,w))
            self.w2 = torch.nn.Parameter(torch.empty(n,w,w)); self.b2 = torch.nn.Parameter(torch.zeros(n,1,w))
            self.wo = torch.nn.Parameter(torch.empty(n,w,44)); self.bo = torch.nn.Parameter(torch.zeros(n,1,44))
            self.wi = torch.nn.Parameter(torch.empty(n,20,18)); self.bi = torch.nn.Parameter(torch.zeros(n,1,18))
            for i, (_, seed) in enumerate(choices):
                generator = torch.Generator().manual_seed(seed)
                for p in (self.w1,self.w2,self.wo,self.wi):
                    p.data[i].copy_(torch.randn(p[i].shape, generator=generator)/math.sqrt(p.shape[1]))
            # All shared kinetic controls start with a broad, physically meaningful
            # timescale ladder; no teacher future information enters initialization.
            initial_inf, initial_tau = [], []
            for name in ionic.CHANNELS:
                inf, tau = ionic.rates(name,np.array([-65.]),np.array([1e-4]))
                initial_inf.extend(np.asarray(inf).reshape(-1).tolist())
                initial_tau.extend(np.asarray(tau).reshape(-1).tolist())
            for i,(arm,_) in enumerate(choices):
                if arm['head']=='explicit':
                    probability=torch.tensor(initial_inf).clamp(1e-5,1-1e-5)
                    self.bo.data[i,0,:18]=torch.logit(probability)
                    self.bo.data[i,0,18:36]=torch.tensor(initial_tau).clamp_min(.011).log()
                else:
                    self.bo.data[i,0,18:36]=torch.linspace(math.log(.2),math.log(200.),18)
            def flag(key, value):
                return torch.tensor([a[key] == value for a,_ in choices]).reshape(n,1,1)
            self.register_buffer('explicit', flag('head','explicit'))
            self.register_buffer('group', flag('head','conductance'))
            self.register_buffer('direct', flag('head','direct'))
            self.register_buffer('warm', flag('init','warmup'))
            self.register_buffer('standard', flag('init','standard'))
            self.register_buffer('mask', torch.tensor([[j<a['latent'] for j in range(18)] for a,_ in choices])[:,None,:])
            self.register_buffer('gbars', torch.tensor(ionic.GBAR.tolist(),dtype=torch.float32))
            self.register_buffer('reversals', torch.tensor(ionic.REVERSALS.tolist(),dtype=torch.float32))

        def initialize(self, v, ca, gates):
            data = torch.cat(((v+65)/40, torch.log(ca.clamp_min(1e-8)/1e-4), gates),-1)
            z = torch.sigmoid(torch.bmm(data,self.wi)+self.bi)*self.mask
            return torch.where(self.explicit, gates, torch.where(self.standard|self.warm, torch.zeros_like(z),z))

        def forward(self, v, ca, z, current, mult):
            # Explicit kinetic rates depend on V/Ca only, never hidden gates or forcing.
            x = torch.cat(((v+65)/40, torch.log(ca.clamp_min(1e-8)/1e-4),
                           torch.where(self.explicit,torch.zeros_like(current),current/.006),
                           torch.where(self.explicit,torch.zeros_like(mult),mult/4),
                           torch.where(self.explicit,torch.zeros_like(z),z*self.mask)),-1)
            x = torch.nn.functional.silu(torch.bmm(x,self.w1)+self.b1)
            x = torch.nn.functional.silu(torch.bmm(x,self.w2)+self.b2)
            y = torch.bmm(x,self.wo)+self.bo
            inf = torch.sigmoid(y[...,:18]); tau = .01+torch.exp(y[...,18:36].clamp(-9,10))
            new_z = (z+(inf-z)*(-torch.expm1(-spec['dt_ms']/tau)))*self.mask
            openings = []
            offsets = (0,2,4,6,8,10,11,12,14,16,17)
            powers = (2,2,3,3,3,1,1,2,4,1,1)
            for k, names in enumerate(ionic.STATE_NAMES):
                opening = new_z[...,offsets[k]].pow(powers[k])
                if len(names)==2: opening = opening*new_z[...,offsets[k]+1]
                openings.append(opening)
            g = torch.stack(openings,-1)*self.gbars*mult
            total_g = g.sum(-1,keepdim=True); reversal_g = (g*self.reversals).sum(-1,keepdim=True)
            ica = (g[...,:2]*(v-120)).sum(-1,keepdim=True)
            # Grouped conductances: Ca, Na, Ih, K. Positive readout bounded only by softplus.
            limits = torch.stack(((self.gbars[:2]*mult[...,:2]).sum(-1),
                (self.gbars[2:5]*mult[...,2:5]).sum(-1), self.gbars[5]*mult[...,5],
                (self.gbars[6:]*mult[...,6:]).sum(-1)),-1)
            grouped = torch.nn.functional.softplus(y[...,36:40])*limits
            group_g = grouped.sum(-1,keepdim=True)
            group_ge = (grouped*torch.tensor([120.,55.,-45.,-85.],device=v.device)).sum(-1,keepdim=True)
            cap = 1e-3*base['cm_uf_cm2']/spec['dt_ms']; leak=base['g_pas_s_cm2']
            ev = (cap*v+leak*base['e_pas_mv']+reversal_g+current)/(cap+leak+total_g)
            gv = (cap*v+leak*base['e_pas_mv']+group_ge+current)/(cap+leak+group_g)
            iv = (cap*v+leak*base['e_pas_mv']+current-.01*y[...,36:37])/(cap+leak)
            dv = v+5*torch.tanh(y[...,36:37])
            prediction = torch.where(self.explicit,ev,torch.where(self.group,gv,torch.where(self.direct,dv,iv)))
            decay = math.exp(-spec['dt_ms']/calcium['decay_ms'])
            factor = 10000*calcium['gamma']/(2*96485.33212*calcium['depth_um'])*calcium['decay_ms']*(1-decay)
            calcium_current = torch.where(self.explicit,ica,torch.where(self.group,grouped[...,:1]*(v-120),
                                            -1e-5*torch.nn.functional.softplus(y[...,37:38])))
            physical_ca = 1e-4+(ca-1e-4)*decay-calcium_current*factor
            direct_ca = ca*torch.exp(.1*torch.tanh(y[...,37:38]))
            next_ca = torch.where(self.direct,direct_ca,physical_ca)
            return prediction, next_ca, new_z
    return Cell(), choices


def clip_members(torch, model, limit):
    norm = torch.sqrt(sum(p.grad.square().flatten(1).sum(1) for p in model.parameters() if p.grad is not None))
    scale = (limit/(norm+1e-12)).clamp(max=1)
    for p in model.parameters():
        if p.grad is not None: p.grad.mul_(scale.reshape((-1,)+(1,)*(p.ndim-1)))


def equivalence(torch, model, spec, calcium, base):
    """Batched Adam and clipping must agree with isolated members, before training."""
    clone,_ = make_model(torch,spec,calcium,base)
    clone.load_state_dict(model.state_dict())
    device = next(model.parameters()).device; clone=clone.to(device)
    n = len(spec['arms'])*len(spec['seeds'])
    rng = torch.Generator(device=device).manual_seed(999)
    v=torch.randn((n,3,1),generator=rng,device=device)-65; ca=torch.full_like(v,1e-4)
    z=torch.rand((n,3,18),generator=rng,device=device); mult=torch.ones((n,3,11),device=device); current=torch.zeros_like(v)
    def loss(cell, args):
        pv,pc,pz=cell(*args)
        return (pv.square().mean((1,2)) + (pc*1e4).square().mean((1,2))+pz.square().mean((1,2)))
    opt=torch.optim.Adam(clone.parameters(),lr=spec['training']['learning_rate'])
    losses=loss(clone,(v,ca,z,current,mult)); losses.sum().backward()
    clip_members(torch,clone,spec['training']['gradient_clip_per_member']);opt.step()
    prediction_error=0.;update_error=0.
    for i,choice in enumerate(itertools.product(spec['arms'],spec['seeds'])):
        single,_=make_model(torch,spec,calcium,base,[choice]);single=single.to(device)
        state={k:t[i:i+1].clone() for k,t in model.state_dict().items() if k not in ('gbars','reversals')}
        state.update(gbars=model.gbars,reversals=model.reversals);single.load_state_dict(state)
        args=tuple(t[i:i+1] for t in (v,ca,z,current,mult))
        with torch.no_grad():
            for a,b in zip(model(v,ca,z,current,mult),single(*args)):
                prediction_error=max(prediction_error,float((a[i:i+1]-b).abs().max()))
        one=torch.optim.Adam(single.parameters(),lr=spec['training']['learning_rate'])
        loss(single,args).sum().backward();clip_members(torch,single,spec['training']['gradient_clip_per_member']);one.step()
        update_error=max(update_error,max(float((a[i:i+1]-b).abs().max()) for a,b in zip(clone.parameters(),single.parameters())))
    report={'valid':max(prediction_error,update_error)<=1e-5,'prediction_max_abs_error':prediction_error,
            'adam_update_max_abs_error':update_error,'independent_moments':True,'independent_clipping':True}
    if not report['valid']:raise RuntimeError('N2 batched equivalence failed: '+str(report))
    return report


def prepare_tensors(torch,data,device,members):
    return {k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in data.items()}


def reference_evaluation(torch,root,data,roles,start,spec,cfg,calcium,base):
    models,seeds=ionic.load_frozen(root,ionic.config(root),torch,'cuda')
    reference=[]
    with torch.no_grad():
        key=n1.contract(root)[0]['gpu']['frozen_arm']
        for name,frozen in [('formula',None)]+[(f,models[f,key]) for f in ('independent','shared_heads')]:
            # The frozen arm key is verified from N1 rather than assumed.
            for role in roles:
                ids=roles[role];v=data['voltage'][start,ids][None].expand(3,-1)
                ca=data['calcium'][start,ids][None].expand(3,-1)
                z=data['gates'][start,ids][None].expand(3,-1,-1)
                mult=data['multipliers'][ids][None].expand(3,-1,-1)
                vs=[];cs=[];zs=[]
                for t in range(start,start+300):
                    v,ca,z=n1.torch_step(torch,v,ca,z,mult,data['injection'][t,ids][None].expand(3,-1),
                        torch.zeros((*v.shape,4),device=v.device),100*np.pi,cfg,calcium,base,frozen)
                    vs.append(v[...,None]);cs.append(ca[...,None]);zs.append(z)
                for steps in (80,300):
                    rows=metrics(torch,(torch.stack(vs[:steps]),torch.stack(cs[:steps]),torch.stack(zs[:steps])),data,ids,start)
                    reference.extend({'arm':name,'seed':int(seed),'role':role,'horizon_ms':steps*.1,**row} for seed,row in zip(seeds,rows))
    return reference


def initial_state(torch,model,data,ids,start,spec,detach_warmup=False):
    n=next(model.parameters()).shape[0]
    def tile(x):return x.unsqueeze(0).expand(n,*x.shape)
    v=tile(data['voltage'][start,ids,None]);ca=tile(data['calcium'][start,ids,None]);g=tile(data['gates'][start,ids])
    z=model.initialize(v,ca,g)
    # All cells compute the same warmup graph; only registered warm arm uses it.
    if bool(model.warm.any()):
        begin=max(0,start-spec['training']['warmup_steps'])
        history=torch.zeros_like(z)
        mult=tile(data['multipliers'][ids])
        for t in range(begin,start):
            _,_,history=model(tile(data['voltage'][t,ids,None]),tile(data['calcium'][t,ids,None]),history,
                              tile(data['injection'][t,ids,None]),mult)
            if detach_warmup and (t-begin+1)%spec['training']['warmup_detach_steps']==0 and t<start-1:
                history=history.detach()
        z=torch.where(model.warm,history,z)
    return v,ca,z


def rollout(torch,model,data,ids,start,steps,spec,intervention=None):
    n=next(model.parameters()).shape[0]
    v,ca,z=initial_state(torch,model,data,ids,start,spec)
    if intervention=='reset': z=torch.where(model.explicit,z,torch.zeros_like(z))
    elif intervention=='permute': z=torch.where(model.explicit,z,torch.roll(z,1,1))
    mult=data['multipliers'][ids].unsqueeze(0).expand(n,-1,-1)
    vs,cs,zs=[],[],[]
    for t in range(start,start+steps):
        current=data['injection'][t,ids,None].unsqueeze(0).expand(n,-1,-1)
        v,ca,z=model(v,ca,z,current,mult)
        vs.append(v);cs.append(ca);zs.append(z)
    return torch.stack(vs),torch.stack(cs),torch.stack(zs)


def metrics(torch,pred,data,ids,start):
    v,ca,z=pred
    target_v=data['voltage'][start+1:start+1+len(v),ids][...,None].unsqueeze(1)
    target_ca=data['calcium'][start+1:start+1+len(v),ids][...,None].unsqueeze(1)
    per_v=((v-target_v).square().mean((0,3))).sqrt()
    per_ca=((ca-target_ca).square().mean((0,3))).sqrt()
    pooled=((v-target_v).square().mean((0,2,3))).sqrt()
    cpool=((ca-target_ca).square().mean((0,2,3))).sqrt()
    result=[]
    for i in range(v.shape[1]):
        finite=bool(torch.isfinite(v[:,i]).all() and torch.isfinite(ca[:,i]).all() and torch.isfinite(z[:,i]).all())
        def number(x):
            value=float(x)
            return value if math.isfinite(value) else 1e30
        vr=number(pooled[i]);cr=number(cpool[i])
        result.append({'finite':finite,'voltage_pooled_rmse_mv':vr,'voltage_worst_episode_rmse_mv':number(per_v[i].max()),
            'calcium_pooled_rmse_mM':cr,'calcium_worst_episode_rmse_mM':number(per_ca[i].max()),
            'score':vr/.05+cr/5e-6 if finite else 1e30,
            'physical_violations':int(((v[:,i]<-140)|(v[:,i]>100)|(~torch.isfinite(v[:,i]))).sum())+
                                  int(((ca[:,i]<0)|(~torch.isfinite(ca[:,i]))).sum())})
    return result


def gpu_phase(root,output,revision):
    import torch
    spec,cfg,calcium,base=contract(root)
    if not torch.cuda.is_available():raise RuntimeError('N2 CUDA required')
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((output/'dataset_manifest.json').read_text())
    if not manifest['valid'] or n1.sha(output/'native_dataset.npz')!=manifest['dataset_sha256']:raise ValueError('Native data identity')
    def load_role(role):
        source=manifest['role_files'][role]
        if n1.sha(output/source['file'])!=source['sha256']:raise ValueError('Role source changed')
        with np.load(output/source['file']) as archive:return {k:archive[k] for k in archive.files}
    split={r:load_role(r) for r in ('fit','development')}
    raw={k:np.concatenate([split[r][k] for r in ('fit','development')],axis=0 if k=='multipliers' else 1) for k in split['fit']}
    data=prepare_tensors(torch,raw,'cuda',len(spec['arms'])*3)
    roles={'fit':list(range(spec['role_counts']['fit'])),
           'development':list(range(spec['role_counts']['fit'],spec['role_counts']['fit']+spec['role_counts']['development']))}
    model,choices=make_model(torch,spec,calcium,base);model=model.cuda()
    preflight=equivalence(torch,model,spec,calcium,base);write(output/'equivalence_preflight.json',preflight)
    torch._dynamo.config.cache_size_limit=max(32,torch._dynamo.config.cache_size_limit)
    compile_report={'requested':True,'enabled':False,'fallback_reason':None}
    eager_forward=model.forward
    try:
        n=len(choices);v=torch.full((n,spec['training']['batch'],1),-65.,device='cuda');ca=torch.full_like(v,1e-4)
        args=(v,ca,torch.full((*v.shape[:2],18),.3,device='cuda'),torch.zeros_like(v),
              torch.ones((*v.shape[:2],11),device='cuda'))
        compiled=torch.compile(eager_forward,fullgraph=True)
        with torch.no_grad():
            expected=eager_forward(*args);actual=compiled(*args)
            error=max(float((a-b).abs().max()) for a,b in zip(expected,actual))
        if error>1e-5:raise RuntimeError(f'compiled forward mismatch {error}')
        model.forward=compiled
        compile_report.update(enabled=True,forward_max_abs_error=error)
    except Exception as error:
        model.forward=eager_forward;compile_report['fallback_reason']=str(error)[:2000]
    write(output/'compile_preflight.json',compile_report)
    print(f'[GIADA CC-N2] vectorization valid; compiled={compile_report["enabled"]}',flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=spec['training']['learning_rate'])
    members=len(choices);start=round(spec['score_start_ms']/spec['dt_ms']);horizon=80
    references=reference_evaluation(torch,root,data,{'development':roles['development']},start,spec,cfg,calcium,base)
    write(output/'baseline_evaluation.json',references)
    history=[];best=np.full(members,np.inf);best_step=[0]*members
    best_state={k:t.detach().clone() for k,t in model.state_dict().items()}
    rng=np.random.default_rng(74701);begin=time.perf_counter()
    def check(step):
        model.eval()
        with torch.no_grad():rows=metrics(torch,rollout(torch,model,data,roles['development'],start,horizon,spec),data,roles['development'],start)
        for i,row in enumerate(rows):
            history.append({'update':step,'arm':choices[i][0]['name'],'seed':choices[i][1],**row})
            if row['score']<best[i]:
                best[i]=row['score'];best_step[i]=step
                for k,t in model.state_dict().items():
                    if k in ('gbars','reversals'):continue
                    best_state[k][i].copy_(t[i])
        write(output/'learning_curves.json',history)
        torch.save({'state_dict':model.state_dict(),'update':step},output/f'checkpoint_{step}.pt')
        print(f'[GIADA CC-N2] checkpoint {step}: development median score={np.median([r["score"] for r in rows]):.4g}',flush=True)
    check(0)
    nsteps=spec['training']['unroll_steps'];total=spec['training']['updates']
    for step in range(1,total+1):
        model.train();ids=rng.choice(roles['fit'],spec['training']['batch'],replace=True).tolist()
        anchor=int(rng.integers(start,len(raw['injection'])-nsteps+1))
        v,ca,z=initial_state(torch,model,data,ids,anchor,spec,detach_warmup=True)
        mult=data['multipliers'][ids].unsqueeze(0).expand(members,-1,-1)
        losses=torch.zeros(members,device='cuda')
        for t in range(anchor,anchor+nsteps):
            v,ca,z=model(v,ca,z,data['injection'][t,ids,None].unsqueeze(0).expand(members,-1,-1),mult)
            tv=data['voltage'][t+1,ids,None];tc=data['calcium'][t+1,ids,None]
            loss=((v-tv)/5).square().mean((1,2))+((ca-tc)/5e-5).square().mean((1,2))
            gate_loss=(z-data['gates'][t+1,ids]).square().mean((1,2))
            losses=losses+loss+model.explicit.flatten().to(loss.dtype)*gate_loss
        opt.zero_grad(set_to_none=True)
        loss=losses.sum()/nsteps
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training at update {step}')
        loss.backward();clip_members(torch,model,spec['training']['gradient_clip_per_member']);opt.step()
        if step in spec['training']['checkpoints']:check(step)
        if step%50==0:
            elapsed=time.perf_counter()-begin
            print(f'[GIADA CC-N2] training {step}/{total} ETA {(elapsed/step*(total-step))/60:.1f} min',flush=True)
    model.load_state_dict(best_state);model.eval()
    torch.save({'state_dict':best_state,'choices':choices},output/'frozen_checkpoint.pt')
    freeze={'selected_updates':best_step,'development_scores':best.tolist(),
            'checkpoint_sha256':n1.sha(output/'frozen_checkpoint.pt'),'diagnostic_opened':False,
            'preregistration_sha256':n1.sha(root/CONFIG),'code_revision':revision}
    freeze['freeze_sha256']=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()
    write(output/'selection_freeze.json',freeze)
    # Only now read diagnostic states/targets into the GPU process.
    diagnostic=load_role('diagnostic');offset=raw['multipliers'].shape[0]
    raw={k:np.concatenate((v,diagnostic[k]),axis=0 if k=='multipliers' else 1) for k,v in raw.items()}
    data=prepare_tensors(torch,raw,'cuda',members)
    roles['diagnostic']=list(range(offset,offset+spec['role_counts']['diagnostic']))
    references+=reference_evaluation(torch,root,data,{'diagnostic':roles['diagnostic']},start,spec,cfg,calcium,base)
    write(output/'baseline_evaluation.json',references)
    results=[];probes=[]
    with torch.no_grad():
        for role in ('development','diagnostic'):
            ids=roles[role]
            for steps in (80,300):
                pred=rollout(torch,model,data,ids,start,steps,spec)
                if steps==300:
                    np.savez_compressed(output/f'{role}_frozen_rollout.npz',
                        voltage=pred[0].cpu().numpy(),calcium=pred[1].cpu().numpy(),
                        latent=pred[2].cpu().numpy(),episode_indices=np.asarray(ids))
                measured=metrics(torch,pred,data,ids,start)
                for i,row in enumerate(measured):
                    limits=spec['diagnostic_gate']
                    passed=row['finite'] and row['physical_violations']==0 and all(row[k]<=x for k,x in limits.items())
                    results.append({'role':role,'horizon_ms':steps*.1,'arm':choices[i][0]['name'],'seed':choices[i][1],
                                    'passed':passed,**row})
            for intervention in ('reset','permute'):
                measured=metrics(torch,rollout(torch,model,data,ids,start,80,spec,intervention),data,ids,start)
                for i,row in enumerate(measured):probes.append({'role':role,'intervention':intervention,
                    'arm':choices[i][0]['name'],'seed':choices[i][1],**row})
        # Linear gate probing is fitted only on fit; development cannot refit it.
        latent={}
        for role in ('fit','development'):
            ids=roles[role];_,_,z=initial_state(torch,model,data,ids,start,spec)
            latent[role]=z.cpu().double().numpy()
        np.savez_compressed(output/'initial_latent_probes.npz',**latent)
        probe_rows=[]
        for i,(arm,seed) in enumerate(choices):
            fit=np.concatenate((latent['fit'][i],np.ones((len(roles['fit']),1))),-1)
            dev=np.concatenate((latent['development'][i],np.ones((len(roles['development']),1))),-1)
            target=raw['gates'][start,roles['fit']]
            ridge=np.linalg.solve(fit.T@fit+1e-4*np.eye(fit.shape[1]),fit.T@target)
            error=np.sqrt(np.mean((dev@ridge-raw['gates'][start,roles['development']])**2))
            probe_rows.append({'arm':arm['name'],'seed':seed,'development_gate_probe_rmse':float(error)})
    write(output/'memory_interventions.json',probes);write(output/'gate_probes.json',probe_rows)
    write(output/'evaluation.json',results)
    allocated=sum(p.numel() for p in model.parameters())//members
    report={'schema_version':spec['schema_version'],'valid':True,'training_performed':True,
        'diagnosis':'N2_INITIAL_LEARNABILITY_MATRIX_COMPLETE','selection_used_diagnostic':False,
        'sealed_confirmation_performed':False,'synapses_tested':False,'new_candidate_promoted':False,
        'dt_ms':.1,'allocated_parameter_count_per_member':allocated,'member_count':members,
        'logical_core_state_scalars':{a['name']:2+a['latent'] for a in spec['arms']},
        'equivalence_preflight':preflight,'results':results,'interpretation':spec['interpretation'],
        'compile':compile_report,
        'n3_preparation_recommended':True,'task36_authorized':False,'inference_speed_claim':False,
        'elapsed_training_seconds':time.perf_counter()-begin,'code_revision':revision}
    write(output/'final_report.json',report)
    print('[GIADA CC-N2] complete: diagnostic artifacts saved; no candidate promotion',flush=True)
