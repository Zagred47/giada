"""Task20: SK_E2 under imposed calcium, explicitly not CaDynamics coupling."""
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from . import single_gate_transfer as engine
from .hh_family_transfer import TEACHER_REVISION, sha, write

CHANNELS = ('calcium_log', 'calcium_linear', 'voltage_only')


def canonical_inf(cai):
    ca = np.asarray(cai, dtype=np.float64)
    if np.any(ca < 0):
        raise ValueError('Negative calcium is outside canonical contract')
    effective = np.where(ca < 1e-7, ca + 1e-7, ca)
    return 1 / (1 + (.00043 / effective) ** 4.8)


def rates(channel, log_calcium):
    return canonical_inf(10. ** np.asarray(log_calcium)), np.ones_like(np.asarray(log_calcium, dtype=float))


def data(seed, count, domain=(-7, -2)):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(*domain, count), rng.uniform(0, 1, count),
                 rng.choice([.025, .1, .5, 1, 5, 25, 100], count), rng.uniform(-135, 75, count)]


def fit(config):
    x = data(200101, config['pool_size'])
    n = len(x) // 2
    x[:n, 0] = np.random.default_rng(200102).uniform(-4, -3, n)
    return x


def grid(fresh=False):
    size, offset = (401, .37) if fresh else (201, .5)
    ca = -7 + 5 * (np.arange(size) + offset) / size
    if fresh:
        ca = np.r_[[-7., -2.], ca]
    c, dt, z = np.meshgrid(ca, [.025, .1, .5, 1, 5, 25, 100], [0., 1.], indexing='ij')
    return np.c_[c.ravel(), z.ravel(), dt.ravel(), np.full(c.size, -65.)]


def development():
    return dict(uniform=data(200201, 2048), negative_tail=data(200202, 2048, (-7, -5)), state_extrema=grid())


def fresh():
    return dict(in_support=data(200401, 4096), negative_tail=data(200404, 4096, (-7, -5)), state_extrema=grid(True),
                ood_negative=data(200402, 2048, (-9, -7)), ood_positive=data(200403, 2048, (-2, -1)))


def roll_inputs():
    ca = np.random.default_rng(200405).uniform(-7, -2, 128)
    return np.c_[np.repeat(ca, 2), np.tile([0., 1.], 128), np.ones(256), np.full(256, -65.)]


def measurements(x, prediction, inf, tau, channel):
    ti, tt = rates(channel, x[:, 0])
    error = prediction - engine.update(x[:, 1], ti, tt, x[:, 2])
    return dict(gate_rmse=float(np.sqrt(np.mean(error**2))), gate_max_error=float(abs(error).max()),
        inf_rmse=float(np.sqrt(np.mean((inf-ti)**2))), log_tau_rmse=float(np.sqrt(np.mean(np.log(tau)**2))),
        open_rmse=float(np.sqrt(np.mean(error**2))), current_rmse_ma_cm2=float(np.sqrt(np.mean((1e-6*error*(x[:, 3]+85))**2))),
        occupancy_violations=int(np.sum((prediction<0)|(prediction>1))),
        finite=bool(np.isfinite(prediction).all() and np.isfinite(inf).all() and np.isfinite(tau).all() and (tau>0).all()))


def parameter_count(width):
    return width*width + 4*width + 1


def model_factory(torch, width, descriptors):
    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encodings = [d[0] for d in descriptors]
            n = len(descriptors)
            for name, shape in [('w1',(n,width,1)),('b1',(n,width)),('w2',(n,width,width)),('b2',(n,width)),('wo',(n,1,width)),('bo',(n,1))]:
                value = torch.zeros(shape)
                if name.startswith('w'):
                    for index, (_, _, seed) in enumerate(descriptors):
                        value[index].normal_(0,.1,generator=torch.Generator().manual_seed(seed+{'w1':0,'w2':100,'wo':200}[name]))
                setattr(self,name,torch.nn.Parameter(value))

        def forward(self,x):
            log = (x[...,0]+4.5)/2.5
            linear = torch.pow(10.,x[...,0])/.00043
            voltage = (x[...,3]+30)/100
            f = torch.stack([{'calcium_log':log[n],'calcium_linear':linear[n],'voltage_only':voltage[n]}[e] for n,e in enumerate(self.encodings)]).unsqueeze(-1)
            f = torch.nn.functional.silu(torch.bmm(f,self.w1.transpose(1,2))+self.b1[:,None])
            f = torch.nn.functional.silu(torch.bmm(f,self.w2.transpose(1,2))+self.b2[:,None])
            inf = torch.sigmoid((torch.bmm(f,self.wo.transpose(1,2))+self.bo[:,None])[...,0])
            tau = torch.ones_like(inf)
            z = -torch.expm1(-x[...,2])
            return (1-z)*x[...,1]+z*inf,inf,tau
    return Ensemble()


def equivalent(torch,model,width,descriptors,probe):
    from .joint_gate_symmetric_confirmation import _clip_per_seed
    clone=model_factory(torch,width,descriptors).to(probe.device)
    clone.load_state_dict(model.state_dict())
    opt=torch.optim.Adam(clone.parameters(),lr=.003)
    weights=torch.arange(1,len(descriptors)+1,device=probe.device)
    singles=[]
    for n,d in enumerate(descriptors):
        s=model_factory(torch,width,[d]).to(probe.device)
        s.load_state_dict({k:v[n:n+1].clone() for k,v in model.state_dict().items()})
        singles.append((s,torch.optim.Adam(s.parameters(),lr=.003)))
    prediction=max(float((clone(probe)[0][n]-s(probe[n:n+1])[0][0]).detach().abs().max()) for n,(s,_) in enumerate(singles))
    clone(probe)[0].square().mean(1).mul(weights).sum().backward()
    _clip_per_seed(clone,torch,1.);opt.step()
    error=0.
    for n,(s,o) in enumerate(singles):
        s(probe[n:n+1])[0].square().mean().mul(weights[n]).backward()
        _clip_per_seed(s,torch,1.);o.step()
        error=max(error,max(float((v[n:n+1]-s.state_dict()[k]).detach().abs().max()) for k,v in clone.named_parameters()))
    return dict(width=width,prediction_error=prediction,adam_error=error,valid=max(prediction,error)<1e-5)


def evaluate(model,values,descriptors,torch,device):
    inp=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(len(descriptors),-1,-1)
    with torch.no_grad():
        result=[v.cpu().double().numpy() for v in model(inp)]
    return [measurements(values,*(r[n] for r in result),d[0]) for n,d in enumerate(descriptors)]


def rollout(model,descriptors,values,horizons,torch,device):
    inp=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(len(descriptors),-1,-1)
    results=[{} for _ in descriptors]
    with torch.no_grad():
        _,inf,tau=model(inp);z=-torch.expm1(-inp[...,2]/tau);state=inp[...,1].clone()
        for step in range(1,max(horizons)+1):
            state=(1-z)*state+z*inf
            if step in horizons:
                p=state.cpu().double().numpy()
                ti,tt=rates('',values[:,0]);target=engine.update(values[:,1],ti,tt,values[:,2]*step)
                for n in range(len(descriptors)):
                    error=p[n]-target
                    results[n][str(step)]=dict(gate_rmse=float(np.sqrt(np.mean(error**2))),gate_max_error=float(abs(error).max()),occupancy_violations=int(np.sum((p[n]<0)|(p[n]>1))),finite=bool(np.isfinite(p[n]).all()))
    return results


def gpu_benchmark(torch,selected_models,values,output):
    # Acceleration is not a Task20 promotion claim; accuracy and input sufficiency are.
    return dict(measured=False,reason='Task20 isolates calcium input and kinetics; no speedup claim. Task19 eager MLP was slower than formula/LUT.')


def imposed_path(model,torch,device,output=None,seed=None):
    rng=np.random.default_rng(200501)
    logs=rng.uniform(-7,-2,(1000,64))
    # Four piecewise-constant dwell lengths and both extreme initial states.
    rows=[];stored={}
    for dwell in (1,5,25,100):
        held=logs[np.arange(1000)//dwell*dwell]
        ca=np.repeat(held,2,axis=1)
        pred=np.tile([0.,1.],64);truth=pred.copy();errors=[];predictions=[];targets=[]
        with torch.no_grad():
            for t in range(1000):
                x=np.c_[ca[t],pred,np.ones(128),np.full(128,-65.)]
                pred=model(torch.tensor(x,dtype=torch.float32,device=device)[None])[0][0].cpu().double().numpy()
                truth=engine.update(truth,canonical_inf(10.**ca[t]),1.,1.)
                errors.append(pred-truth)
                predictions.append(pred.copy());targets.append(truth.copy())
        e=np.asarray(errors)
        rows.append(dict(dwell_ms=dwell,gate_rmse=float(np.sqrt(np.mean(e**2))),gate_max_error=float(abs(e).max()),occupancy_violations=0 if np.all((pred>=0)&(pred<=1)) else int(np.sum((pred<0)|(pred>1))),finite=bool(np.isfinite(e).all())))
        stored[f'log_cai_dwell{dwell}']=ca
        stored[f'prediction_dwell{dwell}']=np.asarray(predictions)
        stored[f'target_dwell{dwell}']=np.asarray(targets)
    if output is not None:
        np.savez_compressed(Path(output)/f'imposed_paths_seed{seed}.npz',**stored)
    return rows


def finalize(report,models,torch,device,output,config):
    # Reconstruct all three frozen seeds, not just benchmark seed17.
    import json
    freeze=json.loads((Path(output)/'selection_freeze.json').read_text())
    selected=freeze['selection']['calcium_log/rate_supervised']
    descriptors=[(c,o,s) for c in CHANNELS for o in config['objectives'] for s in config['seeds']]
    cp=Path(output)/f'checkpoint_w{selected["width"]}_step{selected["step"]}.pt'
    assert sha(cp)==freeze['checkpoint_hashes'][cp.name]
    state=torch.load(cp,map_location=device,weights_only=True)
    paths=[]
    for index in selected['indices']:
        model=model_factory(torch,selected['width'],[descriptors[index]]).to(device)
        model.load_state_dict({k:v[index:index+1].clone() for k,v in state.items()})
        paths.append(dict(seed=descriptors[index][2],rows=imposed_path(model,torch,device,output,descriptors[index][2])))
    path_pass=all(engine.passes(row,config['rollout_gates']) for p in paths for row in p['rows'])
    primary=report['per_channel_transfer']['calcium_log'] and path_pass
    report.update(schema_version='giada-roadmap-task20-v1',calcium_gate_passed=primary,task21_authorized=primary,
        imposed_calcium_paths=paths,imposed_path_passed=path_pass,single_gate_transfer_passed=primary,
        calcium_units='cai mM; stored input column0 log10(cai/mM); column3 voltage mV',
        domain_names={'negative_tail':'low calcium 1e-7..1e-5mM','ood_negative':'1e-9..1e-7mM canonical branch diagnostic','ood_positive':'.01...1mM diagnostic'})
    report.pop('task20_authorized',None)
    ladder=json.loads((Path(output)/'development_ladder.json').read_text())
    by={(r['channel'],r['objective'],r['seed'],r['width'],r['step']):r for r in ladder}
    contrasts=[]
    for row in ladder:
        if row['channel']!='calcium_log':continue
        for alternative in ('calcium_linear','voltage_only'):
            other=by[alternative,row['objective'],row['seed'],row['width'],row['step']]
            contrasts.append(dict(factor='input_encoding' if alternative=='calcium_linear' else 'calcium_information',
                base='calcium_log',alternative=alternative,objective=row['objective'],seed=row['seed'],width=row['width'],step=row['step'],
                metric_differences={d:{k:other['metrics'][d][k]-row['metrics'][d][k] for k in config['gates']} for d in row['metrics']}))
    write(Path(output)/'paired_input_contrasts.json',contrasts)
    return report


def native_audit(teacher,output):
    from neuron import h,load_mechanisms
    import neuron
    assert neuron.__version__.split('+')[0]=='8.2.7'
    assert subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'],text=True).strip()==TEACHER_REVISION
    root=Path(output)/'native';root.mkdir()
    source=Path(teacher)/'L5PC_NEURON_simulation/mods/SK_E2.mod'
    raw=source.read_text();assert raw.count('SUFFIX SK_E2')==1
    (root/'SK_E2.mod').write_text(raw.replace('SUFFIX SK_E2','SUFFIX SK_E2\n RANGE zInf, zTau',1))
    command=shutil.which('nrnivmodl') or str(Path(sys.executable).parent/'nrnivmodl')
    with (root/'compile.log').open('w') as log:
        subprocess.run([command,str(root.resolve())],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    load_mechanisms(str(root.resolve()));h.CVode().active(0);h.secondorder=0
    sec=h.Section();sec.L=sec.diam=10;sec.insert('SK_E2');mech=sec(.5).SK_E2;mech.gSK_E2bar=0
    maximum_rate=maximum_gate=calcium_drift=0.
    for ca in np.r_[np.logspace(-9,-1,101),[0.,1e-7,np.nextafter(1e-7,0),np.nextafter(1e-7,1),.00043]]:
        for v in (-100.,-65.,30.):
            for dt in (.025,.1,1.,25.,100.):
                h.dt=dt;h.finitialize(v);sec(.5).cai=float(ca);h.fcurrent()
                mech.z=.31;h.fadvance()
                expected=float(canonical_inf(ca))
                maximum_rate=max(maximum_rate,abs(mech.zInf-expected),abs(mech.zTau-1))
                maximum_gate=max(maximum_gate,abs(mech.z-float(engine.update(.31,expected,1.,dt))))
                calcium_drift=max(calcium_drift,abs(sec(.5).cai-ca))
    report=dict(valid=max(maximum_rate,maximum_gate)<1e-7 and calcium_drift<1e-12,
        maximum_inf_error=maximum_rate,maximum_gate_error=maximum_gate,maximum_cai_drift_mM=calcium_drift,
        canonical_mod_sha256=sha(source),instrumented_mod_sha256=sha(root/'SK_E2.mod'),
        canonical_branch='ca<1e-7: local ca +=1e-7; exactly1e-7 unchanged; cai storage not mutated',
        tau_ms=1.,scope='Imposed cai; no CaDynamics mechanism; zero gbar; voltage invariance audited at3voltages.')
    write(Path(output)/'native_audit.json',report)
    if not report['valid']:raise RuntimeError('Task20 native identity failed')
    return report


def run(output,config,revision):
    return engine.run(output,config,revision,backend=sys.modules[__name__])
