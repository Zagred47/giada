import copy,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from . import task51g_optimization_gap as g
f=g.f;b=g.b;a=g.a
def plan(root):
    p=json.loads((root/'experiments/task51i_preregistration_v1.json').read_text());assert p['task']=='51i' and not p['thresholds_changed'];return p

def packed_step(torch,parameters,moments,members,step):
    E=len(members);device=parameters[0].device
    lr=torch.tensor([m['lr'] for m in members],device=device)
    eps=torch.tensor([m['eps'] for m in members],device=device)
    adam=torch.tensor([m['optimizer']=='adam' for m in members],device=device)
    total=torch.zeros(E,device=device)
    with torch.no_grad():
        for v,(first,second) in zip(parameters,moments):
            grad=v.grad;shape=(E,*([1]*(v.ndim-1)))
            first.mul_(.9).add_(grad,alpha=.1);second.mul_(.999).addcmul_(grad,grad,value=.001)
            adaptive=(first/(1-.9**step))/((second/(1-.999**step)).sqrt()+eps.reshape(shape))
            update=torch.where(adam.reshape(shape),adaptive,grad)*lr.reshape(shape)
            old=v.clone();v.sub_(update);total+=(v-old).reshape(E,-1).square().sum(-1)
    return total.sqrt()

def train(torch,p,parts,out,device):
    fit,dev=parts['fit'],parts['development'];net=f.model_class(torch)(p,fit['y'],device)
    coefs=g.initialize(torch,net,p,fit);E=len(net.members)
    for i,m in enumerate(net.members):m.update({k:p['arms'][i//len(p['seeds'])][k] for k in ('optimizer','lr','eps')})
    a.n.p.save_json(out/'fit_initialization.json',dict(fit_only=True,analytic_coefficients_supplied=False,coefficients=coefs))
    parameters=list(net.parameters());moments=[(torch.zeros_like(v),torch.zeros_like(v)) for v in parameters];initial=[v.detach().clone() for v in parameters]
    best=np.full(E,np.inf);chosen=np.zeros(E,int);selected=copy.deepcopy(net.state_dict());curve=[];rng=np.random.default_rng(951999);started=time.monotonic()
    bins=torch.floor(fit['log']).cpu().numpy();pools=[np.flatnonzero(bins[:,k]==v) for k in range(11) for v in np.unique(bins[:,k])]
    gradnorm=torch.zeros(E,device=device);updatenorm=torch.zeros_like(gradnorm)
    for step in range(p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.75:raise TimeoutError('51i budget')
        if step:
            ids=torch.tensor([rng.choice(pools[k]) for k in rng.integers(0,len(pools),p['batch_size'])],device=device)
            raw=net.raw(b.inputs(torch,fit['g'][ids])[None].expand(E,-1,-1));pred=10**raw.clamp_max(0)
            loss=(raw-fit['log'][ids].float()).square().mean((1,2))/900+(pred-fit['y'][ids]).square().mean((1,2))
            net.zero_grad(set_to_none=True);loss.sum().backward()
            gradnorm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in parameters).sqrt();factor=(p['gradient_clip']/gradnorm.clamp_min(1e-12)).clamp_max(1)
            for v in parameters:
                assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
            updatenorm=packed_step(torch,parameters,moments,net.members,step)
            assert all(bool(torch.isfinite(v).all()) for v in parameters)
        if step%500==0:print(f'[GIADA51i] preservation {step}/{p["updates"]}',flush=True)
        if step not in p['checkpoints']:continue
        with torch.no_grad():
            score=torch.zeros(E,device=device);n=0
            for start in range(0,len(dev['g']),512):
                y=dev['y'][start:start+512];raw=net.raw(b.inputs(torch,dev['g'][start:start+512])[None].expand(E,-1,-1));pred=10**raw.clamp_max(0)
                score+=((raw-dev['log'][start:start+512]).square().mean((1,2))/900+(pred-y).square().mean((1,2)))*len(y);n+=len(y)
            score/=n;drift=sum((v-old).reshape(E,-1).square().sum(-1) for v,old in zip(parameters,initial)).sqrt()
        values=score.cpu().numpy();mask=values<best
        for key,v in net.state_dict().items():selected[key][mask]=v[mask].detach()
        best=np.minimum(best,values);chosen[mask]=step
        curve.append(dict(update=step,common_score=values.tolist(),gradient_norm=gradnorm.detach().cpu().tolist(),update_norm=updatenorm.cpu().tolist(),parameter_drift=drift.cpu().tolist()))
        torch.save(net.state_dict(),out/f'checkpoint_{step}.pt');torch.save(dict(model=net.state_dict(),moments=moments,update=step,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve)
        if Path('/kaggle/working').is_dir():
            archive=Path(shutil.make_archive('/kaggle/working/task51i_checkpoint_latest','zip',out))
            with archive.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
            a.n.p.save_json(Path('/kaggle/working/task51i_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=archive.stat().st_size,update=step,partial=True))
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),fresh_accessed=False))
    return net

def run(torch,root,source,c,d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);start=time.monotonic()
    a.n.p.save_json(out/'preregistration.json',p);a.n.p.save_json(out/'controls.json',b.controls(torch,device))
    models,_,_=a.load_frozen(torch,root,source,c,d,out,device);before=a.n.weights_hash(models['P']);assert before==p['weights_sha256']
    with torch.no_grad():
        parts=b.dataset(torch,p,device)
        for value in parts.values():value['log']=f.log_targets(torch,value['g'])
        f.solver_controls(parts,out)
    net=train(torch,p,parts,out,device)
    with torch.no_grad():
        net.load_state_dict(torch.load(out/f'checkpoint_{p["updates"]}.pt',map_location=device,weights_only=True))
        finalrows=b.evaluate(torch,p,net,parts,out,device)
        (out/'evaluation.json.gz').rename(out/'final_evaluation.json.gz');(out/'fresh_predictions.npz').rename(out/'final_predictions.npz')
        net.load_state_dict(torch.load(out/'selected.pt',map_location=device,weights_only=True));rows=b.evaluate(torch,p,net,parts,out,device);b.benchmark(torch,p,net,parts,out,device)
        details=[];data=parts['fresh']
        for label,filename in [('initial','checkpoint_0.pt'),('final',f'checkpoint_{p["updates"]}.pt'),('selected','selected.pt')]:
            net.load_state_dict(torch.load(out/filename,map_location=device,weights_only=True))
            raw=torch.cat([net.raw(b.inputs(torch,data['g'][i:i+512])[None].expand(len(net.members),-1,-1)).double() for i in range(0,len(data['g']),512)],1)
            for i,m in enumerate(net.members):
                for group,lo,hi in [('reachable',0,data['reachable']),('synthetic',data['reachable'],len(data['g']))]:
                    err=raw[i,lo:hi]-data['log'][lo:hi];details.append(dict(**m,checkpoint=label,group=group,exact_raw_log_rmse=float(err.square().mean().sqrt())))
        a.n.p.save_json(out/'exact_checkpoint_evaluation.json',details)
    assert a.n.weights_hash(models['P'])==before
    assert sum(x.stat().st_size for x in out.rglob('*') if x.is_file())<p['output_budget_mb']*1024**2
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='51i',P_frozen=True,finite_parameters=all(bool(torch.isfinite(v).all()) for v in net.parameters()),thresholds_changed=False,temporal_dynamics_tested=False,evaluation_rows=len(rows),elapsed_seconds=time.monotonic()-start,members=net.members,limitations=p['limitations'],stop_policy=p['stop_policy'],final_checkpoint_trajectory_passes=[dict(arm=r['arm'],seed=r['seed'],passed=sum(r['passed_amplitude_only'])) for r in finalrows if r['group']=='reachable_trajectory_current']))
