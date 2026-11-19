import copy, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
from . import task51f_local_targets as f
b=f.b
a=f.a

def plan(root):
    p=json.loads((root/'experiments/task51g_preregistration_v1.json').read_text())
    assert p['task']=='51g' and not p['thresholds_changed']
    return p

def initialize(torch,net,p,fit):
    coefficients=[]
    for k,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):
        x=np.column_stack((fit['g'][:,int(lo):int(hi)].double().log10().cpu().numpy()/10,np.ones(len(fit['g']))))
        y=fit['log'][:,k].cpu().numpy()
        c,_,rank,_=np.linalg.lstsq(x,y,rcond=None)
        coefficients.append(dict(channel=k,values=c.tolist(),rank=int(rank)))
    with torch.no_grad():
        for i,m in enumerate(net.members):
            arm=p['arms'][i//len(p['seeds'])]
            m.update(initialization=arm['initialization'],schedule=arm['schedule'])
            if arm['initialization']=='fit':
                net.o[i].zero_();net.skip[i].zero_()
                for k,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):
                    c=coefficients[k]['values']
                    net.skip[i,int(lo):int(hi),k]=torch.tensor(c[:-1],device=net.skip.device)
                    net.bias[i,0,k]=c[-1]
    return coefficients

def train(torch,p,parts,out,device):
    fit,dev=parts['fit'],parts['development'];net=f.model_class(torch)(p,fit['y'],device)
    coefficients=initialize(torch,net,p,fit);E=len(net.members)
    a.n.p.save_json(out/'fit_initialization.json',dict(fit_only=True,analytic_coefficients_supplied=False,coefficients=coefficients))
    optim=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(951799)
    best=np.full(E,np.inf);chosen=np.zeros(E,int);selected=copy.deepcopy(net.state_dict());curve=[]
    bins=torch.floor(fit['log']).cpu().numpy();pools=[np.flatnonzero(bins[:,k]==v) for k in range(11) for v in np.unique(bins[:,k])]
    decay=torch.tensor([m['schedule']=='decay' for m in net.members],device=device)
    started=time.monotonic()
    for update in range(p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.75:raise TimeoutError('51g training budget')
        if update:
            ids=torch.tensor([rng.choice(pools[k]) for k in rng.integers(0,len(pools),p['batch_size'])],device=device)
            raw=net.raw(b.inputs(torch,fit['g'][ids])[None].expand(E,-1,-1));pred=10**raw.clamp_max(0)
            loss=(raw-fit['log'][ids].float()).square().mean((1,2))/900+(pred-fit['y'][ids]).square().mean((1,2))
            optim.zero_grad(set_to_none=True);loss.sum().backward()
            norm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in net.parameters()).sqrt()
            factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
            for v in net.parameters():
                assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
            previous=[v.detach().clone() for v in net.parameters()];optim.step()
            # Scale each independent Adam parameter delta, not its gradient/moments.
            scale=torch.where(decay,torch.full((E,),.01**(update/p['updates']),device=device),torch.ones(E,device=device))
            with torch.no_grad():
                for v,old in zip(net.parameters(),previous):v.copy_(old+(v-old)*scale.reshape(E,*([1]*(v.ndim-1))))
            assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
        if update%500==0:print(f'[GIADA51g] training {update}/{p["updates"]}',flush=True)
        if update not in p['checkpoints']:continue
        with torch.no_grad():
            score=torch.zeros(E,device=device);n=0
            for start in range(0,len(dev['g']),512):
                y=dev['y'][start:start+512];raw=net.raw(b.inputs(torch,dev['g'][start:start+512])[None].expand(E,-1,-1));pred=10**raw.clamp_max(0)
                score+=((raw-dev['log'][start:start+512]).square().mean((1,2))/900+(pred-y).square().mean((1,2)))*len(y);n+=len(y)
            score/=n
        values=score.cpu().numpy();mask=values<best
        for key,v in net.state_dict().items():selected[key][mask]=v[mask].detach()
        best=np.minimum(best,values);chosen[mask]=update;curve.append(dict(update=update,common_score=values.tolist()))
        torch.save(net.state_dict(),out/f'checkpoint_{update}.pt')
        torch.save(dict(model=net.state_dict(),optimizer=optim.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt')
        a.n.p.save_json(out/'learning_curve.json',curve)
        if Path('/kaggle/working').is_dir():
            archive=Path(shutil.make_archive('/kaggle/working/task51g_checkpoint_latest','zip',out))
            with archive.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
            a.n.p.save_json(Path('/kaggle/working/task51g_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=archive.stat().st_size,update=update,partial=True))
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt')
    a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),fresh_accessed=False,initialization_eligible=True))
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
        rows=b.evaluate(torch,p,net,parts,out,device);b.benchmark(torch,p,net,parts,out,device)
        data=parts['fresh'];details=[]
        # Selection is already frozen. Compare initial/final/selected errors only now.
        for label,filename in [('initial','checkpoint_0.pt'),('final',f'checkpoint_{p["updates"]}.pt'),('selected','selected.pt')]:
            net.load_state_dict(torch.load(out/filename,map_location=device,weights_only=True))
            raw=torch.cat([net.raw(b.inputs(torch,data['g'][i:i+512])[None].expand(len(net.members),-1,-1)).double() for i in range(0,len(data['g']),512)],1)
            for i,m in enumerate(net.members):
                for name,lo,hi in [('reachable',0,data['reachable']),('synthetic',data['reachable'],len(data['g']))]:
                    err=raw[i,lo:hi]-data['log'][lo:hi]
                    details.append(dict(**m,checkpoint=label,group=name,exact_raw_log_rmse=float(err.square().mean().sqrt()),per_channel_rmse=err.square().mean(0).sqrt().cpu().tolist()))
        a.n.p.save_json(out/'exact_checkpoint_evaluation.json',details)
    assert a.n.weights_hash(models['P'])==before
    assert sum(x.stat().st_size for x in out.rglob('*') if x.is_file())<p['output_budget_mb']*1024**2
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='51g',P_frozen=True,finite_parameters=all(bool(torch.isfinite(v).all()) for v in net.parameters()),thresholds_changed=False,temporal_dynamics_tested=False,evaluation_rows=len(rows),elapsed_seconds=time.monotonic()-start,members=net.members,limitations=p['limitations']))
