import copy,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from . import task51b_small_openings as b
a=b.a
def plan(root):
    p=json.loads((root/'experiments/task51e_preregistration_v1.json').read_text());assert p['task']=='51e' and not p['thresholds_changed'];return p
def model_class(torch):
    class Model(torch.nn.Module):
        def __init__(self,p,y,device):
            super().__init__();self.members=[dict(representation=r,loss=l,seed=s) for r in p['representations'] for l in p['losses'] for s in p['seeds']];E=len(self.members)
            for name,shape in [('w',(E,18,88)),('b',(E,1,88)),('o',(E,88,11)),('skip',(E,18,11)),('bias',(E,1,11))]:setattr(self,name,torch.nn.Parameter(torch.zeros(shape,device=device)))
            wm=torch.zeros_like(self.w);om=torch.zeros_like(self.o)
            with torch.no_grad():
                for i,m in enumerate(self.members):
                    gen=torch.Generator();gen.manual_seed(m['seed'])
                    for name in ('w','o','skip'):getattr(self,name)[i].copy_((torch.randn(getattr(self,name)[i].shape,generator=gen)*.02).to(device))
                    self.bias[i,0].copy_(y.clamp_min(1e-30).log10().mean(0))
                    if m['representation']=='local8':
                        for ch,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):wm[i,int(lo):int(hi),8*ch:8*(ch+1)]=1;om[i,8*ch:8*(ch+1),ch]=1
                    elif m['representation']!='linear':
                        width=32 if m['representation']=='shared32' else 88;wm[i,:,:width]=1;om[i,:width,:]=1
            self.register_buffer('wm',wm);self.register_buffer('om',om);sm=torch.ones_like(self.skip)
            for i,m in enumerate(self.members):
                if m['representation']=='local8':
                    sm[i].zero_()
                    for ch,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):sm[i,int(lo):int(hi),ch]=1
            self.register_buffer('sm',sm)
        def raw(self,x):return torch.bmm(torch.nn.functional.silu(torch.bmm(x,self.w*self.wm)+self.b),self.o*self.om)+torch.bmm(x,self.skip*self.sm)+self.bias
        def forward(self,x):return 10**self.raw(x).clamp_max(0)
    return Model
def train(torch,p,parts,out,device):
    fit=parts['fit'];dev=parts['development'];net=model_class(torch)(p,fit['y'],device);E=len(net.members);optim=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(951599);best=np.full(E,np.inf);selected=copy.deepcopy(net.state_dict());chosen=np.zeros(E,int);curve=[];bins=torch.floor(fit['y'].clamp_min(1e-30).log10()).cpu().numpy();pools=[np.flatnonzero(bins[:,ch]==v) for ch in range(11) for v in np.unique(bins[:,ch])];raw_mode=torch.tensor([m['loss']=='raw_log' for m in net.members],device=device);started=time.monotonic()
    for update in range(1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.75:raise TimeoutError('51e budget')
        idx=torch.tensor([rng.choice(pools[k]) for k in rng.integers(0,len(pools),p['batch_size'])],device=device);y=fit['y'][idx];x=b.inputs(torch,fit['g'][idx])[None].expand(E,-1,-1);raw=net.raw(x);pr=10**raw.clamp_max(0);target=y.clamp_min(1e-30).log10();rawloss=(raw-target).square().mean((1,2))/900;hybrid=((pr+1e-30).log10()-target).square().mean((1,2))/900+(pr-y).square().mean((1,2));loss=torch.where(raw_mode,rawloss,hybrid)
        optim.zero_grad(set_to_none=True);loss.sum().backward();norm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in net.parameters()).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in net.parameters():assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
        optim.step();assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
        if update%500==0:print(f'[GIADA51e] training {update}/{p["updates"]}',flush=True)
        if update in p['checkpoints']:
            with torch.no_grad():
                score=torch.zeros(E,device=device);rawscore=score.clone();positive=score.clone();n=0
                for start in range(0,len(dev['g']),512):
                    y=dev['y'][start:start+512];raw=net.raw(b.inputs(torch,dev['g'][start:start+512])[None].expand(E,-1,-1));pr=10**raw.clamp_max(0);target=y.clamp_min(1e-30).log10();score+=(((pr+1e-30).log10()-target).square().mean((1,2))/900+(pr-y).square().mean((1,2)))*len(y);rawscore+=(raw-target).square().mean((1,2))*len(y);positive+=(raw>0).float().mean((1,2))*len(y);n+=len(y)
                score/=n;rawscore/=n;positive/=n
            values=score.cpu().numpy();mask=values<best
            for key,v in net.state_dict().items():selected[key][mask]=v[mask].detach()
            best=np.minimum(best,values);chosen[mask]=update;curve.append(dict(update=update,common_score=values.tolist(),raw_log_mse=rawscore.cpu().tolist(),raw_positive_fraction=positive.cpu().tolist()));torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=optim.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve)
            if Path('/kaggle/working').is_dir():
                archive=Path(shutil.make_archive('/kaggle/working/task51e_checkpoint_latest','zip',out))
                with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                a.n.p.save_json(Path('/kaggle/working/task51e_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=archive.stat().st_size,update=update,partial=True))
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),fresh_accessed=False));return net
def run(torch,root,source,c,d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);start=time.monotonic();a.n.p.save_json(out/'preregistration.json',p);a.n.p.save_json(out/'controls.json',b.controls(torch,device));models,_,_=a.load_frozen(torch,root,source,c,d,out,device);before=a.n.weights_hash(models['P']);assert before==p['weights_sha256']
    with torch.no_grad():parts=b.dataset(torch,p,device)
    print('[GIADA51e] data ready; readout conditioning matrix',flush=True);net=train(torch,p,parts,out,device)
    with torch.no_grad():rows=b.evaluate(torch,p,net,parts,out,device);b.benchmark(torch,p,net,parts,out,device)
    # Fit-only solver control estimates all coefficients without supplying channel powers.
    x=np.column_stack((b.inputs(torch,parts['fit']['g']).cpu().numpy(),np.ones(len(parts['fit']['g']))));y=parts['fit']['y'].clamp_min(1e-30).log10().cpu().numpy();coef,_,rank,singular=np.linalg.lstsq(x,y,rcond=None);xf=np.column_stack((b.inputs(torch,parts['fresh']['g']).cpu().numpy(),np.ones(len(parts['fresh']['g']))));pred=10**np.minimum(xf@coef,0);truth=parts['fresh']['y'].cpu().numpy();np.savez_compressed(out/'fit_only_linear_solver.npz',coefficients=coef,predictions=pred,target=truth)
    a.n.p.save_json(out/'linear_solver_control.json',dict(fit_only=True,analytic_powers_supplied=False,rank=int(rank),singular_values=singular.tolist(),fresh_opening_rmse=float(np.sqrt(np.mean((pred-truth)**2))),fresh_log_rmse=float(np.sqrt(np.mean((np.log10(pred+1e-30)-np.log10(truth+1e-30))**2))),deployment_qualified=False))
    assert a.n.weights_hash(models['P'])==before;assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='51e',P_frozen=True,finite_parameters=all(bool(torch.isfinite(v).all()) for v in net.parameters()),thresholds_changed=False,temporal_dynamics_tested=False,evaluation_rows=len(rows),elapsed_seconds=time.monotonic()-start,members=net.members,limitations=p['limitations']))
