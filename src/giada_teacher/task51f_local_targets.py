import copy,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from . import task51b_small_openings as b
a=b.a
def plan(root):
    p=json.loads((root/'experiments/task51f_preregistration_v1.json').read_text());assert p['task']=='51f' and not p['thresholds_changed'];return p
def log_targets(torch,g):
    w=torch.zeros((18,11),device=g.device,dtype=torch.float64)
    for k,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):
        w[int(lo),k]=b.q.ionic.POWERS[k]
        if hi-lo==2:w[int(lo)+1,k]=1
    assert bool((g>0).all()),'Exact zero input requires separate target contract'
    return g.double().log10()@w
def model_class(torch):
    class Model(torch.nn.Module):
        def __init__(self,p,y,device):
            super().__init__();self.members=[dict(arm=arm['id'],representation=arm['id'],loss=arm['objective'],units=arm['units'],target=arm['target'],seed=s) for arm in p['arms'] for s in p['seeds']];E=len(self.members)
            for name,shape in [('w',(E,18,264)),('b',(E,1,264)),('o',(E,264,11)),('skip',(E,18,11)),('bias',(E,1,11))]:setattr(self,name,torch.nn.Parameter(torch.zeros(shape,device=device)))
            wm=torch.zeros_like(self.w);om=torch.zeros_like(self.o);sm=torch.zeros_like(self.skip)
            with torch.no_grad():
                for i,m in enumerate(self.members):
                    gen=torch.Generator();gen.manual_seed(m['seed'])
                    for name in ('w','o','skip'):getattr(self,name)[i].copy_((torch.randn(getattr(self,name)[i].shape,generator=gen)*.02).to(device))
                    self.bias[i,0].copy_(y.clamp_min(1e-30).log10().mean(0))
                    for k,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):wm[i,int(lo):int(hi),24*k:24*k+m['units']]=1;om[i,24*k:24*k+m['units'],k]=1;sm[i,int(lo):int(hi),k]=1
            self.register_buffer('wm',wm);self.register_buffer('om',om);self.register_buffer('sm',sm)
        def raw(self,x):return torch.bmm(torch.nn.functional.silu(torch.bmm(x,self.w*self.wm)+self.b),self.o*self.om)+torch.bmm(x,self.skip*self.sm)+self.bias
        def forward(self,x):return 10**self.raw(x).clamp_max(0)
    return Model
def train(torch,p,parts,out,device):
    fit=parts['fit'];dev=parts['development'];net=model_class(torch)(p,fit['y'],device);E=len(net.members);optim=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(951699);best=np.full(E,np.inf);selected=copy.deepcopy(net.state_dict());chosen=np.zeros(E,int);curve=[]
    bins=torch.floor(fit['log']).cpu().numpy();pools=[np.flatnonzero(bins[:,k]==v) for k in range(11) for v in np.unique(bins[:,k])];exact=torch.tensor([m['target']=='exact' for m in net.members],device=device)[:,None,None];legacy=torch.tensor([m['loss']=='legacy' for m in net.members],device=device);started=time.monotonic()
    for update in range(1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.75:raise TimeoutError('51f budget')
        ids=torch.tensor([rng.choice(pools[k]) for k in rng.integers(0,len(pools),p['batch_size'])],device=device);y=fit['y'][ids];raw=net.raw(b.inputs(torch,fit['g'][ids])[None].expand(E,-1,-1));pred=10**raw.clamp_max(0);floor=y.clamp_min(1e-30).log10();target=torch.where(exact,fit['log'][ids].float(),floor);logloss=torch.where(legacy[:,None,None],(pred+1e-30).log10(),raw)
        loss=(logloss-target).square().mean((1,2))/900+(pred-y).square().mean((1,2));optim.zero_grad(set_to_none=True);loss.sum().backward();norm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in net.parameters()).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in net.parameters():assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
        optim.step();assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
        if update%500==0:print(f'[GIADA51f] training {update}/{p["updates"]}',flush=True)
        if update in p['checkpoints']:
            with torch.no_grad():
                score=torch.zeros(E,device=device);n=0
                for start in range(0,len(dev['g']),512):
                    y=dev['y'][start:start+512];raw=net.raw(b.inputs(torch,dev['g'][start:start+512])[None].expand(E,-1,-1));pred=10**raw.clamp_max(0);score+=((raw-dev['log'][start:start+512]).square().mean((1,2))/900+(pred-y).square().mean((1,2)))*len(y);n+=len(y)
                score/=n
            values=score.cpu().numpy();mask=values<best
            for key,v in net.state_dict().items():selected[key][mask]=v[mask].detach()
            best=np.minimum(best,values);chosen[mask]=update;curve.append(dict(update=update,common_score=values.tolist()));torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=optim.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve)
            if Path('/kaggle/working').is_dir():
                archive=Path(shutil.make_archive('/kaggle/working/task51f_checkpoint_latest','zip',out))
                with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                a.n.p.save_json(Path('/kaggle/working/task51f_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=archive.stat().st_size,update=update,partial=True))
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),fresh_accessed=False));return net
def solver_controls(parts,out):
    result={};coeff={};fit=parts['fit'];fresh=parts['fresh'];predictions={}
    for kind in ('floored','exact'):
        y=fit['log'].cpu().numpy() if kind=='exact' else fit['y'].clamp_min(1e-30).log10().cpu().numpy();rows=[];pred=np.empty((len(fresh['g']),11))
        for k,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):
            x=np.column_stack((fit['g'][:,int(lo):int(hi)].double().log10().cpu().numpy()/10,np.ones(len(y))));cf,_,rank,_=np.linalg.lstsq(x,y[:,k],rcond=None);xf=np.column_stack((fresh['g'][:,int(lo):int(hi)].double().log10().cpu().numpy()/10,np.ones(len(pred))));pred[:,k]=xf@cf;rows.append(dict(channel=k,coefficients=cf.tolist(),rank=int(rank)))
        target=fresh['log'].cpu().numpy();result[kind]=dict(exact_log_rmse=float(np.sqrt(np.mean((pred-target)**2))),opening_rmse=float(np.sqrt(np.mean((10**np.minimum(pred,0)-fresh['y'].cpu().numpy())**2))),fit_only=True,analytic_coefficients_supplied=False);coeff[kind]=rows;predictions[kind]=pred
    a.n.p.save_json(out/'solver_controls.json',dict(results=result,coefficients=coeff,not_recurrent_models=True));np.savez_compressed(out/'solver_predictions.npz',**predictions)
def run(torch,root,source,c,d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);start=time.monotonic();a.n.p.save_json(out/'preregistration.json',p);a.n.p.save_json(out/'controls.json',b.controls(torch,device));models,_,_=a.load_frozen(torch,root,source,c,d,out,device);before=a.n.weights_hash(models['P']);assert before==p['weights_sha256']
    with torch.no_grad():
        parts=b.dataset(torch,p,device)
        for value in parts.values():value['log']=log_targets(torch,value['g'])
        solver_controls(parts,out)
    print('[GIADA51f] data ready; capacity/target matrix',flush=True);net=train(torch,p,parts,out,device)
    with torch.no_grad():
        rows=b.evaluate(torch,p,net,parts,out,device);b.benchmark(torch,p,net,parts,out,device)
        data=parts['fresh'];raw=torch.cat([net.raw(b.inputs(torch,data['g'][i:i+512])[None].expand(len(net.members),-1,-1)).double() for i in range(0,len(data['g']),512)],1);details=[]
        for i,m in enumerate(net.members):
            for name,lo,hi in [('reachable',0,data['reachable']),('synthetic',data['reachable'],len(data['g']))]:
                target=data['log'][lo:hi];error=raw[i,lo:hi]-target
                details.append(dict(**m,group=name,exact_raw_log_rmse=float(error.square().mean().sqrt()),per_channel_rmse=error.square().mean(0).sqrt().cpu().tolist(),raw_positive_fraction=float((raw[i,lo:hi]>0).double().mean())))
        a.n.p.save_json(out/'exact_log_evaluation.json',details)
    assert a.n.weights_hash(models['P'])==before;assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='51f',P_frozen=True,finite_parameters=all(bool(torch.isfinite(v).all()) for v in net.parameters()),thresholds_changed=False,temporal_dynamics_tested=False,evaluation_rows=len(rows),elapsed_seconds=time.monotonic()-start,members=net.members,limitations=p['limitations']))
