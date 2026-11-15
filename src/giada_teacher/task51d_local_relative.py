import copy,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from . import task51b_small_openings as b
a=b.a
def plan(root):
    p=json.loads((root/'experiments/task51d_preregistration_v1.json').read_text());assert p['task']=='51d' and not p['thresholds_changed'];return p
def model(torch,p,fit,device):
    net=b.model_class(torch)(p,fit,device);E=len(net.members);wm=torch.ones_like(net.w);om=torch.ones_like(net.o);sm=torch.ones_like(net.skip)
    for i,m in enumerate(net.members):
        if m['representation']=='log_local':
            wm[i].zero_();om[i].zero_();sm[i].zero_()
            for ch,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):
                # 32 units: 3each for first10channels,2for last.
                start=min(3*ch,30);end=min(start+3,32);wm[i,int(lo):int(hi),start:end]=1;om[i,start:end,ch]=1;sm[i,int(lo):int(hi),ch]=1
            with torch.no_grad():net.bias[i,0].copy_(fit.clamp_min(1e-30).log10().mean(0))
    net.register_buffer('wm',wm);net.register_buffer('om',om);net.register_buffer('sm',sm)
    import types
    def forward(self,x):
        raw=torch.bmm(torch.nn.functional.silu(torch.bmm(x,self.w*self.wm)+self.b),self.o*self.om)+torch.bmm(x,self.skip*self.sm)+self.bias
        return 10**raw.clamp_max(0)
    net.forward=types.MethodType(forward,net);return net
def losses(torch,pr,y):
    absolute=(pr-y).square().mean((1,2));log=((pr+1e-30).log10()-(y+1e-30).log10()).square().mean((1,2))/900;relative=torch.asinh((pr-y)/y.clamp_min(1e-4)).square().mean((1,2));return absolute+log,relative
def train(torch,p,parts,out,device):
    fit=parts['fit'];dev=parts['development'];net=model(torch,p,fit['y'],device);E=len(net.members);optim=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(951499);best=np.full(E,np.inf);selected=copy.deepcopy(net.state_dict());chosen=np.zeros(E,int);curve=[]
    bins=torch.floor(fit['y'].clamp_min(1e-30).log10()).cpu().numpy();pools=[np.flatnonzero(bins[:,ch]==v) for ch in range(11) for v in np.unique(bins[:,ch])];relative=torch.tensor([m['loss']=='relative_hybrid' for m in net.members],device=device);started=time.monotonic()
    for update in range(1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.75:raise TimeoutError('51d budget')
        idx=torch.tensor([rng.choice(pools[k]) for k in rng.integers(0,len(pools),p['batch_size'])],device=device);y=fit['y'][idx];pr=net(b.inputs(torch,fit['g'][idx])[None].expand(E,-1,-1));base,rel=losses(torch,pr,y);loss=base+relative*rel
        optim.zero_grad(set_to_none=True);loss.sum().backward();norm=sum(v.grad.reshape(E,-1).square().sum(-1) for v in net.parameters()).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in net.parameters():assert bool(torch.isfinite(v.grad).all());v.grad.mul_(factor.reshape(E,*([1]*(v.ndim-1))))
        optim.step();assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
        if update%500==0:print(f'[GIADA51d] training {update}/{p["updates"]}',flush=True)
        if update in p['checkpoints']:
            with torch.no_grad():
                score=torch.zeros(E,device=device);n=0
                for start in range(0,len(dev['g']),512):
                    y=dev['y'][start:start+512];pr=net(b.inputs(torch,dev['g'][start:start+512])[None].expand(E,-1,-1));base,rel=losses(torch,pr,y);score+=(base+rel)*len(y);n+=len(y)
                score/=n
            values=score.cpu().numpy();mask=values<best
            for key,v in net.state_dict().items():selected[key][mask]=v[mask].detach()
            best=np.minimum(best,values);chosen[mask]=update;curve.append(dict(update=update,common_development_score=values.tolist()));torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=optim.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve)
            if Path('/kaggle/working').is_dir():
                archive=Path(shutil.make_archive('/kaggle/working/task51d_checkpoint_latest','zip',out))
                with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                a.n.p.save_json(Path('/kaggle/working/task51d_checkpoint_receipt.json'),dict(valid=True,sha256=digest,bytes=archive.stat().st_size,update=update,partial=True))
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),fresh_accessed=False));return net
def run(torch,root,source,c,d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);start=time.monotonic();a.n.p.save_json(out/'preregistration.json',p);a.n.p.save_json(out/'controls.json',b.controls(torch,device));models,_,_=a.load_frozen(torch,root,source,c,d,out,device);before=a.n.weights_hash(models['P']);assert before==p['weights_sha256']
    with torch.no_grad():parts=b.dataset(torch,p,device)
    print('[GIADA51d] data ready; local/relative factorial',flush=True);net=train(torch,p,parts,out,device)
    with torch.no_grad():
        rows=b.evaluate(torch,p,net,parts,out,device);b.benchmark(torch,p,net,parts,out,device)
        g=parts['fresh']['g'].double();w=torch.zeros((18,11),device=device,dtype=torch.float64)
        for k,(lo,hi) in enumerate(zip(b.q.ionic.OFFSETS[:-1],b.q.ionic.OFFSETS[1:])):
            w[int(lo),k]=b.q.ionic.POWERS[k]
            if hi-lo==2:w[int(lo)+1,k]=1
        oracle=10**(g.clamp_min(1e-300).log10()@w);target=b.targets(torch,g);error=float((oracle-target).abs().max());assert error<1e-12
        a.n.p.save_json(out/'nonselectable_oracle.json',dict(valid=True,max_abs_error=error,learned=False,selection_eligible=False))
    assert a.n.weights_hash(models['P'])==before;assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='51d',P_frozen=True,finite_parameters=True,thresholds_changed=False,temporal_dynamics_tested=False,evaluation_rows=len(rows),elapsed_seconds=time.monotonic()-start,members=net.members,limitations=p['limitations']))
