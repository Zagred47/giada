"""Orthogonal initial anchor, loss calibration and stateful TBPTT contrasts."""
import copy,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from . import task50_compact_memory as t
q=t.q;a=t.a

def plan(root):
    p=json.loads((root/'experiments/task51_preregistration_v1.json').read_text());assert p['task']=='51' and not p['thresholds_changed'];assert p['weights_sha256']==q.plan(root)['weights_sha256'];assert len(p['arms'])==8;return p

def paired_model(torch,p,device):
    net=t.model_class(torch)(p,device);ns=len(p['seeds'])
    with torch.no_grad():
        for par in net.parameters():
            for arm in range(1,len(p['arms'])):par[arm*ns:(arm+1)*ns].copy_(par[:ns])
    for member in net.members:
        spec=next(v for v in p['arms'] if v['id']==member['arm']);member.update({k:spec[k] for k in ('initial_anchor','calibrated_loss','horizon_ticks')})
    return net

def log_error(torch,got,target):return ((torch.log10(got+1e-8)-torch.log10(target+1e-8))/10).square().mean((1,2))

def train_update(torch,net,p,data,ids,starts,global_scale,optimizer):
    E=len(net.members);device=ids.device;h=net.initialize(data['g'][starts,ids][None].expand(E,-1,-1));lengths=torch.tensor([m['horizon_ticks'] for m in net.members],device=device)
    calibration=torch.tensor([m['calibrated_loss'] for m in net.members],device=device);anchor=torch.tensor([m['initial_anchor'] for m in net.members],device=device)
    maxlen=int(lengths.max());local_scales=[]
    for length in sorted(set(lengths.cpu().tolist())):
        ix=starts[None]+torch.arange(length,device=device)[:,None];target=t.current(torch,data['op'][ix,ids[None]],data['v'][ix,ids[None]])
        local_scales.append((length,target.abs().amax(0).clamp_min(1e-7)))
    scale=torch.stack([next(v for length,v in local_scales if length==m['horizon_ticks']) for m in net.members])
    optimizer.zero_grad(set_to_none=True);block=torch.zeros(E,device=device);total=torch.zeros(E,device=device)
    for k in range(maxlen):
        v=data['v'][starts+k,ids];ca=data['ca'][starts+k,ids];target=data['op'][starts+k,ids];op=net.opening(h);difference=t.current(torch,op,v[None])-t.current(torch,target,v)
        legacy=(difference/global_scale).square().mean((1,2))+.1*(op-target).square().mean((1,2))
        calibrated=torch.asinh(difference/scale).square().mean((1,2))+.1*log_error(torch,op,target)
        value=torch.where(calibration,calibrated,legacy)*(k<lengths)/lengths
        if k==0:value=value+4*log_error(torch,op,target)*anchor
        block=block+value;total+=value.detach()
        proposed=net.step(t.features(torch,v,ca)[None].expand(E,-1,-1),h);h=torch.where((k<lengths)[:,None,None],proposed,h)
        if (k+1)%p['bptt_ticks']==0 or k+1==maxlen:
            block.sum().backward();h=h.detach();block=torch.zeros(E,device=device)
    norms=torch.zeros(E,device=device)
    for par in net.parameters():
        assert bool(torch.isfinite(par.grad).all());norms+=par.grad.reshape(E,-1).square().sum(-1)
    factor=(p['gradient_clip']/norms.sqrt().clamp_min(1e-12)).clamp_max(1)
    for par in net.parameters():par.grad.mul_(factor.reshape(E,*([1]*(par.ndim-1))))
    optimizer.step();assert all(bool(torch.isfinite(par).all()) for par in net.parameters());return total

def development(torch,net,data,dev,p):
    scores=[];initial=[];E=len(net.members)
    for start in (0,1500,2750,6500):
        if start>=len(data['v']):continue
        length=min(512,len(data['v'])-start);target=t.current(torch,data['op'][start:start+length,dev],data['v'][start:start+length,dev]);scale=target.abs().amax(0).clamp_min(1e-7);h=net.initialize(data['g'][start,dev][None].expand(E,-1,-1));loss=torch.zeros(E,device=dev.device)
        initial.append(log_error(torch,net.opening(h),data['op'][start,dev]))
        for k in range(length):
            v=data['v'][start+k,dev];ca=data['ca'][start+k,dev];got=t.current(torch,net.opening(h),v[None]);loss+=((got-target[k])/scale).square().mean((1,2));h=net.step(t.features(torch,v,ca)[None].expand(E,-1,-1),h)
        scores.append(loss/length)
    return torch.stack(scores).mean(0),torch.stack(initial).mean(0)

def train(torch,p,data,out,device):
    net=paired_model(torch,p,device);E=len(net.members);fit=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='fit'],device=device);dev=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='development'],device=device);global_scale=t.current(torch,data['op'][:,fit],data['v'][:,fit]).abs().amax((0,1)).clamp_min(1e-7)
    optimizer=torch.optim.Adam(net.parameters(),lr=p['learning_rate']);rng=np.random.default_rng(951001);best=np.full(E,np.inf);selected=copy.deepcopy(net.state_dict());chosen=np.zeros(E,dtype=int);curve=[];started=time.monotonic()
    for update in range(1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.75:raise TimeoutError('51 training budget, checkpoints retained')
        ids=fit[torch.tensor(rng.integers(0,len(fit),p['batch_size']),device=device)];starts=torch.tensor(rng.integers(0,len(data['v'])-p['chunk_ticks'],p['batch_size']),device=device)
        loss=train_update(torch,net,p,data,ids,starts,global_scale,optimizer)
        if update%20==0:print(f'[GIADA51] training {update}/{p["updates"]} mean_loss={loss.mean().item():.6g}',flush=True)
        if update in p['checkpoints']:
            with torch.inference_mode():score,initial=development(torch,net,data,dev,p)
            values=score.cpu().numpy();improved=values<best
            for key,value in net.state_dict().items():selected[key][improved]=value[improved].detach()
            best=np.minimum(best,values);chosen[improved]=update;curve.append(dict(update=update,development=values.tolist(),initial_log_error=initial.cpu().tolist(),elapsed_seconds=time.monotonic()-started));torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),update=update,rng=rng.bit_generator.state),out/'resumable_training.pt');a.n.p.save_json(out/'learning_curve.json',curve)
            public=Path('/kaggle/working')
            if public.is_dir():
                archive=Path(shutil.make_archive(str(public/'task51_checkpoint_latest'),'zip',out))
                with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                a.n.p.save_json(public/'task51_checkpoint_receipt.json',dict(valid=True,update=update,sha256=digest,bytes=archive.stat().st_size,partial=True,selection_frozen=False))
                print(f'[GIADA51] checkpoint {update} persisted outside temporary workspace',flush=True)
    net.load_state_dict(selected);torch.save(selected,out/'selected.pt');a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),development=best.tolist(),fresh_accessed=False));return net

def probes(torch,net,data,p,out,device):
    E=len(net.members);ids=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='fresh'][:6],device=device);records=[];latent=[];reset_records=[]
    for cut in p['evaluation_cuts_ms']:
        idx=round(cut/p['dt_ms']);g=data['g'][idx,ids];h=net.initialize(g[None].expand(E,-1,-1));target=data['op'][idx,ids];got=net.opening(h)
        records.append(dict(cut_ms=cut,predicted_openings=got.cpu().tolist(),target_openings=target.cpu().tolist(),log_error=log_error(torch,got,target).cpu().tolist()))
        trace=[]
        for tick in range(round(p.get('probe_ms',1000)/p['dt_ms'])+1):
            if tick%100==0:trace.append(h.cpu().numpy())
            v=torch.full((len(ids),),-65.,device=device);ca=torch.full_like(v,5e-5);h=net.step(t.features(torch,v,ca)[None].expand(E,-1,-1),h)
        latent.append(np.stack(trace))
        intact=net.initialize(g[None].expand(E,-1,-1));reset=intact.clone();maximum=torch.zeros((E,len(ids),11),device=device)
        for tick in range(round(p.get('probe_ms',1000)/p['dt_ms'])+1):
            ms=tick*p['dt_ms']
            if tick==round(100/p['dt_ms']):reset=torch.zeros_like(reset)
            vv,cc=q.future_input('pulse_recovery',ms);v=torch.full((len(ids),),vv,device=device);ca=torch.full_like(v,cc)
            difference=(t.current(torch,net.opening(intact),v[None])-t.current(torch,net.opening(reset),v[None])).abs()
            if ms>=100:maximum=torch.maximum(maximum,difference)
            if tick in [round(x/p['dt_ms']) for x in (100,110,200,500,1000)]:reset_records.append(dict(cut_ms=cut,time_ms=ms,absolute_current_difference_ma_cm2=difference.cpu().tolist(),post_reset_max_difference_ma_cm2=maximum.cpu().tolist(),latent_max_difference=(intact-reset).abs().amax(-1).cpu().tolist(),reference='same trained model intact vs zero-reset; raw difference not teacher error'))
            x=t.features(torch,v,ca)[None].expand(E,-1,-1);intact=net.step(x,intact);reset=net.step(x,reset)
    a.n.p.save_json(out/'initial_probe.json',records);np.savez_compressed(out/'latent_rest_probes.npz',latent=np.stack(latent),sample_dt_ms=10.,fresh_case_ids=ids.cpu().numpy())
    a.n.p.save_json(out/'post_reset_probe.json',reset_records)

def run(torch,root,source,c,d,out,device='cuda',override=None):
    p=plan(root) if override is None else override;out.mkdir(exist_ok=True);started=time.monotonic();a.n.p.save_json(out/'preregistration.json',p);models,_,_=a.load_frozen(torch,root,source,c,d,out,device);P=models['P'];before=a.n.weights_hash(P)
    with torch.no_grad():data=t.dataset(torch,p,device)
    print('[GIADA51] dataset ready; packed factorial training',flush=True);net=train(torch,p,data,out,device)
    with torch.inference_mode():
        probes(torch,net,data,p,out,device);rows=t.evaluate(torch,p,net,P,data,out,device);t.benchmark(torch,p,net,P,out,device)
        v=torch.full((len(net.members),p['cost']['batch'],18),.5,device=device);times=[]
        for _ in range(3):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();net.initialize(v)
            if device=='cuda':torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        a.n.p.save_json(out/'encoding_cost.json',dict(samples_ms=times,packed_members=len(net.members),not_per_member=True))
    assert a.n.weights_hash(P)==before==p['weights_sha256'];assert all(bool(torch.isfinite(v).all()) for v in net.parameters())
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file())<p['output_budget_mb']*1024**2
    a.n.p.save_json(out/'final_report.json',dict(valid=True,complete=True,task='51',P_frozen=True,finite_parameters=True,thresholds_changed=False,evaluation_rows=len(rows),elapsed_seconds=time.monotonic()-started,members=net.members,limitations=p['limitations'],speedup_claimed=False))
