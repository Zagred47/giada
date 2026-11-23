"""Training-only current preservation x gradient-conflict intervention."""
import copy,json,time
import numpy as np
from . import task52f_encoder_credit as f

def plan(root):
    p=json.loads((root/'experiments/task52g_preregistration_v1.json').read_text())
    assert p['task']=='52g' and not p['thresholds_changed'] and not p['abstraction_changed']
    assert all(a['bptt']==512 and not a['recalibrate'] for a in p['arms'])
    return p

def merge_gradients(torch,current,trajectory,mask):
    E=len(mask);dot=sum((a.double()*b.double()).reshape(E,-1).sum(1) for a,b in zip(current,trajectory));norm=sum(a.double().reshape(E,-1).square().sum(1) for a in current)
    factor=torch.where(mask&(dot<0),dot/norm.clamp_min(1e-30),torch.zeros_like(dot))
    projected=[b-factor.to(b.dtype).reshape(E,*([1]*(b.ndim-1)))*a for a,b in zip(current,trajectory)]
    after=sum((a.double()*b.double()).reshape(E,-1).sum(1) for a,b in zip(current,projected))
    return [a+b for a,b in zip(current,projected)],dict(dot=dot.detach(),after=after.detach(),conflict=(dot<0).detach(),norm=norm.detach())

def reference_errors(torch,p,reference,data,ids,starts):
    L=p['chunk_ticks'];ref=torch.stack([f.t.current(torch,data['op'][starts+k,ids],data['v'][starts+k,ids]) for k in range(L)]);scale=ref.abs().amax(0).clamp_min(1e-7)
    errors=[]
    with torch.no_grad():
        h=reference.initialize(data['g'][starts,ids])
        for k in range(L):
            current=f.t.current(torch,reference.opening(h),data['v'][starts+k,ids]);errors.append(((current-ref[k])/scale).square().mean(1))
            h=reference.step(f.t.features(torch,data['v'][starts+k,ids],data['ca'][starts+k,ids]),h)
    return torch.stack(errors)

def teacher(torch,p,net,data,ids,starts,cal,reference_mse,backward=True):
    E,B=ids.shape;L=p['chunk_ticks'];h=net.initialize(data['g'][starts,ids]);target=torch.stack([data['op'][starts+k,ids] for k in range(L)])
    ref=torch.stack([f.t.current(torch,target[k],data['v'][starts+k,ids]) for k in range(L)]);scale=ref.abs().amax(0).clamp_min(1e-7)
    weights=torch.where(target>=.2*cal['maxop'],5.,1.);weights/=weights.mean((0,2),keepdim=True)
    common=torch.zeros(E,device=ids.device);fam=common.clone();block=common.clone();charge=torch.zeros_like(ref[0]);peak=torch.zeros_like(ref[0]);reg=torch.zeros(E,11,device=ids.device);perchannel=reg.clone();activation=reg.clone();penalty_value=common.clone()
    extra=torch.tensor([arm['family_loss'] for arm in p['arms'] for _ in p['seeds']],device=ids.device)
    mask=torch.tensor([arm['guard'] for arm in p['arms'] for _ in p['seeds']],device=ids.device)
    width=p['preservation']['block_ticks']
    for k in range(L):
        logs=net.logopening(h);op=10**logs;err=(f.t.current(torch,op,data['v'][starts+k,ids])-ref[k])/scale;delta=logs-data['logop'][starts+k,ids]
        basic=(((delta/cal['span']).square()+.25*err.square()+.25*((op-target[k])/cal['maxop']).square())*weights[k]).mean((1,2))/L
        if k==0:basic=basic+.25*(delta/cal['span']).square().mean((1,2))
        charge=charge+err/L;peak=torch.maximum(peak,f.t.current(torch,op,data['v'][starts+k,ids]).abs()/scale)
        values=(torch.asinh(delta).square()+err.square()).mean(1);grouped=torch.stack([values[:,js].mean(1) for js in p['families'].values()],1);value=.1*(grouped.mean(1)+grouped.amax(1))/L
        perchannel+=err.detach().square().mean(1)/L;reg=reg+err.square().mean(1);common+=basic.detach();fam+=value.detach();block=block+basic+extra*value
        h=net.step(f.t.features(torch,data['v'][starts+k,ids],data['ca'][starts+k,ids]),h)
        if (k+1)%width==0 or k+1==L:
            first=(k//width)*width;count=k+1-first;base=reference_mse[first:k+1].mean(0);relative=(reg/count-base)/base.clamp_min(p['preservation']['denominator_floor'])
            excess=relative.relu()
            worst=torch.tensor([a.get('worst_guard',False) for a in p['arms'] for _ in p['seeds']],device=ids.device)
            penalty=torch.where(worst,excess.mean(1)+excess.amax(1),excess.mean(1))*p['preservation']['weight']*count/L
            block=block+mask.to(penalty.dtype)*penalty;penalty_value+=penalty.detach();activation+=(relative.detach()>0).float()*count/L;reg=torch.zeros_like(reg)
        if backward and ((k+1)%p['bptt_ticks']==0 or k+1==L):
            if k+1==L:
                end=charge.square().mean((1,2))+(peak-ref.abs().amax(0)/scale).square().mean((1,2));block=block+.1*extra*end;fam+=.1*end.detach()
            block.sum().backward(retain_graph=k+1<L)
            # All arms have full512 credit; only detach at the same end boundary.
            if k+1==L:h=h.detach();charge=charge.detach();peak=peak.detach()
            block=torch.zeros_like(common)
    return common,fam,perchannel,dict(penalty=penalty_value,activation=activation)

def train(torch,p,net,data,aut,cal,out,ordinary_count):
    device=data['v'].device;E=len(net.members);fit=torch.tensor([i for i,m in enumerate(data['meta'][:ordinary_count]) if m['role']=='fit'],device=device);sup=torch.tensor([i for i,m in enumerate(data['meta']) if i>=ordinary_count and m['role']=='fit'],device=device);ai=torch.tensor([i for i,m in enumerate(aut['meta']) if m['role']=='fit'],device=device)
    resumed=(out/'resume_request.json').exists();start_update=0
    if resumed:
        baseline=json.loads((out/'validation_baseline.json').read_text());ch0=torch.tensor(baseline['channels'],device=device);tr0=torch.tensor(baseline['trajectory'],device=device);best=np.array(baseline['score']);chosen=np.zeros(E,int)
        selected=torch.load(out/'initial.pt',map_location=device,weights_only=True);curve=json.loads((out/'learning_curve.json').read_text());saved=torch.load(out/'resumable_training.pt',map_location=device,weights_only=True);start_update=int(saved['update'])
        assert start_update==json.loads((out/'resume_request.json').read_text())['checkpoint_update']
        for row in curve:
            assert row['update']<=start_update
            values=np.array(row['score']);take=np.array(row['eligible'])&(values<best)
            if take.any():
                state=torch.load(out/f'checkpoint_{row["update"]}.pt',map_location=device,weights_only=True)
                for key in selected:selected[key][take]=state[key][take]
            best=np.where(take,values,best);chosen[take]=row['update']
    else:
        score0,ch0,tr0=f.validate(torch,p,net,data,aut,cal);best=score0.cpu().numpy();chosen=np.zeros(E,int);selected=copy.deepcopy(net.state_dict());curve=[]
        torch.save(selected,out/'initial.pt');f.a.n.p.save_json(out/'validation_baseline.json',dict(score=best.tolist(),channels=ch0.cpu().tolist(),trajectory=tr0.cpu().tolist()))
    # The preservation reference always remains the ORIGINAL model, not the
    # resumed candidate. Otherwise resumption would silently move the anchor.
    reference=copy.deepcopy(net)
    for v in reference.parameters():v.requires_grad_(False)
    if resumed:net.load_state_dict(saved['model'])
    encoder={k:v.clone() for k,v in net.state_dict().items() if k in ('enc','eb')};params=[v for v in net.parameters() if v.requires_grad]
    opt=torch.optim.Adam(params,lr=p['learning_rate'],eps=p['adam_epsilon']);rng=np.random.default_rng(9526599);started=time.monotonic()
    mask=torch.tensor([a['project'] for a in p['arms'] for _ in p['seeds']],device=device)
    conflict_sum=torch.zeros(E,device=device,dtype=torch.float64);dot_sum=conflict_sum.clone();after_sum=conflict_sum.clone();guard_sum=conflict_sum.clone();step_uphill=conflict_sum.clone();step_current=conflict_sum.clone();step_trajectory=conflict_sum.clone();active_sum=torch.zeros(E,11,device=device,dtype=torch.float64)
    if resumed:
        opt.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng']
        keys=['conflict_fraction','mean_dot_before','mean_dot_after','mean_guard','actual_adam_current_uphill_fraction','actual_step_current_dot','actual_step_trajectory_dot','channel_penalty_activation']
        values=[conflict_sum,dot_sum,after_sum,guard_sum,step_uphill,step_current,step_trajectory,active_sum]
        if 'diagnostic_sums' in saved:
            for key,value in zip(keys,values):value.copy_(saved['diagnostic_sums'][key].to(device))
            diagnostic_source='exact_saved_sums'
        else:
            assert curve[-1]['update']==start_update
            for key,value in zip(keys,values):value.copy_(torch.tensor(curve[-1][key],device=device,dtype=value.dtype)*start_update)
            diagnostic_source='reconstructed_from_saved_means; roundoff_only; not_used_for_training_or_selection'
        f.a.n.p.save_json(out/'resume_state_verified.json',dict(valid=True,checkpoint_update=start_update,optimizer_restored=True,rng_restored=True,original_reference_preserved=True,historical_selection_rescored=False,diagnostic_source=diagnostic_source))
    for update in range(start_update+1,p['updates']+1):
        if time.monotonic()-started>p['runtime_budget_seconds']*.8:raise TimeoutError('52g training time limit; checkpoints retained')
        ii=fit[torch.tensor(rng.integers(len(fit),size=p['batch_size']),device=device)];ss=torch.tensor(rng.integers(len(data['v'])-p['chunk_ticks'],size=p['batch_size']),device=device);ids=ii[None].expand(E,-1).clone();starts=ss[None].expand(E,-1).clone();n=p['batch_size']//2;si=sup[torch.tensor(rng.integers(len(sup),size=n),device=device)]
        for i,a in enumerate(p['arms']):
            if a['support']:ids[len(p['seeds'])*i:len(p['seeds'])*(i+1),:n]=si
        if p.get('targeted_Ih'):
            slow=torch.tensor([i for i,m in enumerate(data['meta']) if m['role']=='fit' and m.get('protocol')=='slow_recovery'],device=device)
            assert len(slow)>0
            count=max(1,n//2);ti=slow[torch.tensor(rng.integers(len(slow),size=count),device=device)]
            anchors=np.array(p['targeted_Ih']['start_ms']);ts=torch.tensor(np.rint(anchors[rng.integers(len(anchors),size=count)]/p['dt_ms']),device=device,dtype=torch.long).clamp_max(len(data['v'])-p['chunk_ticks']-1)
            for i,a in enumerate(p['arms']):
                if a.get('targeted_Ih'):
                    block=slice(i*len(p['seeds']),(i+1)*len(p['seeds']));ids[block,:count]=ti;starts[block,:count]=ts
        rmse=reference_errors(torch,p,reference,data,ids,starts);opt.zero_grad(set_to_none=True);_,_,_,diagnostic=teacher(torch,p,net,data,ids,starts,cal,rmse)
        gc=[v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v) for v in params];opt.zero_grad(set_to_none=True)
        aid=ai[torch.tensor(rng.integers(len(ai),size=p['coupled_training']['batch_size']),device=device)];length=p['coupled_training']['horizon_before300' if update<=300 else 'horizon_after300'];ast=torch.tensor(rng.integers(len(aut['inj'])-length,size=len(aid)),device=device)
        f.trajectory(torch,p,net,aut,aid,ast,length,torch.ones(E,device=device),True);gv=[v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v) for v in params]
        grads,diagnosis=merge_gradients(torch,gc,gv,mask)
        for v,g in zip(params,grads):v.grad=g
        norm=sum(v.grad.double().reshape(E,-1).square().sum(-1) for v in params).sqrt();factor=(p['gradient_clip']/norm.clamp_min(1e-12)).clamp_max(1)
        for v in params:assert torch.isfinite(v.grad).all();v.grad.mul_(factor.to(v.dtype).reshape(E,*([1]*(v.ndim-1))))
        previous=[v.detach().clone() for v in params];opt.step();assert all(torch.isfinite(v).all() for v in net.parameters())
        actual_c=sum(((v.detach()-old).double()*gradient.double()).reshape(E,-1).sum(1) for v,old,gradient in zip(params,previous,gc));actual_v=sum(((v.detach()-old).double()*gradient.double()).reshape(E,-1).sum(1) for v,old,gradient in zip(params,previous,gv))
        step_uphill+=(actual_c>0);step_current+=actual_c;step_trajectory+=actual_v;conflict_sum+=diagnosis['conflict'];dot_sum+=diagnosis['dot'];after_sum+=diagnosis['after'];guard_sum+=diagnostic['penalty'];active_sum+=diagnostic['activation']
        if update%20==0:
            # Recovery-only snapshots: no extra validation or ranking point.
            sums=dict(zip(['conflict_fraction','mean_dot_before','mean_dot_after','mean_guard','actual_adam_current_uphill_fraction','actual_step_current_dot','actual_step_trajectory_dot','channel_penalty_activation'],[v.detach().clone() for v in (conflict_sum,dot_sum,after_sum,guard_sum,step_uphill,step_current,step_trajectory,active_sum)]))
            temporary=out/'recovery_latest.pt.tmp'
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),update=update,rng=rng.bit_generator.state,diagnostic_sums=sums,recovery_only=True),temporary);temporary.replace(out/'recovery_latest.pt')
            print(f'[GIADA52g] training {update}/{p["updates"]}',flush=True)
        if update in p['checkpoints']:
            score,ch,tr=f.validate(torch,p,net,data,aut,cal);eligible=((ch<=ch0*(1+1e-6)+1e-12).all(1)&(tr<=tr0*(1+1e-6)+1e-12).all(1));values=score.cpu().numpy();take=eligible.cpu().numpy()&(values<best)
            for k,v in net.state_dict().items():selected[k][take]=v[take].detach()
            best=np.where(take,values,best);chosen[take]=update;curve.append(dict(update=update,score=values.tolist(),eligible=eligible.cpu().tolist(),channel_errors=ch.cpu().tolist(),trajectory_error=tr.cpu().tolist(),gradient_norm=norm.cpu().tolist(),conflict_fraction=(conflict_sum/update).cpu().tolist(),mean_dot_before=(dot_sum/update).cpu().tolist(),mean_dot_after=(after_sum/update).cpu().tolist(),mean_guard=(guard_sum/update).cpu().tolist(),channel_penalty_activation=(active_sum/update).cpu().tolist(),actual_adam_current_uphill_fraction=(step_uphill/update).cpu().tolist(),actual_step_current_dot=(step_current/update).cpu().tolist(),actual_step_trajectory_dot=(step_trajectory/update).cpu().tolist()))
            diagnostic_sums=dict(zip(['conflict_fraction','mean_dot_before','mean_dot_after','mean_guard','actual_adam_current_uphill_fraction','actual_step_current_dot','actual_step_trajectory_dot','channel_penalty_activation'],[v.detach().clone() for v in (conflict_sum,dot_sum,after_sum,guard_sum,step_uphill,step_current,step_trajectory,active_sum)]))
            torch.save(net.state_dict(),out/f'checkpoint_{update}.pt');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),update=update,rng=rng.bit_generator.state,diagnostic_sums=diagnostic_sums),out/'resumable_training.pt');f.a.n.p.save_json(out/'learning_curve.json',curve);f.checkpoint(out,update)
    assert all(torch.equal(net.state_dict()[k],v) for k,v in encoder.items());net.load_state_dict(selected);torch.save(selected,out/'selected.pt');f.a.n.p.save_json(out/'selection_freeze.json',dict(members=net.members,updates=chosen.tolist(),score=best.tolist(),baseline_fallback=(chosen==0).tolist(),fresh_accessed=False,encoder_frozen_after_calibration=True))

def run(torch,root,source,cost_source,recovery,package,out,device='cuda',override=None,coupled_override=None):
    return f.run(torch,root,source,cost_source,recovery,package,out,device,override or plan(root),coupled_override,train_fn=train,task_name='52g')
