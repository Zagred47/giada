"""Fit-only feasibility probe and paired Ih support x worst-current guard."""
import copy,json
from . import task52g_gradient_preservation as g
f=g.f
class FeasibilityBlocked(RuntimeError):
    """Expected bounded diagnostic stop, not an infrastructure failure."""
def plan(root):
    p=json.loads((root/'experiments/task52h_preregistration_v1.json').read_text());assert p['task']=='52h' and not p['thresholds_changed'] and not p['abstraction_changed'];return p
def probe(torch,p,net,data,cal,out):
    cfg=p['feasibility_probe'];pp=copy.deepcopy(p);pp['chunk_ticks']=cfg['chunk_ticks']
    E=len(net.members);device=data['v'].device
    ii=[i for i,m in enumerate(data['meta']) if m['role']=='fit' and m.get('protocol')=='slow_recovery'][:cfg['batch_size']];assert len(ii)==cfg['batch_size']
    ids=torch.tensor(ii,device=device)[None].expand(E,-1);starts=torch.full_like(ids,min(round(350/p['dt_ms']),len(data['v'])-cfg['chunk_ticks']-1))
    def errors():
        h=net.initialize(data['g'][starts,ids]);values=[];ref=[]
        for k in range(cfg['chunk_ticks']):
            v=data['v'][starts+k,ids];values.append(f.t.current(torch,net.opening(h),v));ref.append(f.t.current(torch,data['op'][starts+k,ids],v));h=net.step(f.t.features(torch,v,data['ca'][starts+k,ids]),h)
        target=torch.stack(ref);return ((torch.stack(values)-target)/target.abs().amax(0).clamp_min(1e-7)).square().mean((0,2))
    initial=copy.deepcopy(net.state_dict());params=[v for v in net.parameters() if v.requires_grad];base=errors();base[:,5].sum().backward();grads=[v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v) for v in params]
    norm=sum(v.double().reshape(E,-1).square().sum(1) for v in grads).sqrt().clamp_min(1e-30);trials=[];success=torch.zeros(E,device=device,dtype=torch.bool)
    for step in cfg['steps']:
        net.load_state_dict(initial)
        with torch.no_grad():
            for v,grad in zip(params,grads):v.add_(-step*grad/norm.to(v.dtype).reshape(E,*([1]*(v.ndim-1))))
            after=errors();safe=(after<=base.detach()*(1+cfg['relative_tolerance'])+cfg['absolute_tolerance']).all(1);improved=after[:,5]<base.detach()[:,5]*(1-cfg['relative_tolerance']);good=safe&improved;success|=good
            trials.append(dict(step=step,passed=good.cpu().tolist(),errors=after.cpu().tolist()))
    net.load_state_dict(initial);net.zero_grad(set_to_none=True)
    # Arms are paired copies; require a successful bounded fit-only direction for each seed.
    valid=bool(success.reshape(len(p['arms']),len(p['seeds'])).all())
    report=dict(valid=valid,fit_only=True,weights_restored=all(torch.equal(v,initial[k]) for k,v in net.state_dict().items()),baseline=base.detach().cpu().tolist(),trials=trials,qualification='Local fit probe only; failure is inconclusive about global capacity; no development/fresh access.')
    f.a.n.p.save_json(out/'Ih_feasibility_preflight.json',report);assert report['weights_restored'];return valid
def train(torch,p,net,data,aut,cal,out,ordinary_count):
    if not probe(torch,p,net,data,cal,out):
        f.a.n.p.save_json(out/'preflight_blocked.json',dict(valid=True,full_training_started=False,task='52h',reason='No non-regressing Ih step found for every seed in bounded fit-only probe',thresholds_changed=False));raise FeasibilityBlocked('52h fit-only feasibility preflight blocked full training; inspect report')
    g.train(torch,p,net,data,aut,cal,out,ordinary_count)
def run(torch,root,source,cost,recovery,package,out,device='cuda',override=None,coupled_override=None):
    return f.run(torch,root,source,cost,recovery,package,out,device,override or plan(root),coupled_override,train_fn=train,task_name='52h')
