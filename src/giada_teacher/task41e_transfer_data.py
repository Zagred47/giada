"""Materialize FIT histories, paired windows and held-input tail labels.

Canonical functions supply labels before optimization. Students receive only
initial states and causal exogenous inputs; histories have new seeded inputs.
"""
import numpy as np
from . import task41e_transfer_model as m
p, t = m.p, m.t


def tensor_inputs(torch, raw, device, dtype=None):
    return {k: torch.tensor(raw[k], device=device, dtype=dtype or torch.float32)
            for k in ('v', 'ca', 'g', 'mult', 'inj', 'syn')}


def history_pool(torch, spec, cfg, calcium, base, device, small=False):
    rng = np.random.default_rng(spec['new_seeds']['feedback'])
    nh = 8 if small else spec['feedback_histories']
    steps = 400 if small else spec['history_ticks']
    snap = np.unique(np.geomspace(40, steps, 16).round().astype(int))
    initial_v = rng.uniform(-85, -30, nh)
    initial_ca = 10**rng.uniform(-4.8, -3.5, nh)
    initial_g, _ = p.coefficients(rng.uniform(-95, -25, nh), initial_ca, .1)
    mult = rng.uniform(.5, 2, (nh, 11))
    levels = rng.uniform(-.001, .018, (nh, 4))
    on = rng.integers(20, 80, nh)
    off = on + rng.integers(40, 180, nh)
    syn = rng.uniform(0, .00012, (nh, 4))
    dtype = torch.float64
    args = [torch.tensor(a[None], device=device, dtype=dtype) for a in
            (initial_v, initial_ca, initial_g, mult, levels[:,0], syn)]
    core = t.make_core(torch, cfg, calcium, base, 100*np.pi)
    runner = t.Stepper(torch, core, args, device=='cuda')
    snapshots = []
    with torch.no_grad():
        for tick in range(1, steps+1):
            # Random rectangular excitation, a smaller later pulse and a long
            # recovery; no development protocol or trajectory is copied.
            inj = np.where((tick >= on) & (tick < off), levels[:,1], levels[:,0])
            inj = np.where((tick >= 200) & (tick < 250), levels[:,2], inj)
            inj = np.where(tick > 400, levels[:,3]*.1, inj)
            vv, cc, gg, _ = runner.tick(torch.tensor(inj[None], device=device, dtype=dtype), args[5])
            if tick in snap:
                snapshots.append(dict(v=vv[0].cpu().numpy().copy(), ca=cc[0].cpu().numpy().copy(),
                                      g=gg[0].cpu().numpy().copy(), mult=mult.copy(),
                                      inj=inj.copy(), syn=syn.copy(), history=np.arange(nh),
                                      time_tick=np.full(nh,tick)))
    raw = {k: np.concatenate([a[k] for a in snapshots]) for k in snapshots[0]}
    assert all(np.isfinite(v).all() for v in raw.values())
    assert (raw['ca'] > 0).all() and (raw['g'] >= 0).all() and (raw['g'] <= 1).all()
    n = len(raw['v']); actions = np.random.default_rng(spec['new_seeds']['intervention']).permutation(np.arange(n) % 13)
    values = {k: np.repeat(v[:,None],2,axis=1) for k,v in raw.items() if k not in ('history','time_tick')}
    for k, action in enumerate(actions):
        if action < 11:
            values['mult'][k,1,action] = 0
        elif action == 11:
            values['inj'][k,1] *= 1.01
        else:
            values['syn'][k,1] = 0
    values.update(history=raw['history'],time_tick=raw['time_tick'],action=actions)
    return values


def reference_windows(torch, raw, spec, cfg, calcium, base, device):
    data = tensor_inputs(torch, raw, device, torch.float64)
    vv, cc, gg = (data[k] for k in ('v','ca','g'))
    core = t.make_core(torch, cfg, calcium, base, 100*np.pi)
    currents, states = [], []
    with torch.no_grad():
        for _ in range(max(a['ticks'] for a in m.arms())):
            vv, cc, gg, current = core(vv,cc,gg,data['mult'],data['inj'],data['syn'])
            currents.append(current.cpu().numpy())
            states.append(gg.cpu().numpy())
    current = np.stack(currents)
    delta = current[:,:,1] - current[:,:,0]
    scale = np.maximum(abs(delta).max(0), spec['quality']['current_scale_floor_ma_cm2'])
    # Slow relaxation is measured on held-input labels from the actual history
    # states; support stratification is FIT-only and not a model selection.
    u,z = p.coefficients(raw['v'][:,0],raw['ca'][:,0],.1)
    tail_change = abs((raw['g'][:,0]-u)*np.maximum(1-z,0)**400)
    slow = tail_change[:,list(m.GATES)].max(-1)
    small = abs(delta).max(0)[:,list(m.CHANNELS)].max(-1)
    return dict(current=current, delta=delta, scale=scale, small_response=small, slow_tail=slow)


def select_support(raw, labels, spec, count, selection=False):
    n = len(raw['v']); count = min(count,n)
    rng = np.random.default_rng(spec['new_seeds']['selection_support' if selection else 'support'])
    order = rng.permutation(n)
    broad = order[:count]
    small = labels['small_response']; tail = labels['slow_tail']
    # Equal-count strata with an explicit broad component. Stable sort and
    # seeded ranks prevent a zero-response tie from selecting arbitrary cases.
    small_rank = order[np.argsort(small[order],kind='stable')]
    tail_rank = order[np.argsort(-tail[order],kind='stable')]
    wanted = list(order[:count//2])
    for source in (small_rank,tail_rank):
        used=set(wanted)
        wanted.extend(int(i) for i in source if int(i) not in used)
        wanted = list(dict.fromkeys(wanted))[:(3*count//4 if source is small_rank else count)]
    assert len(wanted)==count
    return dict(broad=np.array(broad),stratified=np.array(wanted,dtype=np.int64))


def prepare_feedback(torch, spec, cfg, calcium, base, device, output=None, selection=False, small=False):
    effective = dict(spec)
    if selection:
        effective['new_seeds'] = {k:v+100000 for k,v in spec['new_seeds'].items()}
        effective['feedback_histories'] = spec['selection_feedback_histories']
    raw = history_pool(torch,effective,cfg,calcium,base,device,small)
    labels = reference_windows(torch,raw,spec,cfg,calcium,base,device)
    count = spec['selection_feedback_anchors' if selection else 'feedback_anchors']
    supports = select_support(raw,labels,effective,count,selection)
    data = tensor_inputs(torch,raw,device)
    data.update(target=torch.tensor(labels['delta'][...,list(m.CHANNELS)],device=device),
                scale=torch.tensor(labels['scale'][:,list(m.CHANNELS)],device=device))
    supports = {k:torch.tensor(v,device=device) for k,v in supports.items()}
    if output is not None:
        prefix = 'selection' if selection else 'fit'
        np.savez_compressed(output/(prefix+'_feedback.npz'),**raw,**labels,
                            **{'support_'+k:v.cpu().numpy() for k,v in supports.items()})
        p.save_json(output/(prefix+'_support.json'),dict(total_history_states=len(raw['v']),
                    supports={k:len(v) for k,v in supports.items()},
                    history_seed=effective['new_seeds']['feedback'],
                    gate_occupancy_min=float(raw['g'].min()),gate_occupancy_max=float(raw['g'].max()),
                    response_range=[float(labels['small_response'].min()),float(labels['small_response'].max())],
                    history_ticks=int(raw['time_tick'].max()),independent_protocols=int(raw['history'].max()+1)))
    return data,supports,raw,labels


def held_tail_data(torch, parent, spec, device, count=256):
    rng=np.random.default_rng(spec['new_seeds']['protection'])
    v=rng.uniform(-90,-25,count).astype('float32');ca=(10**rng.uniform(-5,-3.5,count)).astype('float32')
    g0,_=p.coefficients(rng.uniform(-95,20,count),ca,.1)
    u,z=p.coefficients(v.astype('float64'),ca.astype('float64'),.1)
    ticks=np.array(spec['protection_ticks'],dtype='float64')
    truth=g0[:,None]-np.expm1(ticks[None,:,None]*np.log1p(-np.minimum(z,1-1e-15))[:,None]) *(u[:,None]-g0[:,None])
    target=p.readout(v[:,None],truth,np.ones((count,1,11)),m.c.ionic.REVERSALS)[1][...,list(m.CHANNELS)]
    vv=torch.tensor(v,device=device)[None].expand(3,-1);cc=torch.tensor(ca,device=device)[None].expand(3,-1)
    with torch.no_grad():ru,rz=parent.raw(vv,cc)
    return dict(v=torch.tensor(v,device=device),features=m.d.feature(torch,torch.tensor(v,device=device),64),
                base_u=ru,base_z=rz,initial=torch.tensor(g0,device=device),ticks=torch.tensor(ticks,device=device),
                target=torch.tensor(target,device=device),
                scale=torch.tensor(np.maximum(abs(target).max(1),1e-7),device=device))
