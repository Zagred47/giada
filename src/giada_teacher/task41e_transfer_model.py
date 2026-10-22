"""Matched local corrections and objective geometry for Task 41e.

P110 is frozen. Only two 64-weight equilibrium corrections are trained;
all 18 recurrent states and the physical current/membrane interface persist.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from . import task41d_precision_response as d

c, p, t, x = d.c, d.p, d.t, d.x
CONFIG = 'experiments/task41e_preregistration_v1.json'
GATES = (0, 8)
CHANNELS = (0, 4)


def arms():
    def arm(name, objective='censored', balance='dynamic', ticks=4,
            support='broad', joint=False, protection=False, feedback=True):
        return dict(name=name, objective=objective, balance=balance, ticks=ticks,
                    support=support, joint=joint, protection=protection,
                    feedback=feedback, basis=64, shape='linear')
    return [
        arm('A00', 'clipped', 'initial'), arm('A01', 'clipped'),
        arm('A10', balance='initial'), arm('A11'),
        arm('Q0', 'clipped', 'initial', feedback=False),
        arm('Q1', balance='initial', feedback=False),
        arm('H10', ticks=40), arm('H01', support='stratified'),
        arm('H11', ticks=40, support='stratified'),
        arm('J01', ticks=40, support='stratified', protection=True),
        arm('J10', ticks=40, support='stratified', joint=True),
        arm('J11', ticks=40, support='stratified', joint=True, protection=True),
    ]


def contrasts():
    return [
        ('objective', 'A00', 'A10'), ('objective', 'A01', 'A11'),
        ('balance', 'A00', 'A01'), ('balance', 'A10', 'A11'),
        ('feedback', 'Q0', 'A00'), ('feedback', 'Q1', 'A10'),
        ('objective_point_only', 'Q0', 'Q1'),
        ('horizon', 'A11', 'H10'), ('horizon', 'H01', 'H11'),
        ('support', 'A11', 'H01'), ('support', 'H10', 'H11'),
        ('joint', 'H11', 'J10'), ('joint', 'J01', 'J11'),
        ('protection', 'H11', 'J01'), ('protection', 'J10', 'J11'),
    ]


def contract(root):
    spec = json.loads((root / CONFIG).read_text())
    old, cfg, calcium, base = d.contract(root)
    assert spec['quality'] == old['quality']
    assert spec['compartment_gates'] == old['compartment_gates']
    assert spec['cost'] == old['cost']
    assert spec['arms'] == arms() and spec['contrasts'] == [list(a) for a in contrasts()]
    assert spec['seeds'] == [17, 29, 43] and spec['dt_ms'] == .1
    assert len(set(spec['new_seeds'].values())) == len(spec['new_seeds'])
    assert not set(spec['new_seeds'].values()) & set(old['new_seeds'].values())
    for name, expected in spec['sources'].items():
        assert p.digest(root / name) == expected, name
    return spec, cfg, calcium, base


def repeat_parent(torch, parent, definitions, device):
    result = c.model_class(torch)(dict(basis=64), [17, 29, 43] * len(definitions)).to(device=device,dtype=next(parent.parameters()).dtype)
    state = {name: (value.repeat(len(definitions), *([1] * (value.ndim - 1)))
                   if value.ndim and value.shape[0] == 3 and name != 'gbar' else value)
             for name, value in parent.state_dict().items()}
    result.load_state_dict(state)
    for value in result.parameters():
        value.requires_grad_(False)
    return result


def correction(torch, v, weights):
    features = d.feature(torch, v, 64, 'linear')
    # The reduction/order matches each old one-head correction. Accumulate
    # in FP64, then use the inference input dtype for tanh and logit addition.
    return torch.stack([
        .1 * torch.tanh((features.double() * weights[:, k, None].double()).sum(-1).to(v.dtype) / .1)
        for k in range(2)
    ], -1)


def add_correction(torch, u, extra):
    return torch.cat((u[..., :1] + extra[..., :1], u[..., 1:8],
                      u[..., 8:9] + extra[..., 1:2], u[..., 9:]), -1)


def model_class(torch):
    class Model(c.model_class(torch)):
        def __init__(self, definitions, seeds):
            super().__init__(dict(basis=64), seeds * len(definitions))
            for value in self.parameters():
                value.requires_grad_(False)
            self.definitions = definitions
            self.transfer_correction = torch.nn.Parameter(torch.zeros(len(seeds) * len(definitions), 2, 64))

        def raw(self, v, ca):
            u, z = super().raw(v, ca)
            return add_correction(torch, u, correction(torch, v, self.transfer_correction)), z
    return Model


def make_model(torch, parent, definitions, device):
    result = model_class(torch)(definitions, [17, 29, 43]).to(device=device,dtype=next(parent.parameters()).dtype)
    state = result.state_dict()
    for name, value in repeat_parent(torch, parent, definitions, device).state_dict().items():
        state[name] = value
    result.load_state_dict(state)
    return result


def extract(torch, model, index, root, device, weights=None):
    parent = d.prior.load_models(torch, root, device)['P110']
    result = make_model(torch, parent, [model.definitions[index]], device)
    with torch.no_grad():
        result.transfer_correction.copy_((model.transfer_correction if weights is None else weights)[index*3:index*3+3])
    result.eval()
    return result


class MemberView:
    """Frozen coefficient trunk with gradients flowing to selected shared heads."""
    def __init__(self, torch, parent, model, definitions, indices, device, scenarios=False):
        self.torch, self.model, self.indices = torch, model, indices
        self.parent = repeat_parent(torch, parent, definitions, device)
        self.scenarios = scenarios
        self.joint = torch.tensor([a['joint'] for a in definitions for _ in range(3)], device=device)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def raw(self, v, ca):
        u, z = self.parent.raw(v, ca)
        extra = correction(self.torch, v, self.model.transfer_correction[self.indices])
        if self.scenarios:
            # Samples are [batch, head-scenario, base/intervention]. Separate
            # training evolves each head with the other correction disabled;
            # joint training evolves both, with the same inference map.
            mask = self.torch.eye(2, device=v.device, dtype=v.dtype)[:, None].expand(2, 2, 2)
            mask = mask.reshape(4, 2).repeat(v.shape[1] // 4, 1)
            mask = self.torch.where(self.joint[:, None, None], self.torch.ones_like(mask)[None], mask[None])
            extra = extra * mask
        return add_correction(self.torch, u, extra), z

    def __call__(self, v, ca, g):
        u, z = self.raw(v, ca)
        return g + self.torch.sigmoid(z) * (self.torch.sigmoid(u) - g)


def clipped_or_censored_error(torch, raw, target, censored):
    lo, hi = 1e-7, 1-1e-7
    censored = torch.as_tensor(censored,device=raw.device,dtype=torch.bool)
    residual = raw.double() - torch.logit(target.double().clamp(lo, hi))
    interval = torch.where(target <= lo, residual.clamp_min(0),
                           torch.where(target >= hi, residual.clamp_max(0), residual))
    return torch.where(censored, interval, residual)


def atomic_data(torch, parent, spec, device, selection=False, count=None):
    count = count or spec['selection_anchors' if selection else 'anchors']
    data = d.anchors(spec['new_seeds']['selection' if selection else 'fit'], count)
    v, ca = (torch.tensor(data[k], device=device) for k in ('v', 'ca'))
    with torch.no_grad():
        base, _ = parent.raw(v.reshape(1, -1).expand(3, -1), ca.reshape(1, -1).expand(3, -1))
    n = 3 * len(arms())
    return dict(features=d.feature(torch, v, 64)[None].expand(n, -1, -1, -1),
                base=base[..., list(GATES)].reshape(3, count, 3, 2).repeat(len(arms()), 1, 1, 1),
                target=torch.tensor(data['u'][..., list(GATES)], device=device)[None].expand(n, -1, -1, -1),
                eps=torch.tensor(data['eps'], device=device)[None].expand(n, -1)), data


def atomic_terms(torch, weights, data, idx, censored):
    member = torch.arange(len(weights), device=weights.device)[:, None]
    features, base = data['features'][member, idx], data['base'][member, idx]
    extra = torch.stack([(features.double()*weights[:, k, None, None].double()).sum(-1).float()
                         for k in range(2)], -1)
    extra = .1 * torch.tanh(extra/.1)
    raw = base + extra
    target = data['target'][member, idx]
    residual = clipped_or_censored_error(torch, raw, target, censored[:, None, None, None])
    point = residual.square().mean((1, 2))
    occupancy = torch.sigmoid(raw).double()
    scale = (2*data['eps'][member, idx]).clamp_min(1e-7)[..., None]
    secant = (((occupancy[:, :, 2]-occupancy[:, :, 0])-(target[:, :, 2]-target[:, :, 0]))/scale).square().mean(1)
    return point, secant


def balance_weights(torch, point_norm, auxiliary_norm, previous, spec, dynamic, ema=None):
    cfg = spec['balance_policy']
    if ema is not None:
        ema.mul_(cfg['ema']).add_(torch.stack((point_norm, auxiliary_norm), -1), alpha=1-cfg['ema'])
        point_norm, auxiliary_norm = ema.unbind(-1)
    valid = (point_norm > cfg['gradient_floor']) & (auxiliary_norm > cfg['gradient_floor'])
    desired = (cfg['target_gradient_ratio'] * point_norm / auxiliary_norm.clamp_min(cfg['gradient_floor'])).clamp(0, cfg['maximum_weight'])
    desired = torch.where(valid, desired, torch.zeros_like(desired))
    if previous is None:
        return desired
    lower, upper = previous * cfg['minimum_weight_ratio'], previous * cfg['maximum_weight_ratio']
    limited = torch.where(previous > 0, desired.clamp(min=lower, max=upper), desired)
    limited = torch.where(valid, limited, torch.zeros_like(limited))
    return torch.where(dynamic[:, None], limited, previous)


def frozen_controls(torch, root, spec, device):
    parents = d.prior.load_models(torch, root, device)
    path = root / spec['task41d_checkpoint']
    assert p.digest(path) == spec['task41d_checkpoint_sha256']
    saved = torch.load(path, map_location=device, weights_only=True)
    ca = d.make_model(torch, parents['P110'], [next(a for a in d.arms() if a['name']=='C01')], [17,29,43], device)
    nap = d.make_model(torch, parents['P110'], [next(a for a in d.arms() if a['name']=='N000')], [17,29,43], device)
    ca.load_state_dict(saved['C01']); nap.load_state_dict(saved['N000'])
    from .task41d_evaluation import compose
    combined = compose(torch, ca, nap, parents['P110'])
    fused = make_model(torch, parents['P110'], [arms()[0]], device)
    with torch.no_grad():
        fused.transfer_correction[:,0].copy_(ca.local_equilibrium)
        fused.transfer_correction[:,1].copy_(nap.local_equilibrium)
    return dict(**parents, D_C01=ca, D_N000=nap, D_combined=combined, D_fused=fused)
