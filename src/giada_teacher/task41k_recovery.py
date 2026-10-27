"""Execution-only recovery. Preregistered source files and predicates unchanged."""
import gc
import types
import weakref
from . import task41k_evaluation as ev


def managed_scanner_class(original):
    """Break Scanner.step -> self cycle without replacing its numerical code."""
    class ManagedScanner(original):
        def __init__(self, *args, **kwargs):
            gc.collect()
            super().__init__(*args, **kwargs)
            step = self.step
            def cell(value):
                return (lambda: value).__closure__[0]
            cells = tuple(cell(weakref.proxy(self)) if name == 'self' else old
                          for name, old in zip(step.__code__.co_freevars, step.__closure__))
            self.step = types.FunctionType(step.__code__, step.__globals__,
                                           step.__name__, step.__defaults__, cells)
    return ManagedScanner


def preflight(torch, models, spec, device, out):
    import numpy as np
    dev = ev.previous.d.development(spec['new_seeds']['development'])
    n = dev['voltage'].shape[1]
    scanner = ev.previous.c.j.i.old.Scanner
    rows = []
    for name, model in models.items():
        runner = scanner(torch, [model], n, 32, device)
        v, ca = dev['voltage'][:65], dev['calcium'][:65]
        initial = dev['initial'].astype('float32')
        actual = runner.run(v, ca, initial)[0]
        state = torch.as_tensor(initial, device=device)[None].expand(model.a.shape[0], -1, -1).clone()
        expected = []
        for tick in range(len(v)):
            vv = torch.as_tensor(v[tick], device=device)[None].expand(state.shape[0], -1)
            cc = torch.as_tensor(ca[tick], device=device)[None].expand_as(vv)
            state = model(vv, cc, state)
            expected.append(state.cpu().numpy().copy())
        error = float(np.max(np.abs(actual - np.asarray(expected))))
        assert error <= spec['computational_parity_atol'], (name, error)
        reference = weakref.ref(runner)
        del runner
        assert reference() is None, 'Scanner reference cycle retained'
        rows.append(dict(model=name,graph_eager_error=error,graph_released=True,batch=n,ticks=len(v)))
        print('[GIADA Task41k] recovery graph/lifetime preflight '+name+' passed', flush=True)
    ev.j.p.save_json(out/'scanner_recovery_preflight.json',dict(valid=True,rows=rows,
        identical_recurrence=True,thresholds_changed=False,training_updates_performed=0))


def install():
    module = ev.previous.c.j.i.old
    if getattr(module.Scanner, '_task41k_managed', False):
        return
    cls = managed_scanner_class(module.Scanner)
    cls._task41k_managed = True
    module.Scanner = cls
    original = ev.evaluate
    def evaluate(torch,root,out,controls,trained,spec,cfg,calcium,base,device):
        preflight(torch,{**controls,**trained},spec,device,out)
        return original(torch,root,out,controls,trained,spec,cfg,calcium,base,device)
    ev.evaluate = evaluate
