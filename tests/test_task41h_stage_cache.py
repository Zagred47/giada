"""Regression checks for remaining-stage orchestration and clone defaults."""
import inspect,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))

def test_guard_transformations_match_original_sources():
    from src.giada_teacher import task41h_path_repair as h,task41h_evaluation as ev
    from task41h_stage_cache import guard_core,guard_cost,guard_compact_cost
    guard_core(h.g.fe.evaluate);guard_cost(ev.ee.e.benchmark);guard_compact_cost(ev.execution_cost)

def test_recentered_clone_preserves_defaults_and_runs():
    import torch
    from src.giada_teacher import task41h_path_repair as h,task41h_evaluation as ev
    clone=types.FunctionType(h.t.make_core.__code__,dict(h.t.__dict__,membrane=ev.stable_membrane),argdefs=h.t.make_core.__defaults__)
    assert inspect.signature(clone)==inspect.signature(h.t.make_core)
    spec,cfg,ca,base=h.contract(ROOT)
    core=clone(torch,cfg,ca,base,100.0)
    v=torch.full((3,2),-65.);c=torch.full_like(v,1e-4);g=torch.full((3,2,18),.5)
    result=core(v,c,g,torch.ones(3,2,11),torch.zeros_like(v),torch.zeros(3,2,4))
    assert all(bool(torch.isfinite(value).all()) for value in result)
