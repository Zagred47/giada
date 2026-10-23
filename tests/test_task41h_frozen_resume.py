"""Recovery orchestration must not change scientific code or call training."""
import ast
import json
import tempfile
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_no_training_or_selection_calls():
    tree=ast.parse((ROOT/'scripts/resume_task41h_frozen_evaluation.py').read_text())
    forbidden={'backward','step','atomic_data','feedback_terms','choose','train'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
            assert name not in forbidden,name

def test_freeze_and_backend_remain_checked():
    source=(ROOT/'scripts/resume_task41h_frozen_evaluation.py').read_text()
    for required in ('verify_freeze(output)',"torch.__version__==seal['torch_version']", "torch.cuda.get_device_name()==seal['GPU']", "selection_recomputed=False", "original_suite(torch,folder,models,*args)"):
        assert required in source

def test_completed_cache_and_uncached_oracle_routing():
    tree=ast.parse((ROOT/'scripts/resume_task41h_frozen_evaluation.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='install_completed_panel_cache')
    calls=[]
    def suite(*args):calls.append('suite');return 'computed'
    def stress(*args):calls.append('stress');return 'oracle-computed'
    old=types.SimpleNamespace(suite=suite,stress=stress)
    namespace=dict(ev=types.SimpleNamespace(ee=types.SimpleNamespace(old=old)),json=json)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'cache_test','exec'),namespace)
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
        output=Path(tmp);folder=output/'opened41';folder.mkdir()
        cached={f'A/{mode}':{'passed':True} for mode in ('imposed','autonomous')}
        (folder/'causal_results.json').write_text(json.dumps(cached))
        for mode in ('imposed','autonomous'):
            (folder/f'floor_{mode}.json').write_text(json.dumps(dict(cases=[dict(passed=True)],interventions=[])))
        (folder/'native_phase_floor.json').write_text(json.dumps([dict(passed=True)]))
        (output/'stress41.json').write_text(json.dumps(dict(rows=[dict(model='A')],formula_floor=[dict(passed=True)])))
        reused=namespace['install_completed_panel_cache'](output)
        assert old.suite(None,folder,{'A':None})==cached
        assert old.stress(None,{'A':None})['rows']==[dict(model='A')]
        assert calls==[] and len(reused)==2
        assert old.stress(None,{'diagnostic_oracle':None})=='oracle-computed'
        assert calls==['stress']
