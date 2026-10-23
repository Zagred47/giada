"""Recovery orchestration must not change scientific code or call training."""
import ast
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
