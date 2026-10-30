"""Ensure native phase imports/contracts do not require or initialize Torch."""
import builtins,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name=='torch' or name.startswith('torch.'):
        raise ImportError('Native environment intentionally excludes Torch')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from src.giada_teacher import task41n_independent_confirmation as n
n.contract(ROOT)
print('Native imports and contract verified without Torch',flush=True)
