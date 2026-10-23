import ast,importlib.util,unittest
from pathlib import Path

class RecoveryTests(unittest.TestCase):
    def test_earliest_lexicographic_selection(self):
        # Compile only the pure reducer; no torch/model import or training.
        source=(Path(__file__).resolve().parents[1]/'scripts/recover_task41h_evaluation.py').read_text()
        tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose')
        import numpy as np
        scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'choose','exec'),scope)
        curves=[dict(step=s,ranks={r:[[1,0,v]] for r in ('aggregate','stratified')}) for s,v in ((0,3.),(512,2.),(60000,2.))]
        best,steps=scope['choose'](curves,1)
        self.assertEqual(steps,dict(aggregate=[512],stratified=[512]))
        self.assertEqual(best['stratified'],[(1,0,2.)])
    def test_no_training_calls(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/recover_task41h_evaluation.py').read_text())
        banned={'backward','step','zero_grad','capture','auxiliary_step','point_step','train_and_select','train'}
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls&banned,calls&banned)

if __name__=='__main__':unittest.main()
