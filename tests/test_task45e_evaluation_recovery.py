"""Run the same full recovery fixture against45e bindings and new cost closure."""
import inspect,sys,textwrap,unittest
from pathlib import Path
from tests import test_task45d_evaluation_recovery as prior
from src.giada_teacher import task45e_evaluation_recovery as r
ROOT=prior.ROOT
class Evaluation45e(unittest.TestCase):
    def test_no_training_selection_and_notebook(self):
        self.assertEqual(r.engine.ARMS,tuple(r.e.spec(ROOT)[0]['coupled_arms']))
        self.assertFalse(r.calls&{'fit','choose','nominate','backward','step','train_loss'})
        sys.path.insert(0,str(ROOT/'scripts'));import task45e_evaluation_kaggle as k
        text=''.join(k.notebook('a'*40,'b'*64)['cells'][1]['source']);compile(text,'45e-eval-notebook','exec')
        self.assertIn('run_task45e_evaluation_recovery.py',text);self.assertIn('task45e_final.zip',text)
        self.assertIn('str(saved[0]),str(checkpoints[0])',text);self.assertIn('selection_recomputed=False',text);self.assertNotIn('selection_recomputed=(out/',text)
        self.assertFalse(k.plan()['new_upload']);self.assertEqual(k.plan()['transport'],'existing_private_kernel_output')
        self.assertNotIn('dataset_create_new',Path(k.__file__).read_text())
source=textwrap.dedent(inspect.getsource(prior.EvaluationRecovery.test_skip_completed_block_and_load_saved_freeze)).replace('GPU_training_transition_preflight.json','GPU_cost_preflight.json')
scope=dict(prior.__dict__,r=r)
exec(compile(source,'45e_bound_original_full_recovery_fixture','exec'),scope)
Evaluation45e.test_skip_completed_block_and_load_saved_freeze=scope['test_skip_completed_block_and_load_saved_freeze']
if __name__=='__main__':unittest.main()
