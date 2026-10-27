"""Frozen evaluation entry point; original preregistered worker is unchanged."""
import json
import subprocess
from pathlib import Path
import run_task41k_learning_alignment as worker
from src.giada_teacher.task41k_recovery import install

original = worker.i.run


def run(root,out,*args,**kwargs):
    assert kwargs.get('evaluation_only') and kwargs.get('resume')
    freeze=worker.i.j.verify_freeze(out)
    worker.i.p.save_json(out/'evaluation_recovery.json',dict(
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        original_revision=json.loads((out/'revision.json').read_text())['revision'],
        freeze_sha256=freeze['freeze_sha256'],checkpoint_sha256=freeze['checkpoint_sha256'],
        training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,
        fix='Release unreachable recurrent CUDA graph pools; identical Scanner.step bytecode'))
    install()
    return original(root,out,*args,**kwargs)


worker.i.run=run
if __name__=='__main__':worker.main()
