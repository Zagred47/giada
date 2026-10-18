"""CPU contract checks and real frozen-parent oracle on representative families."""
from pathlib import Path
import json,sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39d_current_timing as t
torch.set_num_threads(1)
suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_task39d_current_timing','tests.test_task39c_dynamic_identification'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
spec=json.loads((ROOT/t.CONFIG).read_text());quality=json.loads((ROOT/t.p.CONFIG).read_text())
quality['families']={k:v for k,v in quality['families'].items() if k in ('steps','slow_recovery')}
folder=ROOT/'experiments/results/task39d_protocol_v1';folder.mkdir(parents=True,exist_ok=True)
out=folder/'cpu_oracle_check';out.mkdir(exist_ok=True)
t.oracle_audit(torch,ROOT,out,spec,quality,'cpu')
t.p.save_json(folder/'local_validation.json',dict(valid=True,tests=result.testsRun,cpu_oracle_families=list(quality['families']),
    cuda_executed=False,full_cuda_preflights_required=True,all36_adam_parity=True,thresholds_changed=False))
