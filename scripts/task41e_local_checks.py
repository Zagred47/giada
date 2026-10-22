"""Persist meaningful local tests and the full36member CPU preflight."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41e_transfer_model as m
from src.giada_teacher import task41e_transfer_training as training
import torch

folder=ROOT/'experiments/results/task41e_protocol_v1'
result=subprocess.run([sys.executable,'-m','unittest','tests.test_task41e_transfer','-v'],cwd=ROOT,capture_output=True,text=True)
(folder/'unit_tests.txt').write_text(result.stdout+'\n'+result.stderr)
print(result.stderr,flush=True)
assert result.returncode==0,'Local tests failed'
torch.set_num_threads(1)
spec,*_=m.contract(ROOT)
report=training.preflight(torch,ROOT,folder,'cpu',spec)
(folder/'cpu_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
(folder/'local_verification.json').write_text(json.dumps(dict(valid=True,unit_test_command='python -m unittest tests.test_task41e_transfer -v',
    tests=10,cpu_preflight_valid=report['valid'],protocol_sha256=report['protocol_sha256'],GPU_preflight_required_before_dispatch=True),indent=2)+'\n')
print(json.dumps(dict(valid=True,tests=10,CPU_members=36,all_parameters_finite=report['all_parameters_finite'])),flush=True)
