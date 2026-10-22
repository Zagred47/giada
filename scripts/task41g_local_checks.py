"""Actual CPU verification recorded before dispatch."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.giada_teacher import task41g_preservation as g
folder=ROOT/'experiments/results/task41g_protocol_v1'
test=subprocess.run([sys.executable,'-m','unittest','tests.test_task41g_preservation','tests.test_task41f_diagnostic_json','-v'],cwd=ROOT,capture_output=True,text=True)
(folder/'unit_tests.txt').write_text(test.stdout+'\n'+test.stderr,encoding='utf-8')
print(test.stderr,flush=True);assert test.returncode==0
torch.set_num_threads(1);spec,*_=g.contract(ROOT)
report=g.preflight(torch,ROOT,folder,'cpu',spec)
g.p.save_json(folder/'local_verification.json',dict(valid=True,unit_tests_passed=True,CPU_preflight=report,protocol_sha256=report['protocol_sha256'],GPU_preflight_required=True))
print(json.dumps(report),flush=True)
