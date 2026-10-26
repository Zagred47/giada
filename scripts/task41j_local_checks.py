"""Repeatable CPU contract and gradient checks; no scientific training."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.giada_teacher import task41j_operator_reset as j
folder=ROOT/'experiments/results/task41j_protocol_v1'
result=subprocess.run([sys.executable,'-m','unittest','tests.test_task41j_operator_reset','-v'],cwd=ROOT,capture_output=True,text=True)
(folder/'unit_tests.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
print(result.stderr,flush=True);assert result.returncode==0
torch.set_num_threads(1);spec=j.contract(ROOT)[0]
report=j.preflight(torch,ROOT,folder,'cpu',spec)
j.p.save_json(folder/'local_verification.json',dict(valid=True,unit_tests_passed=True,CPU_preflight=report,
    protocol_sha256=report['protocol_sha256'],GPU_preflight_required=True))
print(json.dumps(report),flush=True)
