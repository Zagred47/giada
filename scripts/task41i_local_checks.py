"""CPU contract tests and technical preflight; no scientific training."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.giada_teacher import task41i_selective_repair as i
out=ROOT/'experiments/results/task41i_protocol_v1'
result=subprocess.run([sys.executable,'-m','unittest','tests.test_task41i_selective_repair','-v'],cwd=ROOT,capture_output=True,text=True)
(out/'unit_tests.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
print(result.stderr,flush=True);assert result.returncode==0
torch.set_num_threads(1);spec,*_=i.contract(ROOT)
report=i.preflight(torch,ROOT,out,'cpu',spec)
i.p.save_json(out/'local_verification.json',dict(valid=True,unit_tests_passed=True,CPU_preflight=report,protocol_sha256=report['protocol_sha256'],GPU_preflight_required=True))
print(json.dumps(report),flush=True)
