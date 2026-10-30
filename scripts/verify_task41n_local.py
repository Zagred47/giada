"""Record actual local unit-test execution, not an inferred readiness flag."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
result=subprocess.run([sys.executable,'-m','unittest','tests.test_task41n','-v'],cwd=ROOT,capture_output=True,text=True)
print(result.stdout+result.stderr);result.check_returncode()
folder=ROOT/'experiments/results/task41n_protocol_v1';folder.mkdir(parents=True,exist_ok=True)
cpu=json.loads((ROOT/'artifacts/task41n_cpu_preflight/gpu_preflight.json').read_text())
protocol=hashlib.sha256((ROOT/'experiments/task41n_preregistration_v1.json').read_bytes()).hexdigest();assert cpu['protocol_sha256']==protocol
(folder/'local_verification.json').write_text(json.dumps(dict(valid=True,tests_exit_code=result.returncode,tests_output=result.stdout+result.stderr,cpu_parity=cpu,protocol_sha256=protocol,actual_GPU_preflight_pending=True,fresh_confirmation_accessed=False),indent=2),encoding='utf-8',newline='\n')
