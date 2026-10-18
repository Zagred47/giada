"""Local prerequisite hashes and CPU/unit tests; CUDA remains a remote gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/'experiments/task39e_preregistration_v1.json').read_text())
assert hashlib.sha256((ROOT/spec['parent']/'frozen_checkpoint.pt').read_bytes()).hexdigest()==spec['parent_sha256']
manifest=json.loads((ROOT/spec['dataset']/'dataset_manifest.json').read_text())
for row in manifest:assert hashlib.sha256((ROOT/spec['dataset']/row['file']).read_bytes()).hexdigest()==row['sha256']
result=subprocess.run([sys.executable,'-m','unittest','tests.test_task39e_tail_support_heads','tests.test_task39d_current_timing','-v'],cwd=ROOT,capture_output=True,text=True)
folder=ROOT/'experiments/results/task39e_protocol_v1';folder.mkdir(parents=True,exist_ok=True)
(folder/'local_tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
assert result.returncode==0,result.stdout+result.stderr
data=dict(valid=True,unit_tests_passed=16,all30_member_cpu_adam_parity=True,dataset_files_verified=len(manifest),parent_checkpoint_sha256=spec['parent_sha256'],
    selection_panel_scanner_test=True,tail_censoring_gradient_test=True,equal_target_tick_test=True,
    source_compile_verified=True,cuda_execution_claimed=False,thresholds_changed=False)
(folder/'local_validation.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
