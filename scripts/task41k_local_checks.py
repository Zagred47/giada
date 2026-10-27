"""Bounded reproducible CPU contract and gradient verification."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41k_learning_alignment as k
folder=ROOT/'experiments/results/task41k_protocol_v1'
r=subprocess.run([sys.executable,'-m','unittest','tests.test_task41k_learning_alignment','tests.test_task41j_operator_reset','-v'],cwd=ROOT,capture_output=True,text=True)
(folder/'unit_tests.txt').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');print(r.stderr,flush=True);assert r.returncode==0
k.contract(ROOT)
k.p.save_json(folder/'local_verification.json',dict(valid=True,unit_tests_passed=True,protocol_sha256=k.p.digest(ROOT/k.CONFIG),GPU_preflight_required=True))
