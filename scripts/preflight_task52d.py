import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_task52d.py');result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
    folder=ROOT/'experiments/results/task52d_protocol_v2';folder.mkdir(exist_ok=True)
    names=['src/giada_teacher/task52d_balance_trajectory.py','scripts/run_task52d.py','scripts/task52d_kaggle.py','scripts/task52d_input.py','tests/test_task52d.py','experiments/task52d_preregistration_v2.json']
    write_json(folder/'local_preflight.json',dict(valid=True,tests=result.testsRun,real_frozen_models=True,CPU_only=True,remote_GPU_preflight_required=True,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}))
    write_json(folder/'input_verified.json',json.loads((ROOT/'experiments/results/task52c_protocol_v1/input_verified.json').read_text()))
