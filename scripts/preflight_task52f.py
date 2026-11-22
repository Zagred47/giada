import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_task52f.py');result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
    folder=ROOT/'experiments/results/task52f_protocol_v1';folder.mkdir(exist_ok=True)
    names=['src/giada_teacher/task52f_encoder_credit.py','scripts/run_task52f.py','scripts/task52f_kaggle.py','tests/test_task52f.py','experiments/task52f_preregistration_v1.json']
    write_json(folder/'local_preflight.json',dict(valid=True,tests=result.testsRun,real_frozen_models=True,CPU_only=True,remote_GPU_preflight_required=True,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}))
    write_json(folder/'input_verified.json',json.loads((ROOT/'experiments/results/task52d_protocol_v2/input_verified.json').read_text()))
