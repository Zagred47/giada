import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_task52b.py');result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    folder=ROOT/'experiments/results/task52b_protocol_v1';folder.mkdir(exist_ok=True)
    names=['src/giada_teacher/task52b_decoder_memory.py','scripts/run_task52b.py','scripts/task52b_kaggle.py','tests/test_task52b.py','experiments/task52b_preregistration_v1.json']
    write_json(folder/'local_preflight.json',dict(valid=True,tests=result.testsRun,real_frozen_models=True,CPU_only=True,remote_GPU_preflight_required=True,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}))
    for n in ('package.json','input_verified.json','frozen_input_verified.json'):
        write_json(folder/n,json.loads((ROOT/'experiments/results/task52_protocol_v1'/n).read_text()))
