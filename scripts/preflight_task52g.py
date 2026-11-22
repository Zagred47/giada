import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p) for p in ('test_task52g.py','test_task52f.py')]);result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
    folder=ROOT/'experiments/results/task52g_protocol_v1';folder.mkdir(exist_ok=True)
    files=['src/giada_teacher/task52g_gradient_preservation.py','src/giada_teacher/task52f_encoder_credit.py','scripts/run_task52g.py','scripts/task52g_console_manager.py','tests/test_task52g.py','experiments/task52g_preregistration_v1.json']
    (folder/'local_preflight.json').write_text(json.dumps(dict(valid=True,tests=result.testsRun,real_frozen_models=True,control_gradient_parity=True,projection_mask_test=True,full_small_pipeline=True,files_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}),indent=2))
