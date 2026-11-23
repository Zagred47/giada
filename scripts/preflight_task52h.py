import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tests.test_task52h','tests.test_task52g')]);result=unittest.TextTestRunner(verbosity=1).run(suite);assert result.wasSuccessful()
    files=['src/giada_teacher/task52g_gradient_preservation.py','src/giada_teacher/task52h_ih_preservation.py','scripts/run_task52h.py','scripts/task52h_kaggle.py','scripts/task52g_resume_kaggle.py','experiments/task52h_preregistration_v1.json','tests/test_task52h.py']
    F=ROOT/'experiments/results/task52h_protocol_v1';F.mkdir(exist_ok=True)
    (F/'local_preflight.json').write_text(json.dumps(dict(valid=True,tests=result.testsRun,original_control_parity=True,fit_probe_weights_restored=True,GPU_feasibility_probe_required=True,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}),indent=2))
