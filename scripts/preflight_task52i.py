import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tests.test_task52g','tests.test_task52i')]));assert result.wasSuccessful()
    files=['src/giada_teacher/task52g_gradient_preservation.py','src/giada_teacher/task52i_solution_retention.py','scripts/run_task52i.py','scripts/task52i_kaggle.py','experiments/task52i_preregistration_v1.json','tests/test_task52i.py'];F=ROOT/'experiments/results/task52i_protocol_v1';F.mkdir(exist_ok=True)
    (F/'local_preflight.json').write_text(json.dumps(dict(valid=True,tests=result.testsRun,guard_panel_excluded=True,guarded_step_nonregression_test=True,original_control_parity=True,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}),indent=2))
