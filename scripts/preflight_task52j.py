import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    r=unittest.TextTestRunner().run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tests.test_task52g','tests.test_task52i','tests.test_task52j')]));assert r.wasSuccessful()
    files=['src/giada_teacher/task52g_gradient_preservation.py','scripts/run_task52j.py','scripts/task52j_kaggle.py','experiments/task52j_preregistration_v1.json','tests/test_task52j.py'];F=ROOT/'experiments/results/task52j_protocol_v1';F.mkdir(exist_ok=True);(F/'local_preflight.json').write_text(json.dumps(dict(valid=True,tests=r.testsRun,guard13_metrics=True,thresholds_changed=False,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}),indent=2))
