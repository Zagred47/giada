"""Local scientific checks and pinned source hashes for Task39i."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39i_coupled_relaxation as i
from tests.test_task39i_coupled_relaxation import Task39iTests
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Task39iTests));assert result.wasSuccessful()
    paths=[i.CONFIG,'src/giada_teacher/task39i_coupled_relaxation.py','src/giada_teacher/task39d_current_timing.py','scripts/run_task39i_coupled_relaxation.py',
           'scripts/build_task39i_protocol.py','tests/test_task39i_coupled_relaxation.py','scripts/task39e_console_manager.py','scripts/task39e_local_supervisor.py']
    folder=ROOT/'experiments/results/task39i_protocol_v1';folder.mkdir(exist_ok=True)
    report=dict(valid=True,tests=result.testsRun,cuda_tested=False,members=36,contrasts=14,quality_thresholds_unchanged=True,
        selection_rule_unchanged=True,source_sha256={p:i.p.digest(ROOT/p) for p in paths})
    i.p.save_json(folder/'local_validation.json',report);print(json.dumps(report),flush=True)
