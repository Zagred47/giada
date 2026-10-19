"""Local scientific contract checks; CUDA claims require remote preflights."""
import hashlib,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39f_temporal_recovery as f
from tests.test_task39f_temporal_recovery import Task39fTests
from tests.test_task39e_tail_support_heads import Task39eTests

def main():
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(x) for x in (Task39fTests,Task39eTests)])
    result=unittest.TextTestRunner(verbosity=1).run(suite);assert result.wasSuccessful()
    spec=json.loads((ROOT/f.CONFIG).read_text());previous=json.loads((ROOT/f.e.CONFIG).read_text())
    assert spec['quality']==previous['quality'] and spec['selection']==previous['selection']
    assert f.p.digest(ROOT/f.p.CONFIG)==spec['quality_config_sha256']
    assert spec['checkpoints']==previous['checkpoints'] and len(f.arms())==8
    sample,rows=f.recovery_windows(ROOT/spec['dataset'],'fit',['slow_recovery'],32,256,99591)
    assert sample.shape==(32,256,71) and rows[0]['fit_only']
    folder=ROOT/'experiments/results/task39f_protocol_v1';folder.mkdir(exist_ok=True)
    report=dict(valid=True,tests=result.testsRun,cuda_tested=False,quality_thresholds_unchanged=True,selection_rule_unchanged=True,models=24,contrasts=12,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [f.CONFIG,'src/giada_teacher/task39f_temporal_recovery.py','src/giada_teacher/task39e_tail_support_heads.py','src/giada_teacher/task39d_current_timing.py','scripts/task39e_console_manager.py','scripts/task39e_local_supervisor.py']})
    f.p.save_json(folder/'local_validation.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
