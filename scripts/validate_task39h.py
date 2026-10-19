"""Local39h scientific checks. CUDA behavior is verified on the actual runtime."""
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39h_relaxation_identification as g
from tests.test_task39h_relaxation_identification import Task39hTests

def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Task39hTests));assert result.wasSuccessful()
    spec=json.loads((ROOT/g.CONFIG).read_text());previous=json.loads((ROOT/g.g.CONFIG).read_text())
    assert spec['quality']==previous['quality'] and spec['selection']==previous['selection']
    assert spec['data_seed']==previous['data_seed'] and g.p.digest(ROOT/g.p.CONFIG)==spec['quality_config_sha256']
    assert spec['checkpoints'][-1]==120000 and len(g.arms())==10 and len(g.contrast_pairs())==14
    folder=ROOT/'experiments/results/task39h_protocol_v1';folder.mkdir(exist_ok=True)
    sources=[g.CONFIG,'src/giada_teacher/task39h_relaxation_identification.py','src/giada_teacher/task39f_temporal_recovery.py','src/giada_teacher/task39e_tail_support_heads.py','src/giada_teacher/task39d_current_timing.py','src/giada_teacher/task37_ionic_playground.py','scripts/run_task39h_relaxation_identification.py','scripts/task39e_console_manager.py','scripts/task39e_local_supervisor.py','tests/test_task39h_relaxation_identification.py']
    report=dict(valid=True,tests=result.testsRun,cuda_tested=False,quality_thresholds_unchanged=True,selection_rule_unchanged=True,models=30,contrasts=14,source_sha256={p:g.p.digest(ROOT/p) for p in sources})
    gpu_path=folder/'gpu_preflight.json'
    if gpu_path.exists():
        gpu=json.loads(gpu_path.read_text());assert gpu['valid'] and gpu['capture']['valid']
        assert gpu['source_sha256']==g.p.digest(ROOT/'src/giada_teacher/task39h_relaxation_identification.py')
        assert gpu['config_sha256']==g.p.digest(ROOT/g.CONFIG)
        report.update(cuda_tested=True,gpu_preflight_sha256=g.p.digest(gpu_path),gpu=gpu['gpu'])
    g.p.save_json(folder/'local_validation.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
