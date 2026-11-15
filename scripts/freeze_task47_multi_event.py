"""Lock the evaluator and negative controls before evaluating the full cohort."""
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task47_multi_event_v1'
if __name__=='__main__':
    F.mkdir(exist_ok=True);assert not (F/'freeze.json').exists() and not (F/'summary.json').exists()
    r=subprocess.run([sys.executable,'-m','unittest','tests.test_multi_event_evaluation','-v'],cwd=ROOT,capture_output=True,text=True);print(r.stderr,flush=True);assert r.returncode==0
    names=['experiments/task47_multi_event_evaluation_v1.json','src/giada_teacher/multi_event_evaluation.py','tests/test_multi_event_evaluation.py','scripts/run_task47_multi_event.py','src/giada_teacher/task37_ionic_playground.py','experiments/task47_preregistration_v1.json']
    plan=json.loads((ROOT/names[0]).read_text());quality=json.loads((ROOT/'experiments/task47_preregistration_v1.json').read_text())['quality'];c=plan['numeric_constants']
    assert c['peak_timing_limit_ms']==quality['peak_timing_max_ms'] and c['onset_offset_limit_ms']==quality['onset_offset_max_ms'] and c['peak_plateau_relative_band']==quality['peak_plateau_relative_band']
    write_json(F/'freeze.json',dict(valid=True,created_utc=datetime.now(timezone.utc).isoformat(),post_hoc_known_case_disclosed=True,full_corpus_evaluation_started=False,archive_sha256=plan['archive_sha256'],frozen_P_sha256=plan['frozen_P_sha256'],quality_snapshot=quality,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},unit_tests=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),thresholds_changed=False,training_performed=False,original_evaluator_modified=False))
    print('47 multi-event evaluation plan frozen before full cohort',flush=True)
