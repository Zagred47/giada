"""Structural and provenance audit, not an independent prediction replay."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.giada_teacher.task37_ionic_playground import digest, save_json


def audit():
    folder=ROOT/'experiments/results/task38_kaggle_v2'
    read=lambda n:json.loads((folder/n).read_text())
    final=read('final_report.json'); rows=read('recurrence_evaluation.json'); freeze=read('selection_freeze.json')
    claimed=freeze.pop('freeze_sha256')
    assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==claimed
    assert digest(folder/'frozen_checkpoint.pt')==freeze['checkpoint_sha256']
    assert read('preregistration.json')==json.loads((ROOT/'experiments/task38_joint_transition_preregistration_v1.json').read_text())
    assert read('revision.json')['revision']=='9fe269fcfda669d825ce68ef61852ce8dc4c6a74'
    assert len(rows)==len(set((r['role'],r['family'],r['case'],r['model'],r['seed']) for r in rows))==432
    for name, value in final['summaries'].items():
        selected=[r for r in rows if r['model']==name and r['role']=='development']
        assert len(selected)==72 and sum(r['metrics']['passed'] for r in selected)==value['passed_cases']
        assert all(r['metrics']['passed'] for r in selected)==value['all_development_passed']
        assert max(r['metrics']['gate']['rmse'] for r in selected)==value['worst_gate_rmse']
    assert all(read(n)['valid'] for n in ('formula_preflight.json','optimizer_preflight.json','recurrence_preflight.json'))
    assert not final['quality_thresholds_changed'] and not final['independent_confirmation_accessed']
    assert final['valid'] and final['decision']=='DEVELOPMENT_NO_GO_DIAGNOSE_TASK39'
    report=dict(valid=True,checkpoint_hash_verified=True,freeze_hash_verified=True,rows_verified=432,
        source_revision_verified=True,summary_arithmetic_verified=True,threshold_spec_unchanged=True,
        independent_prediction_replay=False,technical_completion=True,scientific_go=False,
        limitations='Audit of frozen bytes, configuration, preflight and reported metric arithmetic; not an independent CUDA prediction reconstruction. Width32 compiled while width64 fell back to eager; capacity result is scoped, not a clean backend-identical causal contrast.')
    save_json(folder/'result_audit.json',report)
    print(json.dumps(report))


if __name__=='__main__':audit()
