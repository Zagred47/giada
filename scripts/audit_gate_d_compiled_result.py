"""Independent audit of the symmetric compiled Gate D confirmation."""
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'experiments/results/gate_d_compiled_kaggle_e66dfea'


def read(name):return json.loads((FOLDER/name).read_text(encoding='utf-8'))


def audit():
    cfg=json.loads((ROOT/'experiments/task28c_gate_d_compiled_confirmation.json').read_text(encoding='utf-8'))
    report=read('final_report.json');provenance=read('code_provenance.json')
    assert read('run_contract.json')==cfg
    assert read('process_status.json')=={'phase':'gpu','returncode':0}
    assert provenance['code_revision']=='e66dfea644450537445cad2cf8251e9b1a7ee677'
    assert not provenance['dirty_runtime']
    for name,digest in provenance['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in cfg['parent_sha256'].items():
        assert hashlib.sha256((ROOT/cfg['parent_result_dir']/name).read_bytes()).hexdigest()==digest,name
    with zipfile.ZipFile(FOLDER/'artifact_bundle.zip') as z:
        assert z.testzip() is None
        for name in ('final_report.json','run_contract.json','code_provenance.json','process_status.json'):
            assert z.read(name)==(FOLDER/name).read_bytes(),name
    assert report['valid'] and report['eager_registered_passed']
    assert not report['training_performed'] and not report['fresh_used_for_selection']
    assert report['compiler_backend']==cfg['backend'] and report['compiler_mode']==cfg['mode']
    assert len(report['rows'])==len(cfg['families'])
    assert {r['family'] for r in report['rows']}==set(cfg['families'])
    for row in report['rows']:
        assert row['batch_size']==cfg['batch_size']
        assert len(row['accuracy_rows'])==len(cfg['model_seeds'])
        assert {x['seed'] for x in row['accuracy_rows']}==set(cfg['model_seeds'])
        assert row['accuracy_passed']==all(x['passed'] for x in row['accuracy_rows'])
        assert max(row['exact_gate_difference'],row['hybrid_gate_difference'])<=cfg['maximum_compiled_vs_eager_gate_difference']
        assert max(row['exact_current_difference_ma_cm2'],row['hybrid_current_difference_ma_cm2'])<=cfg['maximum_compiled_vs_eager_current_difference_ma_cm2']
        pairs=row['paired_timings_ms']
        assert len(pairs)==cfg['benchmark_repetitions']
        assert all(p['exact']>0 and p['hybrid']>0 and all(math.isfinite(x) for x in p.values()) for p in pairs)
        exact=statistics.median(p['exact'] for p in pairs)
        hybrid=statistics.median(p['hybrid'] for p in pairs)
        gain=statistics.median((p['exact']-p['hybrid'])/p['exact'] for p in pairs)
        assert math.isclose(exact,row['exact_compiled_median_ms'],abs_tol=1e-10)
        assert math.isclose(hybrid,row['hybrid_compiled_median_ms'],abs_tol=1e-10)
        assert math.isclose(gain,row['paired_compute_reduction_fraction'],abs_tol=1e-10)
        assert row['material_reduction_passed']==(gain>=cfg['minimum_material_reduction'])
    passed=all(r['material_reduction_passed'] and r['accuracy_passed'] for r in report['rows'])
    assert report['compiled_confirmation_passed']==passed
    assert report['gate_d_robustly_supported']==passed and report['task29_authorized']==passed
    assert not passed and all(not r['material_reduction_passed'] for r in report['rows'])
    return {'valid':True,'compiled_confirmation_passed':False,'gate_d_completed':False,
            'task29_authorized':False,'exact_compiled_faster_in_both_families':True,
            'rows':[{k:r[k] for k in ('family','exact_compiled_median_ms','hybrid_compiled_median_ms',
                    'paired_compute_reduction_fraction','material_reduction_passed','accuracy_passed',
                    'exact_gate_difference','hybrid_gate_difference')} for r in report['rows']],
            'scope':'Resident Tesla T4, float32, batch 642, same Inductor backend and full local ionic output. No full-neuron speed claim.',
            'numeric_note':'Compiled-vs-eager outputs agree within registered tolerance. Per-seed gate/current RMSEs are below Task28 thresholds; all accuracy flags still fail, consistent with occupancy-bound checks. The report does not store occupancy counts, so this cannot be quantified retrospectively. Compute failure alone blocks Gate D.'}


if __name__=='__main__':
    result=audit()
    (FOLDER/'result_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
