"""Recompute the archived eager Gate D decision and preserve its scope."""
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'experiments/results/gate_d_compute_kaggle_59c91e7'


def read(name):return json.loads((FOLDER/name).read_text(encoding='utf-8'))


def audit():
    cfg=json.loads((ROOT/'experiments/task28b_gate_d_compute.json').read_text(encoding='utf-8'))
    report=read('final_report.json');provenance=read('code_provenance.json')
    assert read('run_contract.json')==cfg
    assert read('process_status.json')=={'phase':'gpu','returncode':0}
    assert provenance['code_revision']=='59c91e73b7b246a201bd40aea22e8f4cc246bfc6'
    assert not provenance['dirty_runtime']
    for name,digest in provenance['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in provenance['parent_hashes'].items():
        assert hashlib.sha256((ROOT/cfg['task28_result_dir']/name).read_bytes()).hexdigest()==digest,name
    with zipfile.ZipFile(FOLDER/'artifact_bundle.zip') as z:
        assert z.testzip() is None
        for name in ('run_contract.json','code_provenance.json','final_report.json','process_status.json'):
            assert z.read(name)==(FOLDER/name).read_bytes()
    assert report['valid'] and report['task28_teacher_forced_passed']
    assert not report['training_performed'] and not report['fresh_used_for_selection']
    assert report['code_revision']==provenance['code_revision']
    expected={(family,batch) for family in cfg['families'] for batch in cfg['batch_sizes']}
    assert len(report['rows'])==len(expected)
    assert {(r['family'],r['batch_size']) for r in report['rows']}==expected
    for row in report['rows']:
        assert row['model_seeds']==cfg['model_seeds']
        assert len(row['accuracy_rows'])==3 and all(x['seed'] in cfg['model_seeds'] for x in row['accuracy_rows'])
        assert row['accuracy_passed']==all(x['passed'] for x in row['accuracy_rows'])
        assert row['exact_gpu_vs_numpy_max_gate_error']<=cfg['preflight_max_gate_error']
        assert row['exact_gpu_vs_numpy_max_current_error_ma_cm2']<=cfg['preflight_max_current_error_ma_cm2']
        pairs=row['paired_timings_ms']
        assert len(pairs)==cfg['benchmark_repetitions']
        assert all(p['exact']>0 and p['hybrid']>0 and all(math.isfinite(x) for x in p.values()) for p in pairs)
        exact=statistics.median(p['exact'] for p in pairs)
        hybrid=statistics.median(p['hybrid'] for p in pairs)
        gain=statistics.median((p['exact']-p['hybrid'])/p['exact'] for p in pairs)
        assert math.isclose(row['exact_median_ms'],exact,abs_tol=1e-10)
        assert math.isclose(row['hybrid_median_ms'],hybrid,abs_tol=1e-10)
        assert math.isclose(row['median_paired_compute_reduction_fraction'],gain,abs_tol=1e-10)
        assert row['material_reduction_passed']==(gain>=cfg['material_compute_reduction_minimum'])
    primary=[r for r in report['rows'] if r['batch_size']==cfg['gate_d_batch_size']]
    eager_pass=all(r['material_reduction_passed'] and r['accuracy_passed'] for r in primary)
    assert report['gate_d_completed']==eager_pass and report['task29_authorized']==eager_pass
    return {'valid':True,'eager_registered_passed':eager_pass,'families':{
        r['family']:{'exact_median_ms':r['exact_median_ms'],'hybrid_median_ms':r['hybrid_median_ms'],
                     'paired_reduction_fraction':r['median_paired_compute_reduction_fraction']}
        for r in primary},'compiled_baseline_tested':False,
        'scientific_caution':'Eager-only same-GPU comparison; optimized formula baseline not yet tested. Task29 promotion awaits symmetric compiled confirmation.'}


if __name__=='__main__':
    result=audit()
    (FOLDER/'result_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
