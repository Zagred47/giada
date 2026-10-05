"""Independently verify a completed Task27 artifact before scientific ingestion."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.giada_teacher import heterogeneous_mechanism_composition as h
from src.giada_teacher.hh_family_transfer import sha,write


def audit(folder):
    folder=Path(folder)
    load=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    report,freeze,cfg=load('final_report.json'),load('selection_freeze.json'),load('run_contract.json')
    status,provenance,native,eq=[load(name) for name in
        ('process_status.json','code_provenance.json','native_audit.json','equivalence_preflight.json')]
    claimed=freeze.pop('freeze_sha256')
    canonical=hashlib.sha256(json.dumps(freeze,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    checks={
        'technical_status':status=={'phase':'gpu','returncode':0} and report['valid'],
        'native_oracle':native['valid'],
        'vectorized_equivalence':eq['valid'],
        'freeze_hash':claimed==canonical,
        'fresh_excluded_from_selection':not freeze['fresh_accessed'] and not report['fresh_used_for_selection'],
        'selection_consistent':freeze['selection']==report['selection'],
        'prerequisite_hashes':freeze['prerequisite_hashes']==report['prerequisite_hashes']==cfg['parent_result_hashes'],
        'revision_consistent':freeze['code_revision']==provenance['code_revision'] and not provenance['dirty_runtime'],
        'checkpoint_hashes':all(sha(folder/name)==digest for name,digest in freeze['checkpoint_hashes'].items()),
        'fresh_seed_disjoint':set(cfg['fresh_seeds']).isdisjoint(cfg['data_seeds']['development'])
            and set(cfg['fresh_seeds']).isdisjoint([cfg['data_seeds']['fit'],cfg['data_seeds']['fit_regions'],cfg['data_seeds']['minibatches']]),
    }
    rows=report['learned'];keys={(r['family'],r['arm'],r['seed']) for r in rows}
    expected={(f,a,s) for f in cfg['families'] for a in cfg['supervision_arms'] for s in cfg['seeds']}
    checks['complete_factorial']=keys==expected and len(rows)==len(expected)
    if checks['complete_factorial']:
        recomputed={}
        for family in cfg['families']:
            recomputed[family]={}
            for arm in cfg['supervision_arms']:
                outcomes=[]
                for row in (r for r in rows if r['family']==family and r['arm']==arm):
                    passed=(all(h.passes(m,cfg['gates'],cfg['current_gates']) for m in row['metrics'].values())
                            and all(h.passes(m,cfg['held_gates'],cfg['path_current_gates']) for m in row['held_rollout'].values())
                            and all(h.passes(m,cfg['path_gates'],cfg['path_current_gates'])
                                    for path in row['path_rollout'].values() for m in path.values()))
                    outcomes.append(passed)
                    if row['passed']!=passed:checks['row_flags']=False
                recomputed[family][arm]=all(outcomes)
        checks.setdefault('row_flags',True)
        checks['family_flags']=recomputed==report['per_arm_passed']
        checks['promotion_flag']=report['task28_preparation_authorized']==all(
            recomputed[f]['none'] and recomputed[f]['both'] for f in cfg['families'])
        medians={}
        for family in cfg['families']:
            gains=[]
            for seed in cfg['seeds']:
                pair={arm:next(r for r in rows if r['family']==family and r['seed']==seed and r['arm']==arm)
                      for arm in ('none','both')}
                worst={arm:max(m['currents'][key]/limit for m in row['metrics'].values()
                               for key,limit in cfg['current_gates'].items()) for arm,row in pair.items()}
                gains.append((worst['none']-worst['both'])/max(worst['none'],1e-12))
            medians[family]=float(np.median(gains))
        checks['material_benefit_flag']=all(abs(medians[f]-report['primary_median_worst_current_gain'][f])<1e-9
            and report['current_supervision_material_benefit'][f]==(recomputed[f]['both'] and medians[f]>=.10)
            for f in cfg['families'])
    else:
        recomputed={};medians={}
    result={'schema_version':'giada-task27-result-audit-v1','valid':all(checks.values()),'checks':checks,
            'recomputed_per_arm_passed':recomputed,'recomputed_median_worst_current_gain':medians,
            'result_sha256':sha(folder/'final_report.json'),'freeze_sha256':sha(folder/'selection_freeze.json')}
    write(folder/'result_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    report=audit(Path(args.folder))
    print(json.dumps({'valid':report['valid'],'checks':report['checks']},sort_keys=True))
    if not report['valid']:raise SystemExit(1)
