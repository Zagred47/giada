"""Post-freeze residual stratification and read-only SQLite precedent retrieval."""
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/task39c_kaggle_v1'


def main():
    read = lambda name: json.loads((OUT / name).read_text(encoding='utf-8'))
    rows = read('recurrence_evaluation.json')
    norms = json.loads((ROOT / 'experiments/results/task38_kaggle_v2/fit_normalizers.json').read_text())['current']
    summaries = {}
    for model in ('joint_wbase_i1_s0_d1', 'joint_wwide_i1_s1_d1'):
        selected = [r for r in rows if r['model'] == model and r['role'] == 'development']
        timing = []; amplitude = []; causes = Counter(); channels = Counter()
        for r in selected:
            reasons = set()
            if not r['metrics']['gate']['passed']: reasons.add('gate')
            for k, c in enumerate(r['metrics']['channels']):
                if c['nrmse'] > .01: reasons.add('current_rmse')
                if c['nmax'] > .05: reasons.add('current_max')
                if max(abs(c['signed_charge_normalized']), c['absolute_charge_normalized']) > .02: reasons.add('charge')
                if not c['blocked_current_passed']: reasons.add('blocked')
                ev = c['event']; failed = []
                if ev['status'] == 'missing_event': failed.append('missing_event')
                if ev['status'] == 'evaluable':
                    failed = [name for name, lim in (('peak_error_ms', .2), ('onset_error_ms', .3), ('offset_error_ms', .3)) if ev[name] > lim+1e-9]
                if failed:
                    reasons.add('timing'); channels[c['channel']] += 1
                    timing.append(dict(family=r['family'], case=r['case'], seed=r['seed'], channel=c['channel'], failed=failed, metrics=c))
                if c['nrmse'] > .01 or c['nmax'] > .05:
                    amplitude.append(dict(family=r['family'], case=r['case'], seed=r['seed'], channel=c['channel'],
                        global_fit_scale_over_evaluation_scale=norms[k]/c['scale_ma_cm2'], metrics=c))
            assert bool(reasons) == (not r['metrics']['passed']), (model, r['family'], reasons)
            causes.update(reasons)
            if reasons == {'timing'}: causes['timing_only'] += 1
        summaries[model] = dict(passed=sum(r['metrics']['passed'] for r in selected), cases=len(selected),
            overlapping_case_failures=dict(causes), timing_channel_failures=dict(channels),
            timing_component_failures=dict(Counter(x for t in timing for x in t['failed'])),
            timing_details=timing, amplitude_details=amplitude,
            worst_amplitude=sorted(amplitude, key=lambda a:a['metrics']['nrmse'], reverse=True)[:8])
    weights = read('fit_loss_calibration.json')['weights']
    gradients = []
    for g in read('gradient_probes.json'):
        if g['step'] != 120000: continue
        norms_g = np.asarray(g['gradient_term_norms']).T
        weighted = norms_g * np.asarray([.5,.5,*weights])
        gradients.append(dict(architecture=g['architecture'], members=g['members'],
            weighted_term_norms=weighted.tolist(),
            auxiliary_over_trajectory_norm_sum=((weighted[:,2:].sum(1))/(weighted[:,:2].sum(1)+1e-30)).tolist(),
            limits='Norm sums are not norms of summed gradients; cosine/conflict was not measured. Information-off members do not apply auxiliary terms. Fixed probe batch only.'))
    db = sqlite3.connect(f'file:{ROOT / "research_memory/data/research_memory.sqlite"}?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    codes = ['finding-task2b-h-physical-tau-identifiability-repaired-v1',
        'finding-task3d-voltage-support-conditional-shape-repair-v1',
        'findings-giada-roadmap-task24b-clipping-v1',
        'findings-giada-roadmap-task26-gradient_interference-v1',
        'findings-giada-ie-p6-outcome-v1']
    query = 'SELECT * FROM findings WHERE "Codice stabile"=?'
    precedents = [dict(db.execute(query,(code,)).fetchone()) for code in codes]
    result = dict(valid=True,scope='Exploratory development analysis only; no refit or reselection',
        models=summaries, gradients=gradients, precedent_query=query, precedent_codes=codes, precedents=precedents,
        sources={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ('recurrence_evaluation.json','gradient_probes.json','fit_loss_calibration.json')},
        normalizer_source='experiments/results/task38_kaggle_v2/fit_normalizers.json',
        limits='Timing details are evaluator outputs, not a new trace replay. Saved timing NPZ files concern 39b parent oracles, not these 39c candidates. Historical parallels do not establish current causality.')
    (OUT/'residual_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(models={k:{n:v[n] for n in ('passed','overlapping_case_failures','timing_channel_failures','timing_component_failures','worst_amplitude')} for k,v in summaries.items()},
        gradient_ratios={g['architecture']:g['auxiliary_over_trajectory_norm_sum'] for g in gradients},
        precedents=[{k:p.get(k) for k in ('Nome','Risultato','Limitazioni')} for p in precedents]),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
