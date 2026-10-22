"""Read-only post-hoc audit of saved 41e traces, ranks and memory precedents.

Does not train, alter original reports, select a new candidate or relax gates.
"""
import io
import json
import sqlite3
import zipfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.giada_teacher import task41e_transfer_training as tr
from src.giada_teacher import task37_ionic_playground as p

OUT = ROOT / 'experiments/results/task41e_audited_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(post_hoc=True, no_training=True, thresholds_changed=False,
                  original_freeze_unchanged=True, new_candidate_selected=False)
    with zipfile.ZipFile(ROOT / 'artifacts/task41e_reserve_external/giada_task41e_result_recovered.zip') as z:
        load = lambda n: json.loads(z.read(n))
        arrays = lambda n: dict(np.load(io.BytesIO(z.read(n))))
        spec = load('preregistration.json')
        freeze = load('selection_freeze.json')
        curves = load('training_curves.json')
        result['selection_steps'] = {a['name']: freeze['chosen_steps'][3*i:3*i+3]
                                     for i, a in enumerate(freeze['arms'])}
        result['step_zero_count'] = freeze['chosen_steps'].count(0)
        result['members'] = len(freeze['chosen_steps'])
        result['selection_predicate_disagreements'] = []
        result['curves_compact'] = []
        corrected_best = [None]*36
        corrected_steps = [None]*36
        corrected_dev = [None]*36
        initial_rows = load('development_0.json')
        for curve in curves:
            step = curve['step']
            rows = load(f'development_{step}.json')
            discrepancies = []
            for member, cases in enumerate(rows):
                official = sum(any(not r['channels'][k]['passed'] for k in (0,4)) for r in cases)
                ranked = curve['primary_ranks'][member][1]
                introduced=sum(not r['channels'][k]['passed'] and b['channels'][k]['passed']
                    for r,b in zip(cases,initial_rows[member]) for k in range(11))
                introduced+=sum(not r['gate']['passed'] and b['gate']['passed']
                    for r,b in zip(cases,initial_rows[member]))
                # Diagnostic reranking uses the existing evaluator predicates.
                # Clamp severity only for channels already officially passing.
                worst=max(min(tr.channel_severity(r,spec['quality'],(k,)),1.)
                    if r['channels'][k]['passed'] else tr.channel_severity(r,spec['quality'],(k,))
                    for r in cases for k in (0,4))
                corrected=(introduced,official,worst,*curve['primary_ranks'][member][3:])
                if corrected_best[member] is None or corrected<corrected_best[member]:
                    corrected_best[member]=corrected
                    corrected_steps[member]=step
                    corrected_dev[member]=sum(r['passed'] for r in cases)
                if official != ranked:
                    discrepancies.append(dict(member=member, arm=freeze['arms'][member//3]['name'],
                        seed=freeze['seeds'][member%3], official_target_failures=official,
                        ranked_target_failures=ranked,
                        boundary_events=[dict(case=k, channel=r['channels'][ch]['channel'],
                            official_passed=r['channels'][ch]['passed'], event=r['channels'][ch]['event'])
                            for k,r in enumerate(cases) for ch in (0,4)
                            if r['channels'][ch]['passed'] and tr.channel_severity(r,spec['quality'],(ch,))>1]))
            result['selection_predicate_disagreements'].append(dict(step=step, rows=discrepancies))
            response = np.asarray(curve['deployed_response'])
            start_response = np.asarray(curves[0]['deployed_response'])
            gradients = curve.get('gradients', {})
            balance = []
            if gradients:
                point = np.asarray(gradients['point_gradient'])
                weighted = np.asarray(gradients['weighted_feedback_gradient'])
                ratio = weighted / np.maximum(point, 1e-30)
                balance = ratio.tolist()
            result['curves_compact'].append(dict(step=step,
                response_ratio_to_initial=(response/start_response).tolist(),
                weighted_feedback_to_point_gradient=balance,
                primary_ranks=curve['primary_ranks'], budgets=curve['budgets']))
        result['diagnostic_reranking'] = dict(original_steps=freeze['chosen_steps'],
            evaluator_consistent_steps=corrected_steps, development_passed_per_member=corrected_dev,
            changed_members=sum(a!=b for a,b in zip(freeze['chosen_steps'],corrected_steps)),
            ranks=corrected_best, status='Post-hoc sensitivity audit only; no new candidate nominated or test evaluation.')
        result['KPst_oracle'] = load('development_KPst_oracle.json')
        with zipfile.ZipFile(ROOT/'artifacts/task41d_opened41c.zip') as inp:
            meta = json.loads(inp.read('cases.json'))
        original = list(dict.fromkeys([int(meta[k]['source_index']) for k in (108,122)]+[108,122]))
        ref = arrays('opened41c/reference_traces.npz')['autonomous__current']
        probe_reports = {r['label']:r for r in load('coupling_probe/clamp_precision.json')}
        probe_audit=[]
        fullreports = load('opened41c/causal_results.json')
        for name in ('P110','D_C01','D_N000','D_combined'):
            full = arrays(f'opened41c/traces_{name}_autonomous.npz')
            for precision in ('FP32','FP64'):
                probe = arrays(f'coupling_probe/{precision}_V0_Ca0_{name}.npz')
                differences={k:float(np.max(np.abs(full[k][:,:,original].astype('float64')-probe[k].astype('float64')))) for k in full}
                rescored=[]
                original_scores=[]
                for si,seed in enumerate((17,29,43)):
                    for case in (108,122):
                        source=int(meta[case]['source_index'])
                        metric=p.evaluate_intervention(ref[:,source],ref[:,case],
                            probe['current'][:,si,original.index(source)],probe['current'][:,si,original.index(case)],.1,spec['quality'])
                        official=next(r['metrics'] for r in fullreports[name+'/autonomous']['interventions'] if r['case']==case and r['seed']==seed)
                        compact=lambda r:dict(passed=r['passed'],Nap_nrmse=r['channels'][4]['nrmse'])
                        rescored.append(dict(case=case,seed=seed,**compact(metric)))
                        original_scores.append(dict(case=case,seed=seed,**compact(official)))
                saved=probe_reports[precision+'_V0_Ca0']['responses'][name]
                probe_audit.append(dict(model=name,precision=precision,
                    original_selected_cases=original,full_vs_probe_max_abs=differences,
                    original_full_FP32_reference=original_scores,
                    probe_rescored_with_original_full_FP32_reference=rescored,
                    probe_saved_with_FP64_reference=[dict(case=r['case'],seed=r['seed'],passed=r['metrics']['passed'],Nap_nrmse=r['metrics']['channels'][4]['nrmse']) for r in saved]))
        result['probe_reproduction'] = probe_audit
        floor = load('opened41c/floor_autonomous.json')
        cases = [k for k,r in enumerate(meta) if r['intervention']!='none']
        result['canonical_FP32_vs_numpy_floor_responses'] = [dict(case=k, metrics=floor['interventions'][cases.index(k)]) for k in (108,122)]
        result['probe_reference_scope'] = ('Official suite compares to FP32 canonical Torch rollout; coupling probes use FP64 canonical rollout and a smaller batch. '
            'Saved traces permit separating batch/numerical trajectory differences from reference changes without rerunning training. '
            'Rescoring is diagnostic only, not a replacement of any original acceptance result.')
    sql='SELECT [Codice stabile] AS code, Nome, Risultato, Limitazioni FROM findings WHERE [Codice stabile] IN ('+','.join('?' for _ in range(9))+')'
    codes=['findings-giada-task39j-result-sufficient_package-v1','findings-giada-task39j-result-freeze_contrast-v1',
           'findings-giada-task39j-result-numerics-v1','findings-giada-task39j-result-registered_contrasts-v1',
           'findings-giada-task17d-result-voltage-dominance-limited-v1','findings-giada-task17db-result-attribution-v1',
           'findings-giada-task41d-result-objective-v1','findings-giada-task41d-result-feedback-v1',
           'findings-giada-task41d-result-selection-v1']
    with sqlite3.connect((ROOT/'research_memory/data/research_memory.sqlite').as_uri()+'?mode=ro',uri=True) as con:
        con.row_factory=sqlite3.Row
        rows=[dict(r) for r in con.execute(sql,codes)]
    assert len(rows)==len(codes)
    result['memory_query']=dict(sql=sql,parameters=codes,rows=rows,read_only=True,
                                caveat='Historical analogies, not causal proof for 41e.')
    (OUT/'residual_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(step_zero=result['step_zero_count'],members=result['members'],
        rank_disagreements={r['step']:len(r['rows']) for r in result['selection_predicate_disagreements']},
        probes=[dict(model=r['model'],precision=r['precision'],differences=r['full_vs_probe_max_abs'],
            original_pass=sum(x['passed'] for x in r['original_full_FP32_reference']),
            rescored_pass=sum(x['passed'] for x in r['probe_rescored_with_original_full_FP32_reference']),
            saved_pass=sum(x['passed'] for x in r['probe_saved_with_FP64_reference'])) for r in probe_audit],
        memory_precedents=len(rows)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
