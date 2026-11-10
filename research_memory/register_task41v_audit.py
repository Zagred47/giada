"""Record the post-run41v/41u audit, without training, selection or remote access."""
import argparse
import hashlib
import io
import json
import statistics
import zipfile
from pathlib import Path

from .incremental_batch import IncrementalBatchMirror
from .mirror import ROOT, write_json
from .register_task41d_result import ResultPreview, PreviewComplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    repo = ROOT.parent
    folder = repo / 'experiments/results/task41v_protocol_v1'
    destination = folder / 'detailed_audit_memory_receipt.json'
    assert args.validate_only or not destination.exists(), 'Audit already recorded; inspect receipt instead of repeating.'
    baseline = json.loads((folder / 'result_memory_receipt.json').read_text())['verification']
    final = json.loads((repo / 'artifacts/task41v_kaggle_external/external_backup_receipt.json').read_text())
    assert final['valid'] and not final['technical_failure']
    archive = Path(final['local'])
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == final['remote']['sha256']
    reports, costs = {}, {}
    for task in ('41u', '41v'):
        path = repo / f'artifacts/task{task}_kaggle_external/task{task}_final.zip'
        with zipfile.ZipFile(path) as source:
            reports[task] = json.loads(source.read('final_report.json'))
            costs[task] = [dict(batch_per_seed=row['batch_per_seed'], scope=row['scope'],
                               median_wall_ms={key: statistics.median(value['wall_ms']) for key, value in row['samples'].items()},
                               paired_vs_dual=row['vs_dual']) for row in json.loads(source.read('cost.json'))]
    with zipfile.ZipFile(archive) as source:
        import torch
        torch.set_num_threads(1)
        def finite(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(v) for v in value)
            return True
        checkpoints = []
        for step in (0, 64, 256, 768):
            checkpoint = torch.load(io.BytesIO(source.read(f'checkpoint_{step}.pt')), map_location='cpu', weights_only=True)
            assert finite(checkpoint) and len(checkpoint['models']) == 10
            checkpoints.append(dict(step=step, model_count=10, parameters_masters_optimizer_finite=True))
        incumbent = torch.load(io.BytesIO(source.read('frozen_incumbent.pt')), map_location='cpu', weights_only=True)
        selected = {}
        for name in reports['41v']['arms']:
            weights = torch.load(io.BytesIO(source.read(f'frozen_{name}.pt')), map_location='cpu', weights_only=True)
            assert finite(weights)
            same = all(torch.equal(value, incumbent[key]) for key, value in weights.items())
            if name != 'baseline':
                assert same
            selection = f'selection_{name}.json'
            selected[name] = dict(identical_incumbent=same, selection=json.loads(source.read(selection)) if selection in source.namelist() else None)
        budget = json.loads(source.read('budget_development.json'))['rows']
        old_fail = set(map(tuple, reports['41v']['arms']['incumbent']['development']['failed']))
        for row in budget:
            failures = set(map(tuple, row['development']['failed']))
            row['rescued_cases'] = sorted(old_fail - failures)
            row['newly_failed_cases'] = sorted(failures - old_fail)
        probes = json.loads(source.read('gpu_causal_replication.json'))
        bounds = json.loads(source.read('local_bank_feasibility.json'))
    limitations = ('Opened diagnostic stress, not independent confirmation. Development41u and41v use different inputs/denominators. '
                   'Stress/native/cost results concern selected checkpoints (all fallback0), not every trained intermediate. '
                   'No proof of exhausted optimization or impossible learnability. Teacher probes nonselectable. '
                   'Original FP32 failed controls218/234 and primary330/336 retained. No Task42 or qualified speedup.')
    conclusions = {
        'outcome': 'Technical completion, ten arms reached768, all saved model/master/optimizer states finite. No learned promotion: all ten selected step0, recomposition identical to incumbent; stress66/72 unchanged from41u, same six failures.',
        'regressions': 'No acquired regression in retained model:528/528 trajectories; matched responses336/336; original primary330/336 retained. Trained development regressions excluded before stress/native evaluation. Nap-z local at256 rescues2 but breaks1:138/144 vs incumbent137/144, therefore rejected.',
        'supervision_locality': 'Neither direct raw-logit supervision nor existing local correction bank produced an eligible improvement within768 updates. K direct-global768 preserves137/144 but target timing margin worsens; local K also preserves counts but worsens margin. This does not rule out other optimization/data/representation choices.',
        'causal': 'Canonical GPU probes:control66, K-u12=69, Nap-u8=64, Nap-z9=65, jointNap-u8/z9=68, combined=71/72. Joint/combined have no new stress failures. Remaining seed43 staircase K_Pst offset0.4ms exceeds unchanged0.3ms. Earlier41u K oracle70 used paired offlineFP64 recurrence, not bitwise canonical GPU: do not conflate the two.',
        'local_bound': 'On preregistered voltage anchors, required Nap-u8 residual exceeds existing local bound in17.24%-21.68% of anchors depending modelseed; Nap-z9 in2.93%-8.30%. K-u12 has no anchor outside bound. This proves a range limitation for full anchor-curve matching, not that this alone causes all task failures or all relevant-path errors.',
        'speed': 'No new speedup: selected learned arms are bit-identical incumbent. Integrated medians(ms)41u->41v:batch1 94.117->94.077;642 213.547->214.311;4096 673.333->666.481. Inter-run differences are not an introduced algorithmic speedup. All per-arm paired timings preserved in cost observations.',
        'pause': 'User requested record audit then wait. No next task, training, provider run, selection or threshold/abstraction change authorized by this audit.'}
    audit = dict(valid=True, task='41v', compared_with='41u', conclusions=conclusions, limitations=limitations,
                 checkpoints=checkpoints, selected=selected, all_budget_development=budget,
                 arms={task: report['arms'] for task, report in reports.items()}, costs=costs,
                 canonical_gpu_probes=probes, local_bank_bounds=bounds,
                 final_archive_sha256=final['remote']['sha256'], final_archive_bytes=archive.stat().st_size,
                 next_experiment_launched=False, thresholds_changed=False, abstraction_changed=False)
    mirror = ResultPreview() if args.validate_only else IncrementalBatchMirror()
    if not args.validate_only:
        assert mirror.baseline_hash == baseline['snapshot_hash']
    ids = {}
    def put(table, suffix, **fields):
        code = f'{table}-giada-task41v-audit-{suffix}-v1'
        rid = mirror.local_upsert(table, {'Codice stabile': code, **fields})['record_id']
        ids[code] = rid
        return rid
    exp = json.loads((folder / 'preregistration_receipt.json').read_text())['records']['experiments-giada-task41v-factorial-v1']
    observations = [put('observations', 'comparison', Nome='Task41v detailed comparison with41u, all arms and budgets', Descrizione=json.dumps(audit))]
    for task, arms in audit['arms'].items():
        for name, data in arms.items():
            observations.append(put('observations', f'{task}-{name}', Nome=f'{task} audit selected {name}', Descrizione=json.dumps(data)))
    findings = {}
    for name, text in conclusions.items():
        findings[name] = put('findings', name, Nome='Task41v audit '+name, Esito='Misto', Esperimenti=[exp],
                             Osservazioni=observations, Risultato=text, Limitazioni=limitations)
    claim_receipt = json.loads((folder / 'preregistration_receipt.json').read_text())['records']
    links = {'supervision':'supervision_locality', 'locality':'local_bound', 'Nap_pair':'causal', 'selection':'regressions', 'oracle_replication':'causal'}
    for name, finding in links.items():
        claim = claim_receipt[f'claims-giada-task41v-{name}-v1']
        mirror.local_upsert('claims', {'Codice stabile':f'claims-giada-task41v-{name}-v1',
                           'Stato':'Aperta', 'Limiti':conclusions[finding]+' '+limitations})
        put('evidence', name, Nome='Task41v scoped audit evidence '+name, Esito='Indeterminata',
            **{'Affermazione valutata':[claim], 'Risultati a sostegno':[findings[finding]],
               'Argomentazione':conclusions[finding], 'Limiti e spiegazioni alternative':limitations,
               'Data valutazione':'2026-11-09'})
    for name in ('regressions', 'causal', 'local_bound', 'speed'):
        put('lessons', name, Nome='Task41v '+name, Lezione=conclusions[name], Risultati=[findings[name]], Limiti=limitations)
    put('decisions', 'wait', Nome='Retain incumbent66/72 and wait for user instruction', Esito='Sospendere',
        Motivazione=conclusions['outcome']+' '+conclusions['pause'], Risultati=list(findings.values()), **{'Data':'2026-11-09'})
    mirror.commit_batch()
    if args.validate_only:
        return
    verification = mirror.verify_incremental(baseline)
    assert verification['valid']
    write_json(folder / 'detailed_audit.json', audit)
    write_json(destination, dict(valid=True, verification=verification, records=ids, airtable_accessed=False,
                                next_experiment_launched=False, DB_backup_policy='local complete bundle; last2 verified'))
    mirror.export_snapshot_to(ROOT / 'data/airtable_snapshot.json')
    print('Task41v detailed audit and comparison recorded and verified; no new run.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except PreviewComplete:
        print('Task41v audit schema preview valid; no writes')
