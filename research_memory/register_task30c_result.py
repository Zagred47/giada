"""Register the independently audited Task30c native-active finding in SQLite."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task30c_kaggle_df82f42'
    report = json.loads((folder / 'final_report.json').read_text(encoding='utf-8'))
    audit = json.loads((folder / 'result_audit.json').read_text(encoding='utf-8'))
    assert report['valid'] and report['scientific_primary_passed'] and audit['valid']
    assert report['native_floor_admissible'] and audit['native_active_single_compartment_validated']
    assert not report['task31_authorized'] and not audit['full_multicompartment_validated']
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task30c-result-{key}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'bundle', **{
        'Nome': 'Task30c native active Kaggle result', 'Tipo': 'Report',
        'Percorso': (folder / 'artifact_bundle.zip').relative_to(repo).as_posix(),
        'SHA-256': audit['artifact_sha256'],
        'Dimensione byte': (folder / 'artifact_bundle.zip').stat().st_size,
        'Versione': 'v1'})
    failed_folder = repo / 'experiments/results/task30c_kaggle_27bebeb_failed'
    failed_report = json.loads((failed_folder / 'failure_report.json').read_text(encoding='utf-8'))
    assert not failed_report['valid'] and not failed_report['scientific_no_go']
    assert failed_report['error'] == 'Canonical NMODL hash mismatch: Ca_HVA'
    failed_artifact = put('artifacts', 'first-preflight-failure', **{
        'Nome': 'Task30c first technical preflight failure', 'Tipo': 'Report',
        'Percorso': (failed_folder / 'artifact_bundle.zip').relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256((failed_folder / 'artifact_bundle.zip').read_bytes()).hexdigest(),
        'Dimensione byte': (failed_folder / 'artifact_bundle.zip').stat().st_size,
        'Versione': 'v1'})
    failed_run = put('runs', 'first-preflight-failure', **{
        'Nome': 'Task30c first run: CRLF/LF source-hash mismatch',
        'Stato': 'Fallita', 'Seed': 'not reached', 'Artefatti prodotti': [failed_artifact],
        'Validità tecnica': 'Stopped before NMODL compilation and any native episode; no scientific NO-GO.'})
    floor_run = put('runs', 'native-floor', **{
        'Nome': 'Task30c eleven-channel formula versus native NEURON',
        'Stato': 'Completata', 'Seed': '32 deterministic episodes',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Pinned 11 original NMODL sources, NEURON 8.2.7, 32/32 finite episodes, 24/24 active exposure, ZIP/code hashes and row coverage independently audited.'})
    floor_metric = put('metrics', 'native-floor-pooled', **{
        'Nome': 'Task30c formula–native voltage floor at 8ms',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'Pooled 8ms RMSE across 32 preregistered episodes.'})
    floor_eval = put('evaluations', 'native-floor-pooled', **{
        'Nome': 'Task30c formula–native floor criterion',
        'Metrica': [floor_metric], 'Versione': 'v1', 'Ruolo': 'Primaria',
        'Target': '0.2', 'Aggregazione e pesi': 'Pooled across 32 episodes; worst episode independently bounded at 0.5mV.'})
    floor_observation = put('observations', 'native-floor-pooled', **{
        'Nome': 'Task30c native floor pooled 8ms RMSE',
        'Valore': audit['native_floor']['pooled_rmse_mv'],
        'Run': [floor_run], 'Specifica di valutazione': [floor_eval],
        'Artefatti dettagliati': [artifact], 'Strato o sottogruppo': 'formula_vs_native',
        'Descrizione': f"Worst episode {audit['native_floor']['worst_episode_rmse_mv']}mV; six native threshold-crossing episodes."})
    findings = [put('findings', 'first-preflight-failure', **{
        'Nome': 'Task30c first attempt had a cross-platform source-hash preflight defect',
        'Esito': 'Misto',
        'Risultato': 'Inventory hash on Windows CRLF bytes rejected the same canonical source checked out as LF on Kaggle; corrected by newline-only normalization.',
        'Limitazioni': 'Technical failure before simulation. Not evidence against the learned model or native solver.',
        'Osservazioni': [],
    }), put('findings', 'native-floor', **{
        'Nome': 'Task30c active native solver floor is negligible',
        'Esito': 'Positivo', 'Osservazioni': [floor_observation],
        'Risultato': 'Exact-formula 11-channel map and native NEURON agree to approximately 1e-12mV over the primary horizon; the active native reference is admissible.',
        'Limitazioni': 'Single compartment; cai, ionic reversals and external current imposed. This verifies solver semantics, not full biological validity.'})]
    for family in ('independent', 'shared_heads'):
        run = put('runs', family+'-both', **{
            'Nome': f'Task30c frozen {family}/both versus native active NEURON',
            'Stato': 'Completata', 'Seed': '17,29,43', 'Artefatti prodotti': [artifact],
            'Validità tecnica': 'Frozen Task27 checkpoints; valid native floor; all primary comparisons independently aggregated and audited.'})
        observations = []
        for seed, values in audit['models'][family].items():
            for metric_key, limit in (('pooled_rmse_mv', '1.0'), ('worst_episode_rmse_mv', '2.0')):
                key = f'{family}-{seed}-{metric_key}'
                metric = put('metrics', key, **{
                    'Nome': f'Task30c native active {metric_key}',
                    'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
                    'Formula': '8ms frozen autonomous voltage versus independent native active NEURON.'})
                evaluation = put('evaluations', key, **{
                    'Nome': f'Task30c {key}', 'Metrica': [metric],
                    'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': limit,
                    'Aggregazione e pesi': 'Per frozen family and seed; no averaging across seeds.'})
                observations.append(put('observations', key, **{
                    'Nome': f'Task30c {key}', 'Valore': values[metric_key],
                    'Run': [run], 'Specifica di valutazione': [evaluation],
                    'Artefatti dettagliati': [artifact], 'Strato o sottogruppo': family+'/both',
                    'Descrizione': 'Valid native floor, 8ms actively exposed primary horizon.'}))
        findings.append(put('findings', family+'-native', **{
            'Nome': f'Task30c {family}: frozen ionic gates retain native active voltage',
            'Esito': 'Positivo', 'Osservazioni': observations,
            'Risultato': 'All three frozen seeds pass the preregistered 1/2mV pooled/worst-episode 8ms native-active limits.',
            'Limitazioni': 'No CaDynamics, axial coupling, synapses, morphology, speed claim or automatic Task31 authorization.'}))
    put('decisions', 'native-active-scoped-go', **{
        'Nome': 'Task30c native active single-compartment scientific GO',
        'Esito': 'Continuare', 'Risultati': findings,
        'Motivazione': 'Independent active NEURON floor is negligible; all six frozen both-arm family × seed comparisons pass with 24/24 non-rest episodes exposed.',
        'Condizioni di revisione': 'Task31 needs a separate preregistered authorization. Full multicompartment, CaDynamics and speed remain unproven; Gate D performance NO-GO unchanged.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'native_active_single_compartment_go': True,
                      'task31_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
