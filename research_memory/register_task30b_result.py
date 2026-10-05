"""Record the independently audited, scope-limited Task30b result in SQLite."""
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task30b_kaggle_ddbffb0'
    report = json.loads((folder / 'final_report.json').read_text(encoding='utf-8'))
    audit = json.loads((folder / 'result_audit.json').read_text(encoding='utf-8'))
    assert report['valid'] and report['scientific_primary_passed'] and audit['valid']
    assert audit['decision_grade_within_formula_map_scope']
    assert audit['active_primary_episode_count'] == 24
    assert not audit['full_active_neuron_native_validated'] and not audit['task31_authorized']
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task30b-result-{key}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'bundle', **{
        'Nome': 'Task30b active-exposure Kaggle artifact', 'Tipo': 'Report',
        'Percorso': (folder / 'artifact_bundle.zip').relative_to(repo).as_posix(),
        'SHA-256': audit['artifact_sha256'],
        'Dimensione byte': (folder / 'artifact_bundle.zip').stat().st_size,
        'Versione': 'v1'})
    findings = []
    for family in ('independent', 'shared_heads'):
        run = put('runs', family, **{
            'Nome': f'Task30b {family}/both, early active exposure',
            'Stato': 'Completata', 'Seed': '17,29,43',
            'Artefatti prodotti': [artifact],
            'Validità tecnica': 'Kaggle COMPLETE; artifact, code hashes, rows, exposure and preregistered limits independently audited.'})
        observations = []
        for seed, values in audit['summary'][family].items():
            for metric_key, unit, limit in (
                ('pooled_rmse_mv', 'mV', '1.0'),
                ('worst_episode_rmse_mv', 'mV', '2.0'),
            ):
                key = f'{family}-{seed}-{metric_key}'
                metric = put('metrics', key, **{
                    'Nome': f'Task30b {metric_key}', 'Famiglia': 'Regressione',
                    'Direzione': 'Minimizzare', 'Unità': unit,
                    'Formula': '8ms active-exposed autonomous voltage RMSE, exact aggregation in Task30b config.'})
                evaluation = put('evaluations', key, **{
                    'Nome': f'Task30b {key}', 'Metrica': [metric],
                    'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': limit,
                    'Aggregazione e pesi': 'Per frozen family/seed, both-arm; no averaging across seeds.'})
                observations.append(put('observations', key, **{
                    'Nome': f'Task30b {key}', 'Valore': values[metric_key],
                    'Run': [run], 'Specifica di valutazione': [evaluation],
                    'Artefatti dettagliati': [artifact], 'Strato o sottogruppo': family + '/both',
                    'Descrizione': 'Primary 8ms horizon, early active-current exposure verified.'}))
        findings.append(put('findings', family, **{
            'Nome': f'Task30b {family}: autonomous formula-map confirmation',
            'Esito': 'Positivo', 'Osservazioni': observations,
            'Risultato': 'All three frozen seeds pass the unchanged 8ms pooled and worst-episode voltage limits with verified stimulus exposure.',
            'Limitazioni': 'Single compartment, imposed calcium and injection; internal exact-formula active reference. Only passive NEURON calibration. No full active native or multicomponent validity, speedup, or Task31 authorization.'}))
    put('decisions', 'scoped-go', **{
        'Nome': 'Task30b scoped scientific GO, no Task31 or speed promotion',
        'Esito': 'Continuare', 'Risultati': findings,
        'Motivazione': '24/24 non-rest episodes exposed before 8ms; six reference active threshold crossings; all registered both-arm seeds pass.',
        'Condizioni di revisione': 'Independent full active NEURON confirmation and separate Task31 authorization required. Gate D performance remains NO-GO.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'decision_grade_within_formula_map_scope': True,
                      'task31_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
