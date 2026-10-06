"""Register completed IV-C1 v2 failure; keep Task33 unauthorized."""

import hashlib
import json
import zipfile

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/iv_c1_v2_kaggle_f28d3bc'
    report_path = folder / 'final_report.json'
    archive_path = folder / 'artifact_bundle.zip'
    report = json.loads(report_path.read_text())
    with zipfile.ZipFile(archive_path) as archive:
        assert json.loads(archive.read('final_report.json')) == report
        assert json.loads(archive.read('process_status.json'))['returncode'] == 0
        assert not json.loads(archive.read('code_provenance.json'))['dirty_runtime']
    assert report['valid'] and report['calibration_passed']
    assert len(report['confirmation']) == 12
    assert all(not cell['passed'] for cell in report['confirmation'].values())
    assert not report['iv_c1_passed'] and not report['task33_authorized']
    state_max = max(cell['max_state_error'] for cell in report['confirmation'].values())
    conductance_max = max(cell['max_g_error_us'] for cell in report['confirmation'].values())
    current_max = max(cell['max_i_error_na'] for cell in report['confirmation'].values())
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c1-v2-{suffix}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'archive', **{
        'Nome': 'IV-C1 v2 prospective confirmation archive', 'Tipo': 'Report',
        'Percorso': archive_path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        'Dimensione byte': archive_path.stat().st_size, 'Versione': 'v2'})
    run = put('runs', 'native', **{
        'Nome': 'IV-C1 v2 native receptor audit', 'Stato': 'Completata',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle exit 0, clean runtime, archive/report equality. Scientific state gate failed.'})
    metric = put('metrics', 'ab-state', **{
        'Nome': 'IV-C1 v2 maximum A/B state error', 'Famiglia': 'Regressione',
        'Direzione': 'Minimizzare', 'Unità': 'state units',
        'Formula': 'Maximum absolute A/B error over every sample, receptor, schedule and voltage.'})
    evaluation = put('evaluations', 'ab-state', **{
        'Nome': 'IV-C1 v2 held-out receptor state gate', 'Metrica': [metric],
        'Versione': 'v2', 'Ruolo': 'Primaria', 'Target': '<=0.01',
        'Aggregazione e pesi': 'Maximum across 12 prospective confirmation cells.'})
    observation = put('observations', 'state-failure', **{
        'Nome': 'IV-C1 v2 state gate fails despite current agreement',
        'Valore': state_max, 'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': f'Max A/B error {state_max:.9g}; max conductance error '
                       f'{conductance_max:.9g} uS; max current error {current_max:.9g} nA.'})
    finding = put('findings', 'left-continuous-falsified', **{
        'Nome': 'Simple left-continuous state timing does not generalize',
        'Esito': 'Negativo', 'Osservazioni': [observation],
        'Risultato': 'Prospective v2 state gate failed in all 12 cells while g/i/charge and negative controls passed.',
        'Limitazioni': 'Precise native NetCon/state recorder order is unknown. Forensic replay is diagnostic only; no IV-C1 promotion.'})
    put('decisions', 'forensic-before-v3', **{
        'Nome': 'IV-C1 remains blocked pending per-event native state forensic',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'V2 falsified the simple left-continuous event-time rule; do not guess another convention or reuse opened schedules as confirmation.',
        'Condizioni di revisione': 'Inspect actual h.t and per-synapse A/B around events; preregister a disjoint prospective confirmation only after identifying the mechanism.'})
    # Correct the original v1 interpretation in place: the proposed timing
    # mechanism was not directly observed and its prospective v2 failed.
    mirror.local_upsert('observations', {
        'Codice stabile': 'observations-giada-iv-c1-v1-v2-state-phase-v1',
        'Descrizione': 'V1 state error equals a normalized AMPA jump; g/i/charge agree. A pre-event recording convention was hypothesized but not established by v1 and did not explain v2.'})
    mirror.local_upsert('findings', {
        'Codice stabile': 'findings-giada-iv-c1-v1-v2-pre-event-state-v1',
        'Nome': 'IV-C1 current kernels match but A/B event timing remains unresolved',
        'Risultato': 'V1 and prospective v2 both fail the A/B state gate while g/i/charge agree; the simple left-continuous convention is insufficient.',
        'Limitazioni': 'Exact native event/state recorder order is not yet identified. No IV-C1 promotion or Task33 authorization.'})
    mirror.local_upsert('claims', {
        'Codice stabile': 'claims-giada-iv-c1-v1-v2-v2-state-convention-v1',
        'Limiti': 'Prospective v2 contradicted the simple global pre-event state convention; per-event native trace is required before any replacement hypothesis.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'state_max': state_max,
                      'iv_c1_passed': False, 'task33_authorized': False,
                      'airtable_accessed': False}))


if __name__ == '__main__':
    main()
