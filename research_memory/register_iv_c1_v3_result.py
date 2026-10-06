"""Record the prospective IV-C1 pass, without promoting IV-C2/IV-C3/Task33."""

import hashlib
import json
import zipfile

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/iv_c1_v3_kaggle_634adea'
    report = json.loads((folder / 'final_report.json').read_text())
    provenance = json.loads((folder / 'code_provenance.json').read_text())
    status = json.loads((folder / 'process_status.json').read_text())
    calibration = json.loads((folder / 'calibration_decision.json').read_text())
    with zipfile.ZipFile(folder / 'artifact_bundle.zip') as archive:
        assert json.loads(archive.read('final_report.json')) == report
    assert status['returncode'] == 0 and not provenance['dirty_runtime']
    assert provenance['code_revision'] == report['code_revision']
    assert calibration['passed'] and not calibration['confirmation_accessed']
    assert report['valid'] and report['iv_c1_passed']
    assert len(report['confirmation']) == 12
    assert all(row['passed'] for row in report['confirmation'].values())
    assert all(report['negative_controls'].values())
    assert not report['iv_c2_passed'] and not report['iv_c3_passed']
    assert not report['task33_authorized']
    state_max = max(row['max_state_error'] for row in report['confirmation'].values())
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c1-v3-result-{suffix}-v1', **fields
        })['record_id']

    path = folder / 'artifact_bundle.zip'
    artifact = put('artifacts', 'archive', **{
        'Nome': 'IV-C1 v3 prospective confirmation bundle', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v3'})
    run = put('runs', 'native', **{
        'Nome': 'IV-C1 v3 deterministic receptor confirmation',
        'Stato': 'Completata', 'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Exit 0; clean code provenance; calibration frozen before confirmation; 12/12 prospective cells and controls pass.'})
    metric = put('metrics', 'ab-state', **{
        'Nome': 'IV-C1 v3 maximum A/B state error', 'Famiglia': 'Regressione',
        'Direzione': 'Minimizzare', 'Unità': 'state units',
        'Formula': 'Maximum absolute A/B error over all 12 confirmation cells.'})
    evaluation = put('evaluations', 'ab-state', **{
        'Nome': 'IV-C1 v3 A/B state confirmation', 'Metrica': [metric],
        'Versione': 'v3', 'Ruolo': 'Primaria', 'Target': '<=0.01',
        'Aggregazione e pesi': 'Maximum over all confirmed schedules, voltages, receptors and samples.'})
    observation = put('observations', 'state-pass', **{
        'Nome': 'IV-C1 v3 A/B state gate passes', 'Valore': state_max,
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': '12/12 prospective cells passed; max A/B error 1.12e-12, g error 9.71e-16 uS, i error 8.04e-14 nA; all controls passed.'})
    finding = put('findings', 'clock-interface', **{
        'Nome': 'Actual solver clock resolves the IV-C1 event-state boundary',
        'Esito': 'Positivo', 'Osservazioni': [observation],
        'Risultato': 'Public h.t as an event-boundary input reconciles native and analytic A/B states on disjoint v3 schedules; unchanged g/i/charge gates pass.',
        'Limitazioni': 'Only isolated deterministic one-release synapses at imposed voltage. Plasticity, RNG, integrated histories, autonomous voltage and dendritic coupling remain untested.'})
    put('decisions', 'iv-c1-pass', **{
        'Nome': 'IV-C1 passes; advance to IV-C2, not Task33',
        'Esito': 'Continuare', 'Risultati': [finding],
        'Motivazione': 'Prospective v3 passes all 12 state/current/charge cells and negative controls after an explanatory diagnostic isolated the public-clock boundary.',
        'Condizioni di revisione': 'IV-C2 plasticity/release RNG and IV-C3 integrated interface must independently pass before Task33.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_c1_passed': True,
                      'iv_c2_passed': False, 'iv_c3_passed': False,
                      'task33_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
