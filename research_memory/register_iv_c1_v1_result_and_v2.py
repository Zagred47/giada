"""Preserve IV-C1 v1 failure and preregister its phase-corrected v2 locally."""

import hashlib
import json
import zipfile

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/iv_c1_kaggle_5920b73'
    report = json.loads((folder / 'final_report.json').read_text())
    with zipfile.ZipFile(folder / 'artifact_bundle.zip') as archive:
        assert json.loads(archive.read('final_report.json')) == report
        assert json.loads(archive.read('process_status.json'))['returncode'] == 0
        assert not json.loads(archive.read('code_provenance.json'))['dirty_runtime']
    from src.giada_teacher.iv_c1_deterministic_synapse import load_v2_contract
    cfg = load_v2_contract(repo)
    assert report['valid'] and report['calibration_passed']
    assert not report['iv_c1_passed'] and not report['task33_authorized']
    assert len(report['confirmation']) == 12
    assert all(not x['passed'] for x in report['confirmation'].values())
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c1-v1-v2-{key}-v1', **fields
        })['record_id']

    path = folder / 'artifact_bundle.zip'
    artifact = put('artifacts', 'v1-archive', **{
        'Nome': 'IV-C1 v1 receptor state-phase diagnostic archive', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})
    run = put('runs', 'v1', **{
        'Nome': 'IV-C1 v1 native receptor audit', 'Stato': 'Completata',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle exit 0, archive/report/provenance verified. Scientific state gate failed; conductance/current/charge gates passed.'})
    metric = put('metrics', 'ab-state', **{
        'Nome': 'IV-C1 maximum A/B state error', 'Famiglia': 'Regressione',
        'Direzione': 'Minimizzare', 'Unità': 'state units',
        'Formula': 'Maximum absolute error among A/B receptor states across confirmation samples.'})
    evaluation = put('evaluations', 'ab-state', **{
        'Nome': 'IV-C1 v1 held-out receptor state gate', 'Metrica': [metric],
        'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': '<=0.01',
        'Aggregazione e pesi': 'Maximum over 12 confirmation schedule×voltage cells.'})
    state_max = max(x['max_state_error'] for x in report['confirmation'].values())
    observation = put('observations', 'state-phase', **{
        'Nome': 'IV-C1 v1 A/B state phase fails', 'Valore': state_max,
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': 'State error equals the normalized AMPA jump (0.7537896848); g/i/charge agree to numerical precision; native sample at event time is pre-jump.'})
    finding = put('findings', 'pre-event-state', **{
        'Nome': 'IV-C1 current kernels match but sampled state is left-continuous',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': 'All receptor conductances/currents/charges match, but every v1 state gate fails at the event timestamp because native A/B are sampled before their equal jump.',
        'Limitazioni': 'V1 not promoted. V2 must confirm on new schedules. No plasticity, release RNG or Task33 claim.'})
    claim = put('claims', 'v2-state-convention', **{
        'Nome': 'Left-continuous event-time state observation reconciles IV-C1 A/B states',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': cfg['decision'], 'Limiti': cfg['scope'],
        'Condizioni di falsificazione': cfg['promotion']})
    experiment = put('experiments', 'v2', **{
        'Nome': 'IV-C1 prospective state-phase confirmation v2',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato',
        'Ipotesi': [claim], 'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Confirm sampled A/B phase on new events without changing receptor, conductance/current, charge or threshold contracts.'})
    spec = repo / 'experiments/iv_c1_deterministic_synapse_v2.json'
    put('protocols', 'v2', **{
        'Nome': 'IV-C1 new-schedule left-continuous state audit',
        'Esperimento': [experiment], 'Versione': 'v2',
        'Modalità': 'Componente isolato',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['promotion'],
        'Hash preregistrazione': hashlib.sha256(spec.read_bytes()).hexdigest(),
        'Regole di arresto': 'Calibrate only on v2 calibration event. If calibration fails, do not open v2 confirmation. IV-C1 alone never authorizes Task33.'})
    put('decisions', 'v1-to-v2', **{
        'Nome': 'IV-C1 v1 requires prospective state-phase v2',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'Equal A/B event jump makes g and i insensitive to the state sampling phase; state gate remains failed.',
        'Condizioni di revisione': 'New-schedule v2 state, g, i, charge and negative-control gates.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v1_iv_c1_passed': False,
                      'v2_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
