"""Preserve Task33 v1 technical stop and preregister the independent v2."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    spec_path = repo / 'experiments/task33_observable_synaptic_feedback_v2.json'
    spec = json.loads(spec_path.read_text())
    for path_key, hash_key in (('base_contract', 'base_contract_sha256'),
                               ('prior_calibration', 'prior_calibration_sha256'),
                               ('prior_final_report', 'prior_final_sha256')):
        assert hashlib.sha256((repo / spec[path_key]).read_bytes()).hexdigest() == spec[hash_key]
    report = json.loads((repo / spec['prior_final_report']).read_text())
    calibration = json.loads((repo / spec['prior_calibration']).read_text())
    assert report['diagnosis'] == 'CALIBRATION_IMPLEMENTATION_FAILURE'
    assert not report['confirmation_opened'] and not report['model_judged']
    assert not calibration['passed'] and calibration['shadow']['release_mismatch_count'] == 0
    assert calibration['shadow']['max_rng_sequence_difference'] == 0
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task33-v1-v2-{suffix}-v1', **fields
        })['record_id']

    path = repo / spec['prior_final_report']
    artifact = put('artifacts', 'v1-report', **{
        'Nome': 'Task33 v1 calibration-only report', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})
    run = put('runs', 'calibration', **{
        'Nome': 'Task33 v1 native calibration stop', 'Stato': 'Completata',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle exit 0 and canonical mechanisms compiled; calibration failed before confirmation; no frozen model judged.'})
    metric = put('metrics', 'calibration-current', **{
        'Nome': 'Task33 calibration shadow current max absolute error',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'nA',
        'Formula': 'Maximum absolute native-vs-causal-shadow synaptic current error in calibration only.'})
    evaluation = put('evaluations', 'calibration-current', **{
        'Nome': 'Task33 v1 calibration current gate', 'Metrica': [metric],
        'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': '<=0.0005 nA',
        'Aggregazione e pesi': 'One pre-confirmation calibration schedule.'})
    observation = put('observations', 'offgrid-phase', **{
        'Nome': 'Task33 v1 off-grid receptor phase mismatch',
        'Valore': calibration['shadow']['max_shadow_current_error_na'],
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': 'A/B error 0.066695; current error 0.0038207 nA; release/plasticity/RNG exact; formula V floor 0.10645 mV. Confirmation unopened.'})
    finding = put('findings', 'offgrid-calibration', **{
        'Nome': 'Task33 v1 off-grid event phase prevents admissible calibration',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': 'The event clock is the leading interface hypothesis because release, plastic state and RNG match while receptor A/B do not. This is a calibration failure, not a model verdict.',
        'Limitazioni': 'Timing explanation is still a hypothesis until on-grid v2 calibration. No confirmation outcome exists in v1.'})
    claim = put('claims', 'on-grid-phase', **{
        'Nome': 'Solver-grid event alignment restores Task33 receptor phase',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': spec['reason'],
        'Limiti': 'Fixed 0.1-ms solver with unchanged active-compartment and synapse mechanisms.',
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'v2', **{
        'Nome': 'Task33 on-grid observable synaptic feedback v2',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['reason'],
        'Obiettivo informativo': 'Confirm whether grid-aligned events restore receptor/current interface before opening the unchanged 16-cell active feedback test.'})
    put('protocols', 'v2', **{
        'Nome': 'Task33 v2 on-grid timing protocol', 'Esperimento': [experiment],
        'Versione': 'v2', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Calibration failure stops before confirmation; no model verdict. All numeric gates unchanged.'})
    put('decisions', 'v2-retry', **{
        'Nome': 'Task33 v1 calibration stop; test v2 grid alignment',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'Only timing of prescribed events changes; calibration and confirmation remained disjoint; no fresh model selection.',
        'Condizioni di revisione': 'V2 calibration must pass unchanged gates before opening confirmation.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v1_model_judged': False,
                      'v2_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
