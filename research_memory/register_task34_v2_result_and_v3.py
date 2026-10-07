"""Preserve Task34 v2 and preregister matched model/formula probes in SQLite."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    spec_path = repo / 'experiments/task34_privileged_current_state_probes_v3.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    for field, digest in (('parent_v2_contract', 'parent_v2_contract_sha256'),
                          ('parent_v2_report', 'parent_v2_report_sha256')):
        assert hashlib.sha256((repo / spec[field]).read_bytes()).hexdigest() == spec[digest]
    report = json.loads((repo / spec['parent_v2_report']).read_text(encoding='utf-8'))
    assert report['valid'] and report['diagnostic_complete']
    assert report['parent_baseline_max_metric_delta'] == 0
    assert not report['model_selection_performed'] and not report['independent_confirmation_claimed']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task34-v2-v3-{suffix}-v1', **fields
        })['record_id']

    path = repo / spec['parent_v2_report']
    artifact = put('artifacts', 'v2-report', **{
        'Nome': 'Task34 v2 phase-corrected privileged-probe report', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v2'})
    run = put('runs', 'v2', **{
        'Nome': 'Task34 v2 matched phase-corrected diagnostic',
        'Stato': 'Completata', 'Artefatti prodotti': [artifact],
        'Validità tecnica': '16 opened Task33 cases; six frozen baselines reproduced exactly; current identity passed; old-state oracle corrected.'})
    metric = put('metrics', 'oracle-v', **{
        'Nome': 'Task34 corrected teacher-state oracle V RMSE',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'Autonomous analytic voltage RMSE with phase-consistent prior-tick native gates.'})
    evaluation = put('evaluations', 'oracle-v', **{
        'Nome': 'Task34 v2 privileged teacher-state oracle',
        'Metrica': [metric], 'Versione': 'v2', 'Ruolo': 'Diagnostica',
        'Target': 'Interpret against Task33 formula floor and frozen baselines, no selection.',
        'Aggregazione e pesi': 'All 16 previously opened Task33 v3 cases.'})
    observation = put('observations', 'oracle-v', **{
        'Nome': 'Task34 v2 old-state teacher oracle approaches formula floor',
        'Valore': report['teacher_state_oracle_voltage_rmse_mv'],
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': 'Old-state oracle V RMSE 0.000339014 mV versus v1 phase-inconsistent 0.020944 mV; formula floor 0.000365566 mV. All six frozen model baselines are 0.000885–0.002237 mV. Teacher V does not reduce model gate RMSE by 2x; teacher state reduces V error by 2x for all six.'})
    finding = put('findings', 'raw-error-ambiguous', **{
        'Nome': 'Task34 v2 raw native-state current ranking is not model-specific',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': 'Phase-consistent teacher-state oracle is near formula floor; raw current errors are state-dominated. Post-hoc exact-formula/native comparison on the same cases shows nearly the same Nap_Et2, Ca_HVA and Im current ranking without a neural model.',
        'Limitazioni': 'Native-state sampling/formula mismatch is not separated from learned-model error in v2. Existing Task33 cases are reused, not an independent test.'})
    claim = put('claims', 'matched-formula', **{
        'Nome': 'Matched formula-relative probes isolate learned gate error',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': spec['diagnosis'],
        'Limiti': spec['interpretation'],
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'v3', **{
        'Nome': 'Task34 matched model-versus-formula diagnostic v3',
        'Tipo': 'Esplorativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['diagnosis'],
        'Obiettivo informativo': 'Disentangle learned-model gate/current discrepancy from exact-formula/native-sampling discrepancy under identical imposed voltage.'})
    put('protocols', 'v3', **{
        'Nome': 'Task34 v3 matched formula-relative probe protocol',
        'Esperimento': [experiment], 'Versione': 'v3', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Parent baseline/factorial mismatch is technical. No checkpoint selection, training or independent confirmation.'})
    put('decisions', 'v3', **{
        'Nome': 'Task34 v2 supports matched formula-relative diagnostic before Task35',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'Raw state/current ranking is shared by the exact formula and the model; added matched probes test the model-specific residual before architecture design.',
        'Condizioni di revisione': 'V3 finite paired measurements and parent reproduction required.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v3_preregistered': True,
                      'independent_confirmation_claimed': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
