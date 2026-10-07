"""Preserve Task34 v1 oracle-arm error and preregister the phase-corrected v2."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    spec_path = repo / 'experiments/task34_privileged_current_state_probes_v2.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    for field, digest in (('parent_v1_contract', 'parent_v1_contract_sha256'),
                          ('parent_v1_report', 'parent_v1_report_sha256')):
        assert hashlib.sha256((repo / spec[field]).read_bytes()).hexdigest() == spec[digest]
    report = json.loads((repo / spec['parent_v1_report']).read_text(encoding='utf-8'))
    assert report['valid'] and report['diagnostic_complete']
    assert report['parent_baseline_max_metric_delta'] == 0
    assert report['current_identity_max_abs_error_ma_cm2'] < 1e-12
    assert not report['teacher_privileged_arms_selection_eligible']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task34-v1-v2-{suffix}-v1', **fields
        })['record_id']

    path = repo / spec['parent_v1_report']
    artifact = put('artifacts', 'v1-report', **{
        'Nome': 'Task34 v1 privileged-probe report', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})
    run = put('runs', 'v1', **{
        'Nome': 'Task34 v1 diagnostic on Task33 traces',
        'Stato': 'Completata', 'Artefatti prodotti': [artifact],
        'Validità tecnica': 'All 16 cases and six frozen candidates ran; Task33 baseline reproduced exactly and current factorial identity passed. Privileged teacher-state voltage oracle used wrong next-tick phase.'})
    metric = put('metrics', 'oracle-phase', **{
        'Nome': 'Task34 teacher-state oracle voltage RMSE',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'RMSE of autonomous voltage with native gate states provided in a specified phase.'})
    evaluation = put('evaluations', 'oracle-phase', **{
        'Nome': 'Task34 v1 privileged gate-state oracle phase',
        'Metrica': [metric], 'Versione': 'v1', 'Ruolo': 'Diagnostica',
        'Target': 'Phase-consistent with actual native solver',
        'Aggregazione e pesi': '16 previously opened Task33 trajectories; no model selection.'})
    observation = put('observations', 'phase-error', **{
        'Nome': 'Task34 v1 next-state oracle phase is inconsistent',
        'Valore': report['teacher_state_oracle_voltage_rmse_mv'],
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': 'V1 next-state recursive oracle RMSE 0.020944 mV. Post-hoc one-step audit: native old-state 0.0000126046 mV versus native next-state 0.000726650 mV. Other current decomposition and baseline-reproduction diagnostics remain valid; this oracle contrast cannot be interpreted scientifically.'})
    finding = put('findings', 'oracle-phase', **{
        'Nome': 'Task34 v1 privileged oracle arm requires prior-tick state',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': 'The native voltage step uses the gate-state sample from the prior tick. The v1 teacher-state oracle injected a later sample, causing a phase-inconsistent error larger than the deployable baseline.',
        'Limitazioni': 'Post-hoc diagnostic on reused Task33 trajectories. Does not promote v1 privileged oracle or claim independent generalization.'})
    claim = put('claims', 'old-phase', **{
        'Nome': 'Prior-tick teacher gate phase restores Task34 oracle consistency',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': spec['diagnosis'],
        'Limiti': spec['reused_data_status'],
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'v2', **{
        'Nome': 'Task34 phase-consistent privileged probe v2',
        'Tipo': 'Esplorativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['change_only'],
        'Obiettivo informativo': 'Repair only the privileged gate-state oracle phase while retaining the six frozen model baselines and all 11-channel factorial probes.'})
    put('protocols', 'v2', **{
        'Nome': 'Task34 v2 old-state oracle phase protocol',
        'Esperimento': [experiment], 'Versione': 'v2', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'No model selection, retraining or independent confirmation. Mismatch with Task33 baseline is technical.'})
    put('decisions', 'v2', **{
        'Nome': 'Preserve Task34 v1 oracle-arm error and test v2 phase',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'One isolated phase error in a diagnostic arm; no change to deployable baseline or current decomposition.',
        'Condizioni di revisione': 'V2 must reproduce Task33 and report all privilege labels before interpretation.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v1_oracle_interpretable': False,
                      'v2_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
