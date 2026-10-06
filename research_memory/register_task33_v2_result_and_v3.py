"""Preserve the Task33 v2 floor stop and preregister an independent v3."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    spec_path = repo / 'experiments/task33_observable_synaptic_feedback_v3.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    for path_key, hash_key in (
        ('parent_v2_contract', 'parent_v2_contract_sha256'),
        ('parent_v2_report', 'parent_v2_report_sha256'),
        ('parent_v2_traces', 'parent_v2_traces_sha256'),
    ):
        assert hashlib.sha256((repo / spec[path_key]).read_bytes()).hexdigest() == spec[hash_key]
    report = json.loads((repo / spec['parent_v2_report']).read_text(encoding='utf-8'))
    assert report['valid'] and report['confirmation_opened'] and report['shadow_passed']
    assert not report['native_floor_admissible'] and not report['model_judged']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task33-v2-v3-{suffix}-v1', **fields
        })['record_id']

    report_path = repo / spec['parent_v2_report']
    artifact = put('artifacts', 'v2-report', **{
        'Nome': 'Task33 v2 native-floor report', 'Tipo': 'Report',
        'Percorso': report_path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'Dimensione byte': report_path.stat().st_size, 'Versione': 'v2'})
    run = put('runs', 'confirmation', **{
        'Nome': 'Task33 v2 16-case native/interface confirmation',
        'Stato': 'Completata', 'Artefatti prodotti': [artifact],
        'Validità tecnica': 'All 16 cases opened; native synaptic shadow and controls passed; calcium formula floor failed; frozen models not judged.'})
    metric = put('metrics', 'calcium-floor', **{
        'Nome': 'Task33 worst formula calcium floor RMSE',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mM',
        'Formula': 'Worst-case calcium RMSE across the 16 active synaptic confirmation cases.'})
    evaluation = put('evaluations', 'calcium-floor', **{
        'Nome': 'Task33 v2 native/interface calcium floor',
        'Metrica': [metric], 'Versione': 'v2', 'Ruolo': 'Primaria',
        'Target': '<=1e-6 mM',
        'Aggregazione e pesi': 'Worst over 16 registered cases; no model judgment if failed.'})
    observation = put('observations', 'floor-failure', **{
        'Nome': 'Task33 v2 calcium floor failure',
        'Valore': 1.3189055e-6, 'Run': [run],
        'Specifica di valutazione': [evaluation], 'Artefatti dettagliati': [artifact],
        'Descrizione': 'All synaptic state/current/release/RNG checks and negative controls passed. Formula V floor pooled 0.130556 mV passed; worst calcium floor 1.3189055e-6 mM exceeded unchanged 1e-6 mM limit. No frozen model verdict.'})
    finding = put('findings', 'old-conductance-phase', **{
        'Nome': 'Task33 v2 prior-tick synaptic conductance phase is required',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': 'Post-hoc native-state one-step audit: prior-tick synaptic conductance gives 0.000714 mV RMSE versus 0.021121 mV for next-tick conductance. Diagnostic replay with only this phase changed gives pooled V floor 0.0001153 mV and worst calcium floor 2.673e-9 mM.',
        'Limitazioni': 'Post-hoc audit is explanatory only; it does not promote v2 or constitute independent confirmation. Frozen models remained unjudged.'})
    claim = put('claims', 'old-phase-v3', **{
        'Nome': 'Old-state synaptic conductance restores active feedback interface',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': spec['diagnosis'],
        'Limiti': 'Task33 fixed 0.1-ms active compartment with four fully observed synapses; no claim about unobserved input or larger morphology.',
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'v3', **{
        'Nome': 'Task33 old-state synaptic feedback v3',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['diagnosis'],
        'Obiettivo informativo': 'Test the single phase correction on unopened seeds and event schedules, then judge frozen Task32 models only if native/interface floors pass.'})
    put('protocols', 'v3', **{
        'Nome': 'Task33 v3 old-state causal phase protocol',
        'Esperimento': [experiment], 'Versione': 'v3',
        'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Calibration/floor failure stops frozen model judgment; all numeric thresholds unchanged.'})
    put('decisions', 'v3-retry', **{
        'Nome': 'Task33 v2 floor stop; independent v3 phase confirmation',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'Single identified causal interface error; post-hoc replay retained only as diagnosis. V3 uses new seed/schedule combinations and unchanged gates.',
        'Condizioni di revisione': 'V3 native/interface floor must pass before any frozen model verdict.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v2_model_judged': False,
                      'v3_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
