"""Record completed Task34 v3 diagnostic in local SQLite, never Airtable."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/results/task34_v3_final_report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['schema_version'] == 'giada-task34-privileged-current-state-probes-v3'
    assert report['code_revision'] == 'e14e2cecb97c17df43dcf55ae8e8a315e79f10dd'
    assert report['valid'] and report['diagnostic_complete'] and report['case_count'] == 16
    assert report['parent_baseline_max_metric_delta'] == 0
    assert report['parent_v2_oracle_metric_delta'] == 0
    assert report['current_identity_max_abs_error_ma_cm2'] < 1e-12
    assert not report['teacher_privileged_arms_selection_eligible']
    assert not report['training_performed'] and not report['model_selection_performed']
    assert not report['independent_confirmation_claimed']
    assert not report['learned_voltage_updater_tested']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task34-v3-{suffix}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'report', **{
        'Nome': 'Task34 v3 matched privileged-probe report', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v3'})
    run = put('runs', 'diagnostic', **{
        'Nome': 'Task34 v3 matched model/formula diagnostic',
        'Stato': 'Completata', 'Artefatti prodotti': [artifact],
        'Validità tecnica': 'All 16 previously opened Task33 cases; six frozen models; exact Task33 and v2 oracle reproduction; current identity passed; no training/selection.'})
    metric = put('metrics', 'state-model-formula', **{
        'Nome': 'Task34 gate-state model/formula RMSE at common teacher V',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'adimensionale',
        'Formula': 'RMSE over 18 gates and 16 cases between frozen model and exact formula, each rolled with the same imposed teacher V.'})
    evaluation = put('evaluations', 'state-model-formula', **{
        'Nome': 'Task34 v3 matched gate-state probe',
        'Metrica': [metric], 'Versione': 'v3', 'Ruolo': 'Diagnostica',
        'Target': 'Descriptive only; no checkpoint selection or new threshold.',
        'Aggregazione e pesi': 'All 18 gate states, 401 ticks and 16 opened cases; family × seed kept separate.'})
    observations = []
    for family, seeds in report['families'].items():
        for seed, row in seeds.items():
            matched = row['matched_model_minus_formula_at_teacher_voltage']
            observations.append(put('observations', f'{family}-{seed}', **{
                'Nome': f'Task34 v3 matched {family} seed {seed}',
                'Valore': matched['state_rmse'], 'Run': [run],
                'Specifica di valutazione': [evaluation],
                'Artefatti dettagliati': [artifact],
                'Descrizione': f"Model/formula gate RMSE {matched['state_rmse']:.9g}; Nap_Et2 current {matched['per_channel_current_rmse_ma_cm2']['Nap_Et2']:.9g} mA/cm2; Ca_HVA current {matched['per_channel_current_rmse_ma_cm2']['Ca_HVA']:.9g} mA/cm2. No model selection."}))
    formula_native = report['formula_minus_native_at_teacher_voltage']
    observations.append(put('observations', 'formula-native', **{
        'Nome': 'Task34 v3 exact formula versus sampled native state',
        'Valore': formula_native['state_rmse'], 'Run': [run],
        'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': f"Formula/native gate RMSE {formula_native['state_rmse']:.9g}; Nap_Et2 current {formula_native['per_channel_current_rmse_ma_cm2']['Nap_Et2']:.9g} mA/cm2; Ca_HVA current {formula_native['per_channel_current_rmse_ma_cm2']['Ca_HVA']:.9g} mA/cm2. This discrepancy exists without a neural model."}))
    finding = put('findings', 'matched-attribution', **{
        'Nome': 'Task34 matched probes separate small learned residual from larger formula/native discrepancy',
        'Esito': 'Positivo', 'Osservazioni': observations,
        'Risultato': 'Six model/formula state RMSEs at matched V span 0.000104–0.000242, versus formula/native 0.003543. Nap_Et2 and Ca_HVA lead the model-specific current residual, but the raw model/native current ranking is dominated by discrepancy already present in exact formula/native comparison. Phase-consistent teacher-state voltage oracle is near formula floor.',
        'Limitazioni': 'All data reused from Task33; no independent generalization, speedup, learned-voltage-updater or full-cell claim. Formula/native per-channel discrepancy should not be attributed entirely to one mechanism without further audit.'})
    put('decisions', 'complete', **{
        'Nome': 'Task34 diagnostic complete; design Task35 separately',
        'Esito': 'Continuare', 'Risultati': [finding],
        'Motivazione': 'The matched probes resolve the attribution ambiguity without retraining. No global gate redesign or checkpoint selection is justified by raw native-state RMSE.',
        'Condizioni di revisione': 'Task35 requires a separately preregistered full-compartment domain and independent evidence for any generalization claim.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task34_diagnostic_complete': True,
                      'observations': len(observations), 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
