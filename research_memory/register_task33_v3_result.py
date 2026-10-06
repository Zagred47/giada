"""Register the independent Task33 v3 pass in the local SQLite mirror only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/results/task33_v3_final_report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['schema_version'] == 'giada-task33-observable-synaptic-feedback-v3'
    assert report['code_revision'] == '21e3c35e883403618a34f17c86a7257f8c0241d2'
    assert report['valid'] and report['task33_passed']
    assert report['native_floor_admissible'] and report['shadow_passed']
    assert report['confirmation_case_count'] == 16 and report['model_judged']
    assert all(row['passed'] for family in report['models'].values() for row in family.values())
    assert not report['training_performed'] and not report['teacher_future_release_used_as_input']
    assert not report['full_cell_claim'] and not report['speedup_claim']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task33-v3-{suffix}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'report', **{
        'Nome': 'Task33 v3 prospective Kaggle report', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v3'})
    run = put('runs', 'confirmation', **{
        'Nome': 'Task33 v3 16-case active synaptic confirmation',
        'Stato': 'Completata', 'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle COMPLETE; native calibration, 16 confirmation cases, negative controls, floor and six frozen checkpoint judgments all valid.'})
    floor_metric = put('metrics', 'floor-voltage', **{
        'Nome': 'Task33 v3 formula/native pooled voltage floor RMSE',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'Pooled 40-ms voltage RMSE over all 16 active synaptic cases.'})
    floor_eval = put('evaluations', 'floor-voltage', **{
        'Nome': 'Task33 v3 native floor before model judgment',
        'Metrica': [floor_metric], 'Versione': 'v3', 'Ruolo': 'Primaria',
        'Target': 'Registered floor gates: pooled V <=0.2 mV, worst V <=0.5 mV, worst cai <=1e-6 mM',
        'Aggregazione e pesi': '16 preregistered cases; no checkpoint judgment if failed.'})
    floor = report['formula_floor']
    floor_observation = put('observations', 'floor', **{
        'Nome': 'Task33 v3 admissible native floor',
        'Valore': floor['pooled_voltage_rmse_mv'], 'Run': [run],
        'Specifica di valutazione': [floor_eval], 'Artefatti dettagliati': [artifact],
        'Descrizione': f"Pooled V {floor['pooled_voltage_rmse_mv']:.9g} mV; worst V {floor['worst_voltage_rmse_mv']:.9g} mV; worst cai {floor['worst_cai_rmse_mM']:.9g} mM. Shadow and controls passed; 88 realized releases/256 scheduled; max synaptic V effect 15.994 mV."})
    model_metric = put('metrics', 'model-voltage', **{
        'Nome': 'Task33 frozen checkpoint pooled voltage RMSE',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'Pooled 40-ms autonomous voltage RMSE over 16 cases for each frozen Task32 family and seed.'})
    model_eval = put('evaluations', 'model-voltage', **{
        'Nome': 'Task33 v3 frozen checkpoint evaluation',
        'Metrica': [model_metric], 'Versione': 'v3', 'Ruolo': 'Primaria',
        'Target': 'All registered voltage/calcium, physical and occupancy gates passed',
        'Aggregazione e pesi': 'Each family × seed evaluated independently on the same 16 cases after floor admission.'})
    model_observations = []
    for family, seeds in report['models'].items():
        for seed, values in seeds.items():
            model_observations.append(put('observations', f'{family}-{seed}', **{
                'Nome': f'Task33 v3 {family} seed {seed}',
                'Valore': values['pooled_voltage_rmse_mv'], 'Run': [run],
                'Specifica di valutazione': [model_eval],
                'Artefatti dettagliati': [artifact],
                'Descrizione': f"Pooled V {values['pooled_voltage_rmse_mv']:.9g} mV; worst V {values['worst_voltage_rmse_mv']:.9g} mV; worst cai {values['worst_cai_rmse_mM']:.9g} mM; passed={values['passed']}."}))
    finding = put('findings', 'confirmation', **{
        'Nome': 'Task33 fully observed synaptic feedback confirmed in one compartment',
        'Esito': 'Positivo',
        'Osservazioni': [floor_observation, *model_observations],
        'Risultato': 'Prospective v3 fixed the causal prior-tick synaptic conductance phase. Native shadow and negative controls passed in 16 new cases; formula/native floor passed; all six frozen Task32 checkpoints passed without retraining or future release input.',
        'Limitazioni': 'One active compartment, four fully observed synapses, 40 ms, 0.1-ms step. No unobserved-input, multicompartment, axial-coupling, whole-cell or speedup claim.'})
    put('decisions', 'complete', **{
        'Nome': 'Task33 passed within its registered one-compartment scope',
        'Esito': 'Continuare', 'Risultati': [finding],
        'Motivazione': 'Independent v3 confirmation passed all frozen gates after resolving v1/v2 interface failures without retrospective promotion.',
        'Condizioni di revisione': 'Task34 privileged probes and subsequent multicompartment steps require separate preregistration and evidence.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task33_passed': True,
                      'model_count': len(model_observations), 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
