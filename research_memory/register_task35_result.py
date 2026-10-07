"""Record the independent Task35 panel-wise result in local SQLite only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    report_path = repo / 'experiments/results/task35_final_report.json'
    trace_path = repo / 'experiments/results/task35_native_shadow_traces.npz'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == '1ced4c0b2af8afb6a1ca0cdac44c58236d5382359c9091ca19fb2b94778b9bc7'
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == '75028e1cf57203969b96b915c29cf7a170e012b9c6eec943c410e51dd10c0971'
    assert report['schema_version'] == 'giada-task35-full-local-compartment-v1'
    assert report['code_revision'] == 'e1a6efd78bbed9e28a160c75e33f7371cdf72d9a'
    assert report['valid'] and report['task35_passed'] and report['case_count'] == 32
    assert report['calibration']['passed'] and report['floor_passed'] and report['shadow_passed']
    assert report['model_judged'] and all(report['negative_controls'].values())
    assert not report['training_performed'] and not report['teacher_future_used_as_input']
    assert not report['full_cell_claim'] and not report['speedup_claim']
    assert len(report['models']) == 2 and all(len(seeds) == 3 for seeds in report['models'].values())
    assert all(row['passed'] and len(row['panels']) == 4
               for seeds in report['models'].values() for row in seeds.values())

    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task35-{suffix}-v1', **fields
        })['record_id']

    artifacts = []
    for suffix, path, kind in [('report', report_path, 'Report'),
                               ('traces', trace_path, 'Dataset')]:
        artifacts.append(put('artifacts', suffix, **{
            'Nome': f'Task35 independent {suffix}', 'Tipo': kind,
            'Percorso': path.relative_to(repo).as_posix(),
            'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'Dimensione byte': path.stat().st_size, 'Versione': 'v1'}))
    run = put('runs', 'independent-panel-matrix', **{
        'Nome': 'Task35 full declared local compartment, independent conductance panels',
        'Stato': 'Completata', 'Artefatti prodotti': artifacts,
        'Validità tecnica': 'Calibration, synaptic shadow and exact-formula native floor passed before frozen-model judgment; 32 independent cases; no retraining or checkpoint selection.'})
    metric = put('metrics', 'autonomous-voltage-rmse', **{
        'Nome': 'Task35 autonomous 40-ms voltage RMSE',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'RMSE between frozen autonomous model and NEURON V over 401 ticks, reported pooled and per-episode for each conductance panel.'})
    evaluation = put('evaluations', 'panel-wise-voltage', **{
        'Nome': 'Task35 panel-wise autonomous V evaluation', 'Metrica': [metric],
        'Versione': 'v1', 'Ruolo': 'Primaria',
        'Target': 'Unchanged Task33: pooled V ≤1 mV and worst episode V ≤2 mV in every family×seed×panel; Ca ≤1e-5 mM and physical gates.',
        'Aggregazione e pesi': 'Eight cases per panel; separate independent/shared_heads × seeds 17/29/43, no model selection or cross-panel hiding.'})
    observations = []
    for panel, floor in report['formula_floor_by_panel'].items():
        observations.append(put('observations', f'formula-floor-{panel}', **{
            'Nome': f'Task35 formula/native floor {panel}',
            'Valore': floor['pooled_voltage_rmse_mv'], 'Run': [run],
            'Specifica di valutazione': [evaluation], 'Artefatti dettagliati': artifacts,
            'Descrizione': f"Pooled V {floor['pooled_voltage_rmse_mv']:.9g} mV; worst V {floor['worst_voltage_rmse_mv']:.9g} mV; worst Ca {floor['worst_cai_rmse_mM']:.9g} mM. Admissible independent floor."}))
    for family, seeds in report['models'].items():
        for seed, model in seeds.items():
            for panel, row in model['panels'].items():
                observations.append(put('observations', f'{family}-{seed}-{panel}', **{
                    'Nome': f'Task35 {family} seed {seed} {panel}',
                    'Valore': row['pooled_voltage_rmse_mv'], 'Run': [run],
                    'Specifica di valutazione': [evaluation],
                    'Artefatti dettagliati': artifacts,
                    'Descrizione': f"Pooled V {row['pooled_voltage_rmse_mv']:.9g} mV; worst V {row['worst_voltage_rmse_mv']:.9g} mV; worst Ca {row['worst_cai_rmse_mM']:.9g} mM; model and panel pass {model['passed']}."}))
    finding = put('findings', 'local-generalization', **{
        'Nome': 'Task35 frozen local surrogate generalizes across independent conductance panels',
        'Esito': 'Positivo', 'Osservazioni': observations,
        'Risultato': 'All 32 independent 40-ms cases, four conductance panels and six frozen Task32 checkpoints passed unchanged panel-wise Task33 V/Ca/physical gates. Worst model case: independent seed 43 calcium_x4, pooled V 0.010606 mV, worst episode V 0.025443 mV, worst Ca 2.7642e-6 mM. Max formula/native worst V floor 0.002833 mV. Shadow and negative controls passed.',
        'Limitazioni': 'Only one cylindrical compartment with fixed E_Ca and observed presynaptic schedule/RNG. Four x4 panels are controlled stress tests, not full physiological distribution. No axial coupling, morphology, whole cell or speedup claim.'})
    put('decisions', 'complete', **{
        'Nome': 'Task35 local confirmation complete; axial Task36 remains separate',
        'Esito': 'Continuare', 'Risultati': [finding],
        'Motivazione': 'New independent conductance panels extend Task33 canonical local confirmation without retraining, and no panel hides a failed candidate. The next causal boundary is axial coupling, not another local checkpoint selection.',
        'Condizioni di revisione': 'Task36 must independently validate IV-D1 passive axial interface before connecting active compartments; Task35 provides no multicompartment or performance-speed evidence.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task35_complete': True,
                      'observations': len(observations), 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
