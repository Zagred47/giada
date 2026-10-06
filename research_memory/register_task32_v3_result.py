"""Register the completed Task32 v3 locally; Airtable remains untouched."""

import hashlib
import json
import zipfile

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task32_v3_kaggle_05bdabb'
    raw = (folder / 'final_report.json').read_bytes()
    report = json.loads(raw)
    with zipfile.ZipFile(folder / 'artifact_bundle.zip') as archive:
        assert json.loads(archive.read('final_report.json')) == report
        status = json.loads(archive.read('process_status.json'))
        provenance = json.loads(archive.read('code_provenance.json'))
    assert status['returncode'] == 0 and not provenance['dirty_runtime']
    assert report['code_revision'].startswith('05bdabb')
    assert report['valid'] and report['native_floor_admissible']
    assert report['scientific_primary_passed']
    assert report['source_hypothesis_frozen_before_run'] == 'updated_gate_ica'
    assert not report['training_performed'] and not report['fresh_used_for_selection']
    assert sum(map(len, report['models'].values())) == 6
    assert all(row['primary']['passed'] for family in report['models'].values()
               for row in family.values())
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task32-v3-{key}-v1', **fields
        })['record_id']

    archive_path = folder / 'artifact_bundle.zip'
    artifact = put('artifacts', 'archive', **{
        'Nome': 'Task32 v3 complete frozen-feedback confirmation', 'Tipo': 'Report',
        'Percorso': archive_path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        'Dimensione byte': archive_path.stat().st_size, 'Versione': 'v3'})
    run = put('runs', 'kaggle-05bdabb', **{
        'Nome': 'Task32 v3 process-isolated frozen feedback', 'Stato': 'Completata',
        'Seed': '17, 29, 43', 'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Final report and ZIP concordant; clean source; worker exit 0; native floor and all six frozen model metrics produced.'})
    experiment = mirror.local_upsert('experiments', {
        'Codice stabile': 'experiments-giada-task32-v2-v3-v3-v1',
        'Nome': 'GIADA Task32 prospective phase-aligned v3',
        'Stato': 'Concluso'})['record_id']
    metric = put('metrics', 'voltage-rmse', **{
        'Nome': 'Task32 v3 autonomous voltage RMSE', 'Famiglia': 'Regressione',
        'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'Pooled episode voltage RMSE on the registered 40ms confirmation horizon.'})
    evaluation = put('evaluations', '40ms', **{
        'Nome': 'Task32 v3 frozen feedback 40ms gate', 'Metrica': [metric],
        'Versione': 'v3', 'Ruolo': 'Primaria', 'Target': 'All six pass registered conjunctive thresholds',
        'Aggregazione e pesi': 'Eight confirmation episodes per family and seed; separate native floor.'})
    observations = []
    floor = report['confirmation_formula_floor']
    observations.append(put('observations', 'native-floor', **{
        'Nome': 'Task32 v3 fixed-E_Ca native floor passes',
        'Valore': floor['pooled_voltage_rmse_mv'], 'Run': [run],
        'Specifica di valutazione': [evaluation], 'Artefatti dettagliati': [artifact],
        'Descrizione': f"E_Ca deviation=0; pooled V={floor['pooled_voltage_rmse_mv']:.9g}mV; worst Ca={floor['worst_episode_cai_rmse_mM']:.9g}mM."}))
    for family, seeds in report['models'].items():
        for seed, row in seeds.items():
            primary = row['primary']
            observations.append(put('observations', f'{family}-{seed}', **{
                'Nome': f'Task32 v3 {family} seed {seed} passes',
                'Valore': primary['pooled_voltage_rmse_mv'], 'Run': [run],
                'Specifica di valutazione': [evaluation],
                'Artefatti dettagliati': [artifact],
                'Descrizione': f"40ms pooled V={primary['pooled_voltage_rmse_mv']:.9g}mV; worst V={primary['worst_episode_voltage_rmse_mv']:.9g}mV; worst Ca={primary['worst_episode_cai_rmse_mM']:.9g}mM; SK={primary['worst_episode_sk_gate_rmse']:.9g}."}))
    finding = put('findings', 'narrow-pass', **{
        'Nome': 'Task32 v3 confirms narrow dynamic Ca-SK feedback', 'Esito': 'Positivo',
        'Esperimenti': [experiment], 'Osservazioni': observations,
        'Risultato': 'Fixed-E_Ca native formula floor passes; all 6 frozen family×seed candidates pass 40ms gates; frozen-calcium counterfactual changes Ca, SK and V.',
        'Limitazioni': report['scope'] + ' No synapses, axial coupling, morphology, long-term stability, or speedup claim.'})
    put('decisions', 'to-next-scope', **{
        'Nome': 'Task32 passes and permits next-scope design', 'Esito': 'Continuare',
        'Risultati': [finding],
        'Motivazione': 'Prospective v3 passed after execution isolation. Prior v1/v2 and crash runs retain original status.',
        'Condizioni di revisione': 'New preregistration and native floor required for any expanded domain.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'six_frozen_passed': True,
                      'airtable_accessed': False}))


if __name__ == '__main__':
    main()
