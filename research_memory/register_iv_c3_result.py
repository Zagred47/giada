"""Record the prospective IV-C3 pass in the local SQLite mirror only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    spec = json.loads((repo / 'experiments/iv_c3_integrated_synaptic_interface_preregistration.json').read_text())
    for parent in spec['parent_reports'].values():
        assert hashlib.sha256((repo / parent['path']).read_bytes()).hexdigest() == parent['sha256']
    folder = repo / 'experiments/results/iv_c3_kaggle_acdac68'
    path = folder / 'final_report.json'
    report = json.loads(path.read_text())
    calibration = json.loads((folder / 'calibration_report.json').read_text())
    assert report['code_revision'] == 'acdac687fb612657e452f68f48ca1c52bc408fcf'
    assert report['teacher_revision'] == spec['teacher_revision']
    assert calibration['passed'] and report['calibration']['passed']
    assert report['valid'] and report['iv_c1_passed'] and report['iv_c2_passed']
    assert report['iv_c3_passed'] and report['task33_authorized']
    assert len(report['confirmation']) == 36
    assert all(row['passed'] for row in report['confirmation'].values())
    assert all(report['negative_controls'].values())
    assert all(report['release_support'][receptor] > 0 for receptor in ('AMPA', 'NMDA', 'GABAA', 'GABAB'))
    for row in report['confirmation'].values():
        for field, limit in spec['gates'].items():
            actual = row[{
                'max_release_mismatch_count': 'release_mismatch_count',
                'max_rng_sequence_difference': 'max_rng_sequence_difference',
            }.get(field, field)]
            assert actual <= limit, (field, actual, limit)
        assert row['release_mismatch_count'] == 0
        assert row['max_native_restart_rng_difference'] == 0
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c3-result-{suffix}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'report', **{
        'Nome': 'IV-C3 Kaggle integrated synaptic interface report', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})
    run = put('runs', 'native', **{
        'Nome': 'IV-C3 mixed-event native confirmation', 'Stato': 'Completata',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle exit 0; pinned code/teacher; calibration passed before 36/36 confirmation cells and three controls.'})
    observations = []
    for suffix, name, field, unit in (
        ('receptor', 'Maximum A/B receptor state error', 'max_receptor_state_error', 'state units'),
        ('conductance', 'Maximum synaptic conductance error', 'max_conductance_error_us', 'µS'),
        ('current', 'Maximum synaptic current error', 'max_current_error_na', 'nA'),
        ('charge', 'Maximum integrated charge error', 'max_integrated_charge_error_na_ms', 'nA·ms'),
        ('restart-state', 'Maximum native restart state error', 'max_native_restart_state_error', 'state units'),
    ):
        value = max(row[field] for row in report['confirmation'].values())
        metric = put('metrics', suffix, **{
            'Nome': name, 'Famiglia': 'Regressione', 'Direzione': 'Minimizzare',
            'Unità': unit, 'Formula': f'Maximum {field} over 36 prospective confirmation cells.'})
        evaluation = put('evaluations', suffix, **{
            'Nome': f'IV-C3 {suffix} confirmation gate', 'Metrica': [metric],
            'Versione': 'v1', 'Ruolo': 'Primaria',
            'Target': f"<={spec['gates'][field]}",
            'Aggregazione e pesi': 'Maximum across all schedules, seeds and imposed voltages.'})
        observations.append(put('observations', suffix, **{
            'Nome': f'IV-C3 {suffix} observed error', 'Valore': value,
            'Run': [run], 'Specifica di valutazione': [evaluation],
            'Artefatti dettagliati': [artifact],
            'Descrizione': 'Calibrated prospective mixed-synapse native/shadow comparison.'}))
    finding = put('findings', 'isolated-interface-pass', **{
        'Nome': 'Causal mixed EMS interface and native restart pass under imposed voltage',
        'Esito': 'Positivo', 'Osservazioni': observations,
        'Risultato': '36/36 cells, all four receptor classes, three negative controls and exact native restart pass; no future teacher-realized release input.',
        'Limitazioni': 'One passive voltage-clamped compartment with four EMS synapses; autonomous voltage and full multicompartmental composition remain untested.'})
    put('decisions', 'task33-preparation', **{
        'Nome': 'IV-C3 passes; Task33 preparation authorized', 'Esito': 'Continuare',
        'Risultati': [finding],
        'Motivazione': 'Prospective isolated mixed-event state, release, conductance/current/charge and restart gates all pass.',
        'Condizioni di revisione': 'Task33 must independently test its own specified domain; IV-C3 does not establish autonomous-voltage validity.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_c3_passed': True,
                      'task33_preparation_authorized': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
