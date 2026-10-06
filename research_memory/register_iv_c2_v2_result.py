"""Record IV-C2 native distribution pass and promote only to IV-C3."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    report_path = repo / 'experiments/results/iv_c2_native_distribution_8edd4c2/final_report.json'
    report = json.loads(report_path.read_text())
    spec = json.loads((repo / 'experiments/iv_c2_native_distribution_confirmation.json').read_text())
    parent_path = repo / spec['parent_report']
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == spec['parent_report_sha256']
    assert report['valid'] and report['iv_c2_passed'] and report['cell_count'] == 16
    assert len(report['cells']) == 16 and all(row['passed'] for row in report['cells'])
    assert report['max_release_absolute_error'] <= spec['max_absolute_probability_error']
    assert report['max_recovery_absolute_error'] <= spec['max_absolute_probability_error']
    assert report['uniform_negative_control_passed']
    assert not report['iv_c3_passed'] and not report['task33_authorized']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c2-v2-result-{suffix}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'report', **{
        'Nome': 'IV-C2 v2 native release-frequency report', 'Tipo': 'Report',
        'Percorso': report_path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'Dimensione byte': report_path.stat().st_size, 'Versione': 'v2'})
    run = put('runs', 'native', **{
        'Nome': 'IV-C2 v2 native frequency confirmation',
        'Stato': 'Completata', 'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle exit 0, pinned teacher and code, 16/16 prospective cells and uniform negative control passed.'})
    metrics = []
    observations = []
    for suffix, name, value in (
        ('release', 'Maximum native release-frequency absolute error', report['max_release_absolute_error']),
        ('recovery', 'Maximum native recovery-frequency absolute error', report['max_recovery_absolute_error'])):
        metric = put('metrics', suffix, **{
            'Nome': name, 'Famiglia': 'Regressione', 'Direzione': 'Minimizzare',
            'Unità': 'probability', 'Formula': 'Maximum absolute difference between native frequency and preregistered negexp(1) prediction over 16 cells.'})
        evaluation = put('evaluations', suffix, **{
            'Nome': f'IV-C2 v2 {suffix} distribution gate', 'Metrica': [metric],
            'Versione': 'v2', 'Ruolo': 'Primaria', 'Target': '<=0.08',
            'Aggregazione e pesi': 'Maximum over both mechanisms, Use, Dep and initial-state conditions.'})
        observation = put('observations', suffix, **{
            'Nome': f'IV-C2 v2 {suffix} frequency error', 'Valore': value,
            'Run': [run], 'Specifica di valutazione': [evaluation],
            'Artefatti dettagliati': [artifact],
            'Descrizione': 'Native release/recovery frequencies across 256 paired fresh seeds per cell.'})
        metrics.append(metric)
        observations.append(observation)
    finding = put('findings', 'distribution-pass', **{
        'Nome': 'Native EMS release and recovery distributions match Random123 negexp predictions',
        'Esito': 'Positivo', 'Osservazioni': observations,
        'Risultato': '16/16 cells pass; max release frequency error 0.03289, max recovery error 0.02655; uniform-control mismatch 0.26094.',
        'Limitazioni': 'Isolated canonical synapses with imposed voltage. Mixed synapse histories/current interface and autonomous voltage are not tested; IV-C3 is required.'})
    put('decisions', 'iv-c2-pass', **{
        'Nome': 'IV-C2 passes; advance to IV-C3, not Task33',
        'Esito': 'Continuare', 'Risultati': [finding],
        'Motivazione': 'v1 matched native release, plastic state, RNG sequence and restart exactly in 64 cases; independent v2 matched native release/recovery frequencies in 16 cells.',
        'Condizioni di revisione': 'IV-C3 mixed-event current interface and checkpoint/rollout gate must pass before Task33.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_c2_passed': True,
                      'iv_c3_passed': False, 'task33_authorized': False,
                      'airtable_accessed': False}))


if __name__ == '__main__':
    main()
