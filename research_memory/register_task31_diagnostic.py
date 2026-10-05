"""Record Task31's retrospective exposure decision in the local SQLite mirror only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/results/task31_retrospective_exposure_diagnostic.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['valid'] and len(report['rows']) == 24
    assert not report['scheduled_sampling_training_performed']
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task31-diagnostic-{key}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'report', **{
        'Nome': 'Task31 frozen exposure diagnostic', 'Tipo': 'Report',
        'Percorso': path.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte': path.stat().st_size, 'Versione': 'v1'})
    run = put('runs', 'retrospective', **{
        'Nome': 'Task31 matched frozen voltage-exposure analysis',
        'Stato': 'Completata', 'Seed': '17,29,43',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Parent SHA-256, 2 families × 3 seeds × 4 horizons, 32 paired episodes per cell verified. No new training or test.'})
    eighty = [row for row in report['rows'] if row['horizon_ms'] == 80.0]
    metric = put('metrics', 'gate-exposure-excess', **{
        'Nome': 'Task31 gate RMSE excess under autonomous voltage',
        'Famiglia': 'Regressione', 'Direzione': 'Minimizzare', 'Unità': 'dimensionless gate occupancy',
        'Formula': 'RMS gate error under autonomous voltage minus RMS gate error with teacher-voltage input, paired by family, seed and episode.'})
    evaluation = put('evaluations', 'gate-exposure-excess', **{
        'Nome': 'Task31 retrospective 80ms gate exposure', 'Metrica': [metric],
        'Versione': 'v1', 'Ruolo': 'Diagnostica', 'Target': 'none—exploratory',
        'Aggregazione e pesi': 'Maximum across six family × seed cells; each cell pooled across 32 episodes.'})
    observation = put('observations', 'gate-exposure-excess', **{
        'Nome': 'Task31 maximum 80ms gate feedback excess',
        'Valore': max(row['gate_feedback_excess_rmse'] for row in eighty),
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact], 'Strato o sottogruppo': 'frozen both-arm, 80ms',
        'Descrizione': 'Relative ratios can exceed 1, but absolute gate excess stays below 6.1e-5; worst native voltage episode below 0.03mV.'})
    finding = put('findings', 'not-current-bottleneck', **{
        'Nome': 'Task31 voltage exposure is not the observed bottleneck at this scale',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': 'Autonomous voltage input amplifies some gate error, but 80ms worst native voltage episode is under 0.03mV. No scheduled-sampling continuation justified for these already-passing frozen models.',
        'Limitazioni': 'Retrospective exploratory diagnostic only; no scheduled-sampling training, independent new test, full cell or speed evidence. Does not rule out exposure bias in Task32–35.'})
    put('decisions', 'defer-training', **{
        'Nome': 'Task31 defer scheduled-sampling training at current single-compartment scale',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'No material voltage-feedback failure is observed through 80ms; spending GPU on a continuation now would not address a demonstrated bottleneck.',
        'Condizioni di revisione': 'Reopen on an independently held-out feedback regime with material absolute voltage degradation, e.g. slow dynamics, synapses or multicompartment coupling. Gate D performance NO-GO unchanged.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'airtable_accessed': False,
                      'scheduled_sampling_training_performed': False}))


if __name__ == '__main__':
    main()
