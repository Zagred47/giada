"""Record audited causal latency attribution in the SQLite mirror only."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/gate_d_forensic_kaggle_b19aa65'
    report = json.loads((folder / 'final_report.json').read_text(encoding='utf-8'))
    audit = json.loads((folder / 'result_audit.json').read_text(encoding='utf-8'))
    assert report['valid'] and audit['valid'] and not report['task29_authorized']
    mirror = ValidatedBatchMirror()
    def put(table, key, **fields):
        return mirror.local_upsert(table, {'Codice stabile': f'{table}-giada-gated-forensic-{key}-v1', **fields})['record_id']
    artifact = put('artifacts', 'bundle', **{
        'Nome': 'Gate D compiled bottleneck forensic artifact bundle', 'Tipo': 'Report',
        'Percorso': (folder / 'artifact_bundle.zip').relative_to(repo).as_posix(),
        'SHA-256': audit['artifact_sha256'],
        'Dimensione byte': (folder / 'artifact_bundle.zip').stat().st_size, 'Versione': 'v1'})
    for row in report['rows']:
        family = row['family']
        run = put('runs', family, **{
            'Nome': f'Gate D bottleneck forensic {family}', 'Stato': 'Completata',
            'Seed': '17,29,43', 'Artefatti prodotti': [artifact],
            'Validità tecnica': 'Kaggle COMPLETE/exit0; immutable commit, source hashes, ZIP CRC, decomposition identity, 40 paired timings and four-call CUDA profiler independently audited.'})
        observations = []
        for key in ('exact_core_five', 'neural_core_five', 'exact_full', 'hybrid_full'):
            metric = put('metrics', family + '-' + key, **{
                'Nome': f'Gate D {family} {key} latency', 'Famiglia': 'Regressione',
                'Direzione': 'Minimizzare', 'Unità': 'ms',
                'Formula': 'Median CUDA event time over 40 warmed, alternating-order calls.'})
            evaluation = put('evaluations', family + '-' + key, **{
                'Nome': f'Gate D {family} {key} latency', 'Metrica': [metric],
                'Versione': 'v1', 'Ruolo': 'Secondaria', 'Target': 'diagnostic',
                'Aggregazione e pesi': 'Same input, float32, Inductor default/fullgraph, T4; do not add component latencies due to fusion.'})
            observations.append(put('observations', family + '-' + key, **{
                'Nome': f'Gate D {family} {key} latency',
                'Valore': row['timing'][key]['median_ms'], 'Run': [run],
                'Specifica di valutazione': [evaluation], 'Artefatti dettagliati': [artifact],
                'Strato o sottogruppo': family + '/batch-642',
                'Descrizione': 'Full sample distribution and kernel profiler preserved in final_report.json.'}))
        put('findings', family + '-core-bottleneck', **{
            'Nome': f'Gate D {family}: learned five-channel core is slower',
            'Esito': 'Negativo', 'Osservazioni': observations,
            'Risultato': f"Core exact {row['timing']['exact_core_five']['median_ms']:.4f} ms vs neural {row['timing']['neural_core_five']['median_ms']:.4f} ms; full exact {row['timing']['exact_full']['median_ms']:.4f} vs hybrid {row['timing']['hybrid_full']['median_ms']:.4f} ms.",
            'Limitazioni': 'Profiler names support extra GEMM launches; precise hardware attribution remains an inference. Single T4, batch 642; diagnostic only.'})
    put('decisions', 'keep-no-go', **{
        'Nome': 'Keep Gate D NO-GO after bottleneck localization', 'Esito': 'Modificare',
        'Motivazione': 'The learned five-channel core itself is slower than five compiled formulas in both families. Recomposition and current head do not reverse the gap.',
        'Condizioni di revisione': 'Any revised primitive must first beat a symmetric compiled formula control at representative scale and satisfy accuracy gates; Task29 remains unauthorized.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'rows': len(report['rows']), 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
