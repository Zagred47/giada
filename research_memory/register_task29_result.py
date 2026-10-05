"""Record the narrow Task29 scientific result in SQLite only."""
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task29_kaggle_1245b70'
    report = json.loads((folder / 'final_report.json').read_text(encoding='utf-8'))
    audit = json.loads((folder / 'result_audit.json').read_text(encoding='utf-8'))
    assert report['valid'] and report['diagnostic_scientific_passed'] and audit['valid']
    assert not report['gate_d_completed'] and not report['task30_autonomous_voltage_authorized']
    mirror = ValidatedBatchMirror()
    def put(table, key, **fields):
        return mirror.local_upsert(table, {'Codice stabile': f'{table}-giada-task29-{key}-v1', **fields})['record_id']
    artifact = put('artifacts', 'result-bundle', **{
        'Nome': 'Task29 Kaggle result bundle', 'Tipo': 'Report',
        'Percorso': (folder / 'artifact_bundle.zip').relative_to(repo).as_posix(),
        'SHA-256': audit['artifact_sha256'],
        'Dimensione byte': (folder / 'artifact_bundle.zip').stat().st_size, 'Versione': 'v1'})
    findings = []
    for family in ('independent', 'shared_heads'):
        summary = audit['summary'][family + '/both']
        run = put('runs', 'both-' + family, **{
            'Nome': f'Task29 external clamp {family}/both', 'Stato': 'Completata',
            'Seed': '17,29,43', 'Artefatti prodotti': [artifact],
            'Validità tecnica': 'Kaggle COMPLETE/exit0, NEURON passive oracle, frozen checkpoint provenance, ZIP CRC, all row keys and thresholds independently audited.'})
        observations = []
        for key in ('worst_gate_rmse', 'worst_individual_current_normalized_rmse',
                    'worst_clamp_demand_normalized_rmse'):
            metric = put('metrics', family + '-' + key, **{
                'Nome': f'Task29 {family} {key}', 'Famiglia': 'Regressione',
                'Direzione': 'Minimizzare', 'Unità': 'adimensionale',
                'Formula': 'Maximum across preregistered external-clamp paths, panels and frozen seeds; exact definition in Task29 config.'})
            evaluation = put('evaluations', family + '-' + key, **{
                'Nome': f'Task29 {family} {key}', 'Metrica': [metric],
                'Versione': 'v1', 'Ruolo': 'Primaria',
                'Target': '0.005' if key == 'worst_gate_rmse' else '0.01',
                'Aggregazione e pesi': 'Maximum; no averaging across seeds or panels to hide failure.'})
            observations.append(put('observations', family + '-' + key, **{
                'Nome': f'Task29 {family} {key}', 'Valore': summary[key],
                'Run': [run], 'Specifica di valutazione': [evaluation],
                'Artefatti dettagliati': [artifact],
                'Strato o sottogruppo': family + '/both',
                'Descrizione': f"{summary['passed']}/{summary['total']} registered rows passed; full rows in final_report.json."}))
        findings.append(put('findings', 'clamped-passive-' + family, **{
            'Nome': f'Task29 {family}: external-clamp ionic-passive diagnostic succeeds',
            'Esito': 'Positivo', 'Osservazioni': observations,
            'Risultato': f"{summary['passed']}/{summary['total']} both-arm rows passed; native passive audit error 0; zero occupancy violations.",
            'Limitazioni': 'V and cai imposed. Clamp balance itself is algebraic; only hybrid-vs-formula difference is informative. No autonomous voltage, authentic full native clamp rollout, or speedup.'}))
    put('decisions', 'scientific-go-scope-limited', **{
        'Nome': 'Task29 diagnostic GO without Gate D or Task30 promotion',
        'Esito': 'Continuare', 'Risultati': findings,
        'Motivazione': 'All preregistered both-arm path, panel, seed and native-passive checks pass under imposed V and cai.',
        'Condizioni di revisione': 'A separate preregistered Task30 autonomous-voltage decision is required. Compiled Gate D performance NO-GO remains unchanged.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'both_arm_rows': 120, 'task30_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
