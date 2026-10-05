"""Register the prospective IV-B v2 confirmation in the local mirror only."""

import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/iv_b_v2_kaggle_8b3a4e5'
    audit = json.loads((folder / 'result_audit.json').read_text())
    report = json.loads((folder / 'final_report.json').read_text())
    assert audit['valid'] and report['valid']
    assert audit['iv_b1_valid'] and audit['iv_b2_valid']
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-b-v2-result-{key}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'archive', **{
        'Nome': 'IV-B v2 prospective native confirmation archive', 'Tipo': 'Report',
        'Percorso': (folder / 'artifact_bundle.zip').relative_to(repo).as_posix(),
        'SHA-256': audit['archive_sha256'],
        'Dimensione byte': (folder / 'artifact_bundle.zip').stat().st_size,
        'Versione': 'v2'})
    run = put('runs', 'kaggle', **{
        'Nome': 'IV-B1/B2 updated-cai prospective Kaggle confirmation',
        'Stato': 'Completata', 'Seed': 'deterministic 72 B1 + 12 B2 episodes',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle CPU, NEURON 8.2.7, source and code revision pinned; ZIP CRC, source report and metrics independently audited.'})
    metric = put('metrics', 'b2-gate-rmse', **{
        'Nome': 'IV-B2 composed SK occupancy RMSE', 'Famiglia': 'Regressione',
        'Direzione': 'Minimizzare', 'Unità': 'gate occupancy',
        'Formula': 'Worst episode RMSE of native SK z versus CaDynamics-to-SK causal analytic composition.'})
    evaluation = put('evaluations', 'b2-gate', **{
        'Nome': 'IV-B2 v2 prospective gate confirmation', 'Metrica': [metric],
        'Versione': 'v2', 'Ruolo': 'Primaria', 'Target': 'RMSE <= 1e-4',
        'Aggregazione e pesi': 'Maximum of 12 episodes; all 1601 steps each.'})
    observation = put('observations', 'b2-gate', **{
        'Nome': 'IV-B2 v2 worst composed SK gate RMSE',
        'Valore': audit['b2_worst_composed_sk_gate_rmse'],
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Strato o sottogruppo': 'prospective updated-cai IV-B2',
        'Descrizione': '72/72 B1 and 12/12 B2 pass unchanged v1 numerical limits; new late_single and triplet schedules pass; pulse/rest controls informative.'})
    finding = put('findings', 'pass', **{
        'Nome': 'IV-B1/B2 causal calcium-to-SK timing confirmed prospectively',
        'Esito': 'Positivo', 'Osservazioni': [observation],
        'Risultato': 'B1 worst cai RMSE 4.80e-14 mM; B2 worst composed SK gate RMSE 2.20e-11; current RMSE 4.40e-16 mA/cm2; imposed-voltage clamp error 1.57e-5 mV. Task32 feedback prerequisite passed.',
        'Limitazioni': 'One compartment, imposed ica and imposed V. No autonomous electrical feedback, morphology, synapses, full-cell validity, training or speedup claim.'})
    put('decisions', 'task32-authorized', **{
        'Nome': 'IV-B2 gate authorizes preparation of Task32 electrical feedback',
        'Esito': 'Continuare', 'Risultati': [finding],
        'Motivazione': 'Independent prospective v2 run passes registered B1/B2 limits and controls under native NEURON, with archived raw trajectories.',
        'Condizioni di revisione': 'Task32 must separately validate autonomous electrical feedback; v2 result is not a Task32 outcome.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task32_feedback_prerequisite_passed': True,
                      'task32_completed': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
