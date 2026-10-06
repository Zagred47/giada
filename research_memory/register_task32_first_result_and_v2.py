"""Preserve Task32 v1 technical floor failure and preregister v2 locally."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task32_kaggle_f8c5d25'
    report = json.loads((folder / 'final_report.json').read_text())
    audit = json.loads((folder / 'result_audit.json').read_text())
    cfg_path = repo / 'experiments/task32_dynamic_calcium_feedback_v2.json'
    cfg = json.loads(cfg_path.read_text())
    assert report['valid'] and not report['native_floor_admissible'] and not report['models']
    assert audit['valid'] and audit['models_not_judged']
    assert audit['local_one_step_audit']['native_ica_vs_fixed_eca120_rmse_ma_cm2'] > 1e-4
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task32-v1-v2-{key}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'v1-archive', **{
        'Nome': 'Task32 v1 native floor failure artifact', 'Tipo': 'Report',
        'Percorso': (folder / 'artifact_bundle.zip').relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256((folder / 'artifact_bundle.zip').read_bytes()).hexdigest(),
        'Dimensione byte': (folder / 'artifact_bundle.zip').stat().st_size,
        'Versione': 'v1'})
    run = put('runs', 'v1', **{
        'Nome': 'Task32 v1 native coupled-calcium floor (model not judged)',
        'Stato': 'Completata', 'Seed': 'deterministic 16 native episodes',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Execution and ZIP valid; scientific model comparison inadmissible because native eca auto-recomputed against fixed-eca formula contract.'})
    metric = put('metrics', 'floor-40ms', **{
        'Nome': 'Task32 formula/native 40ms voltage floor', 'Famiglia': 'Regressione',
        'Direzione': 'Minimizzare', 'Unità': 'mV',
        'Formula': 'Pooled RMSE of exact-formula versus native NEURON voltage on disjoint confirmation episodes.'})
    evaluation = put('evaluations', 'floor-40ms', **{
        'Nome': 'Task32 v1 confirmation native floor', 'Metrica': [metric],
        'Versione': 'v1', 'Ruolo': 'Primaria', 'Target': '<=0.2mV pooled; <=0.5mV worst episode',
        'Aggregazione e pesi': 'Eight confirmation episodes, 40ms.'})
    observation = put('observations', 'floor-40ms', **{
        'Nome': 'Task32 v1 formula/native floor inadmissible',
        'Valore': report['confirmation_formula_floor']['pooled_voltage_rmse_mv'],
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': '6.3955mV pooled / 13.9331mV worst versus 0.2/0.5mV limits; no frozen model evaluation.'})
    finding = put('findings', 'eca-auto', **{
        'Nome': 'Native calcium reversal auto-recomputed when CaDynamics writes cai',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': 'The v1 reference assumed fixed 120mV eca, but NEURON ion style recomputed eca with cai; both source timing hypotheses fail the confirmation floor. Model checkpoints were not judged.',
        'Limitazioni': 'Inferred from native ica, V and gate conductances; v2 directly records eca. V1 retains its registered failure.'})
    claim = put('claims', 'fixed-eca-v2', **{
        'Nome': 'Fixed-eca native control supports independent dynamic-calcium model judgment',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta', 'Enunciato': cfg['purpose'],
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    experiment = put('experiments', 'v2', **{
        'Nome': 'GIADA Task32 fixed-eca dynamic-calcium confirmation v2',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Restore Task30c fixed-calcium-reversal control while adding dynamic cai and SK; measure eca every step and reopen frozen model judgment only if native floor passes.'})
    put('protocols', 'v2', **{
        'Nome': 'Task32 v2 fixed-eca native floor before frozen-model verdict',
        'Esperimento': [experiment], 'Versione': 'v2', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'If eca drifts or formula/native floor fails, do not judge frozen rates. Preserve diagnostic artifacts and v1 result unchanged.'})
    put('decisions', 'retry-v2', **{
        'Nome': 'Task32 v1 technical floor requires fixed-eca v2',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'Violation of the fixed-reversal source contract, not a measured model failure.',
        'Condizioni di revisione': 'V2 native eca and formula floor pass unchanged limits; then inspect frozen family/seed results.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v1_model_judged': False,
                      'v2_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
