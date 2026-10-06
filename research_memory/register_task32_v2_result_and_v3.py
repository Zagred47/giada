"""Preserve v2 floor failure and preregister phase-aligned v3 locally."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    folder = repo / 'experiments/results/task32_v2_kaggle_e22b669'
    report = json.loads((folder / 'final_report.json').read_text())
    audit = json.loads((folder / 'result_audit.json').read_text())
    cfg_path = repo / 'experiments/task32_dynamic_calcium_feedback_v3.json'
    from src.giada_teacher.task32_dynamic_calcium_feedback import load_v3_contract
    cfg = load_v3_contract(repo)
    assert report['valid'] and not report['native_floor_admissible'] and not report['models']
    assert audit['valid'] and audit['model_not_judged'] and audit['native_eca_max_deviation_mv'] == 0
    assert audit['one_step_gate_voltage_alignment']['new_voltage']['max_abs'] < 1e-12
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task32-v2-v3-{key}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'v2-archive', **{
        'Nome': 'Task32 v2 fixed-eca but phase-misaligned calcium artifact', 'Tipo': 'Report',
        'Percorso': (folder / 'artifact_bundle.zip').relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256((folder / 'artifact_bundle.zip').read_bytes()).hexdigest(),
        'Dimensione byte': (folder / 'artifact_bundle.zip').stat().st_size,
        'Versione': 'v2'})
    run = put('runs', 'v2', **{
        'Nome': 'Task32 v2 fixed-eca native floor, model not judged',
        'Stato': 'Completata', 'Seed': 'deterministic 16 episodes',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Execution and ZIP valid, eca fixed exactly; calcium floor fails due formula/native state-phase mismatch; no frozen-model verdict.'})
    metric = put('metrics', 'cai-floor', **{
        'Nome': 'Task32 v2 formula/native calcium floor', 'Famiglia': 'Regressione',
        'Direzione': 'Minimizzare', 'Unità': 'mM',
        'Formula': 'Worst confirmation-episode calcium concentration RMSE to native.'})
    evaluation = put('evaluations', 'cai-floor', **{
        'Nome': 'Task32 v2 confirmation calcium floor', 'Metrica': [metric],
        'Versione': 'v2', 'Ruolo': 'Primaria', 'Target': '<=1e-6mM',
        'Aggregazione e pesi': 'Worst of eight confirmation episodes at 40ms.'})
    observation = put('observations', 'cai-floor', **{
        'Nome': 'Task32 v2 calcium floor missed',
        'Valore': report['confirmation_formula_floor']['worst_episode_cai_rmse_mM'],
        'Run': [run], 'Specifica di valutazione': [evaluation],
        'Artefatti dettagliati': [artifact],
        'Descrizione': '2.89e-5mM versus 1e-6mM; eca deviation zero; voltage floor passed; model not judged.'})
    finding = put('findings', 'phase-alignment', **{
        'Nome': 'Task32 formula/native gate states have different phase conventions',
        'Esito': 'Misto', 'Osservazioni': [observation],
        'Risultato': 'Native gate[t+1] follows V[t+1] to ~1.55e-15 max error; formula next-gate current is the causal state aligned with the voltage update. Updated-gate formula source yields post-hoc confirmation calcium RMSE 4.04e-8mM.',
        'Limitazioni': 'Post-hoc counterfactual; v2 not promoted. V3 prospectively freezes this effective discrete interface before judging models.'})
    claim = put('claims', 'v3-phase', **{
        'Nome': 'Phase-aligned formula current permits autonomous Ca-SK model evaluation',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta', 'Enunciato': cfg['purpose'],
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    experiment = put('experiments', 'v3', **{
        'Nome': 'GIADA Task32 prospective phase-aligned v3',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Confirm the effective causal source phase with fixed eca, then judge frozen families only behind native floor.'})
    put('protocols', 'v3', **{
        'Nome': 'Task32 v3 frozen phase-aligned Ca-SK feedback',
        'Esperimento': [experiment], 'Versione': 'v3', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'If fixed-eca/native formula floor fails, stop before frozen-model verdict. V1/v2 results remain unchanged.'})
    put('decisions', 'v3', **{
        'Nome': 'Task32 v2 floor requires phase-aligned v3',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'Correct fixed eca removed voltage mismatch; phase alignment of formula calcium-current source is still needed.',
        'Condizioni di revisione': 'Prospective v3 native voltage/calcium floor and all frozen model family/seed gates.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'v2_model_judged': False,
                      'v3_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
