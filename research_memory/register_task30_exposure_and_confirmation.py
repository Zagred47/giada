"""Record the Task30 exposure defect and preregister Task30b in SQLite."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    first = repo / 'experiments/results/task30_kaggle_fb08789'
    audit = json.loads((first / 'result_audit.json').read_text(encoding='utf-8'))
    cfg_path = repo / 'experiments/task30b_active_exposure_confirmation.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    from src.giada_teacher.task30_autonomous_voltage import verify_parent, episodes
    verify_parent(repo, cfg)
    design = episodes(cfg)
    k = int(round(cfg['primary_horizon_ms'] / cfg['dt_ms']))
    assert audit['technical_valid'] and not audit['decision_grade']
    assert audit['active_injection_samples_before_primary_horizon'] == 0
    assert int((design['injection_ma_cm2'][:k] != 0).sum()) > 0
    mirror = ValidatedBatchMirror()
    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task30b-{key}-v1', **fields})['record_id']
    artifact = put('artifacts', 'first-audit', **{
        'Nome': 'Task30 first-run exposure audit', 'Tipo': 'Report',
        'Percorso': (first / 'result_audit.json').relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256((first / 'result_audit.json').read_bytes()).hexdigest(),
        'Dimensione byte': (first / 'result_audit.json').stat().st_size, 'Versione': 'v1'})
    run = put('runs', 'first-nondecision', **{
        'Nome': 'Task30 first technically valid but non-decision-grade run',
        'Stato': 'Completata', 'Seed': '17,29,43',
        'Artefatti prodotti': [artifact],
        'Validità tecnica': 'Kaggle COMPLETE/exit0 and native passive calibration valid, but no injected-current sample before the registered 8ms primary horizon.'})
    finding = put('findings', 'zero-primary-exposure', **{
        'Nome': 'Task30 primary horizon was entirely pre-stimulus',
        'Esito': 'Misto', 'Risultato': 'Reported primary numerical pass is non-decision-grade: zero active samples before 8ms in every episode. Later 20–80ms results are diagnostic only.',
        'Limitazioni': 'This is a protocol-design failure, not evidence that frozen ionic dynamics failed. Preserve the original result and do not promote Task31.'})
    claim = put('claims', 'active-short-retention', **{
        'Nome': 'Frozen ionic gate dynamics retain voltage after early active stimuli',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'On disjoint early-stimulated single-compartment cases, each frozen both-arm family and seed stays within the unchanged 8ms voltage RMSE limits.',
        'Condizioni di falsificazione': cfg['decision'], 'Limiti': cfg['limits']})
    experiment = put('experiments', 'early-stimulus-confirmation', **{
        'Nome': 'GIADA Task30b active-exposure confirmation', 'Tipo': 'Confermativo',
        'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': cfg['reason'],
        'Obiettivo informativo': 'Repair absent primary stimulus exposure without changing the frozen checkpoints or original numerical threshold.'})
    put('protocols', 'early-exposure', **{
        'Nome': 'Task30b fresh early-stimulus causal matrix',
        'Esperimento': [experiment], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Require active injection before 8ms. Technical failure or zero exposure cannot be interpreted as model NO-GO.'})
    put('decisions', 'confirm-before-task31', **{
        'Nome': 'Do not promote Task31 from unexposed Task30 primary',
        'Esito': 'Modificare', 'Risultati': [finding],
        'Motivazione': 'The nominal first-run primary pass preceded every stimulus. A fresh early-exposure confirmation is required.',
        'Condizioni di revisione': 'Audit Task30b support, model metrics, active reference crossings and native passive calibration separately; Gate D performance remains NO-GO.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'first_run_decision_grade': False,
                      'task30b_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
