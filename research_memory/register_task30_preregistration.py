"""Preregister Task30 autonomous voltage in the SQLite mirror only."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/task30_autonomous_voltage_microcanary.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    from src.giada_teacher.task30_autonomous_voltage import verify_parent
    verify_parent(repo, cfg)
    mirror = ValidatedBatchMirror()
    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task30-{key}-v1', **fields})['record_id']
    claim = put('claims', 'short-autonomous-retention', **{
        'Nome': 'Frozen ionic gates remain stable under autonomous voltage feedback',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'At 8ms, both frozen both-arm families stay within 1mV pooled and 2mV worst-episode voltage RMSE against an exact-formula discrete single-compartment control.',
        'Limiti': cfg['limits'], 'Condizioni di falsificazione': cfg['decision']})
    exp = put('experiments', 'autonomous-single-compartment', **{
        'Nome': 'GIADA Task30 autonomous-voltage microcanary',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Separate direct gate-update error from feedback amplification across short and long horizons, while leaving Gate D speed unresolved.'})
    protocol = put('protocols', 'paired-causal-matrix', **{
        'Nome': 'Task30 paired formula, teacher-voltage and autonomous hybrid',
        'Esperimento': [exp], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Native passive calibration failure is technical; 8ms failure is scientific within the internal formula-map scope. No Task31 promotion from this test alone.'})
    put('decisions', 'await-autonomous-evidence', **{
        'Nome': 'Run Task30 diagnostic without speed promotion', 'Esito': 'Continuare',
        'Motivazione': 'Task29 under imposed voltage passed; voltage feedback and error amplification remain untested.',
        'Condizioni di revisione': 'Inspect full per-seed/episode/horizon causal matrix, passive calibration and independent audit; Gate D remains NO-GO.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task30_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
