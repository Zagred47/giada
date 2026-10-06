"""Preregister autonomous calcium feedback in the local SQLite mirror only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/task32_dynamic_calcium_feedback_preregistration.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    for parent, digest in [('parent_task30c_report', 'parent_task30c_sha256'),
                           ('parent_iv_b_report', 'parent_iv_b_sha256')]:
        assert hashlib.sha256((repo / cfg[parent]).read_bytes()).hexdigest() == cfg[digest]
    assert cfg['calibration_protocols'] == ['rest', 'early_low']
    assert cfg['confirmation_protocols'] == ['early_high', 'early_paired']
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task32-{key}-v1', **fields
        })['record_id']

    claim = put('claims', 'slow-feedback', **{
        'Nome': 'Frozen ionic rates preserve autonomous voltage with dynamic calcium-SK feedback',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta', 'Enunciato': cfg['purpose'],
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    experiment = put('experiments', 'dynamic-calcium', **{
        'Nome': 'GIADA Task32 dynamic calcium feedback', 'Tipo': 'Confermativo',
        'Stato': 'Preregistrato', 'Ipotesi': [claim], 'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Disentangle within-step calcium-current timing, native formula floor, learned-gate closed-loop error and frozen-calcium counterfactual on paired episodes.'})
    put('protocols', 'native-floor-first', **{
        'Nome': 'Task32 native dynamic-calcium floor and frozen-family comparison',
        'Esperimento': [experiment], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Calibration protocols choose the source scheme; confirmation protocols never choose it. Native floor failure blocks learned-model judgment. Preserve complete diagnostics.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task32_preregistered': True,
                      'result_claimed': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
