"""Preregister Task30c's native-active floor and frozen comparison in SQLite."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    cfg_path = repo / 'experiments/task30c_native_active_confirmation.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    from src.giada_teacher.task30c_native_active_confirmation import verify
    teacher = repo.parent / 'neuron_as_deep_net'
    verify(repo, teacher, cfg)
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task30c-{key}-v1', **fields
        })['record_id']

    claim = put('claims', 'native-active-retention', **{
        'Nome': 'Task30b frozen gates retain voltage under native eleven-channel dynamics',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta', 'Enunciato': cfg['purpose'],
        'Condizioni di falsificazione': cfg['decision'], 'Limiti': cfg['scope']})
    experiment = put('experiments', 'native-active-confirmation', **{
        'Nome': 'GIADA Task30c native active confirmation', 'Tipo': 'Confermativo',
        'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': cfg['purpose'],
        'Obiettivo informativo': 'Separate exact-formula-to-NEURON active solver floor from learned-gate voltage error on the same frozen Task30b design.'})
    put('protocols', 'native-floor-first', **{
        'Nome': 'Task30c eleven-channel native floor before candidate verdict',
        'Esperimento': [experiment], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'If native formula floor fails, no learned-model NO-GO and no Task31 promotion. Preserve diagnostic artifact.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task30c_preregistered': True,
                      'task31_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
