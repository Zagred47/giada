"""Preregister IV-C2 in the local SQLite mirror; never accesses Airtable."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/iv_c2_stochastic_release_preregistration.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c2-{suffix}-v1', **fields
        })['record_id']

    hypothesis = put('claims', 'event-rng-shadow', **{
        'Nome': 'Explicit plasticity and Random123 negexp shadow reproduces native EMS release',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'The public NET_RECEIVE plastic-state equations plus an independently seeded negexp Random123 stream predict the event-by-event native release and RNG draw order.',
        'Limiti': cfg['scope'],
        'Condizioni di falsificazione': cfg['decision']})
    experiment = put('experiments', 'prospective', **{
        'Nome': 'IV-C2 stochastic EMS release and plasticity',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato',
        'Ipotesi': [hypothesis], 'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Separate scheduled from realized events and test Use, Fac, Dep, RNG order, restart, and release distribution with paired streams.'})
    put('protocols', 'prospective', **{
        'Nome': 'IV-C2 paired factorial release protocol',
        'Esperimento': [experiment], 'Versione': 'v1',
        'Modalità': 'Componente isolato',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Runtime/API failure is inconclusive; no IV-C3 or Task33 promotion.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'preregistered': True,
                      'airtable_accessed': False, 'task33_authorized': False}))


if __name__ == '__main__':
    main()
