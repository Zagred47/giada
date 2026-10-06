"""Preregister Task33 in the local SQLite mirror, without Airtable access."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/task33_observable_synaptic_feedback_preregistration.json'
    cfg = json.loads(path.read_text())
    for key, sha_key, passed_key in (
        ('parent_task32_report', 'parent_task32_sha256', 'scientific_primary_passed'),
        ('parent_iv_c3_report', 'parent_iv_c3_sha256', 'iv_c3_passed')):
        parent = repo / cfg[key]
        assert hashlib.sha256(parent.read_bytes()).hexdigest() == cfg[sha_key]
        assert json.loads(parent.read_text())[passed_key]
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task33-{suffix}-v1', **fields
        })['record_id']

    claim = put('claims', 'causal-composition', **{
        'Nome': 'Observable synaptic inputs suffice for active one-compartment feedback',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'Canonical EMS state plus prescribed events/Random123 produce causal synaptic current compatible with autonomous V/Ca and frozen Task32 ion-channel rates.',
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    experiment = put('experiments', 'prospective', **{
        'Nome': 'Task33 observable synaptic feedback', 'Tipo': 'Confermativo',
        'Stato': 'Preregistrato', 'Ipotesi': [claim], 'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Separate stochastic synaptic interface, native/formula floor and frozen neural-candidate behavior in 16 paired confirmation episodes.'})
    put('protocols', 'prospective', **{
        'Nome': 'Task33 active-compartment synaptic protocol',
        'Esperimento': [experiment], 'Versione': 'v1',
        'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': cfg['calibration_policy'] +
                            ' If native/formula floor fails, frozen models are not judged.'})
    put('decisions', 'prospective', **{
        'Nome': 'Run Task33 after IV-C3 and Task32 passes', 'Esito': 'Continuare',
        'Motivazione': 'Both parent reports are hash-pinned and passed in their own domains; integration under autonomous V remains untested.',
        'Condizioni di revisione': cfg['decision']})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task33_preregistered': True,
                      'task33_passed': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
