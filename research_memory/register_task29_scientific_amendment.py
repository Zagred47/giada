"""Separate scientific Task29 authorization from Gate D speed promotion."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/task29_scientific_track_amendment.json'
    protocol_path = repo / 'experiments/task29_external_clamp_ionic_passive.json'
    amendment = json.loads(path.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    assert amendment['task29_diagnostic_authorized']
    assert not amendment['task29_performance_promotion_authorized']
    assert not amendment['task30_autonomous_voltage_authorized']
    assert amendment['gate_d_performance_status'] == 'NO_GO_UNCHANGED'
    mirror = ValidatedBatchMirror()
    def put(table, key, **fields):
        return mirror.local_upsert(table, {'Codice stabile': f'{table}-giada-task29-{key}-v1', **fields})['record_id']
    claim = put('claims', 'clamped-ionic-passive', **{
        'Nome': 'Frozen ionic block composes with passive current under external clamp',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'With voltage and calcium imposed, frozen ionic gates and analytic currents preserve current-balance accuracy when passive and capacitive terms are added.',
        'Condizioni di falsificazione': 'One or more preregistered paths, panels, families or seeds violate the gate/current-balance limits.'})
    experiment = put('experiments', 'clamped-passive', **{
        'Nome': 'GIADA Task29 external-clamp ionic-passive diagnostic',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': amendment['scope'],
        'Obiettivo informativo': 'Test ionic-passive current balance before any autonomous voltage update; preserve compiled Gate D speed NO-GO.'})
    put('decisions', 'split-science-performance', **{
        'Nome': 'Authorize Task29 diagnostic but not speed promotion',
        'Esito': 'Continuare',
        'Motivazione': amendment['reason'],
        'Condizioni di revisione': 'Task29 results can inform Task30 scientific scope only through a separate decision. Gate D performance requires a new fair speed confirmation.'})
    put('protocols', 'scope-amendment', **{
        'Nome': 'Task29 prospective scope amendment', 'Esperimento': [experiment],
        'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(amendment, ensure_ascii=False),
        'Criteri di successo': 'Scientific diagnostic only; no Gate D or Task30 promotion.',
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'No autonomous voltage or speedup claim.'})
    put('protocols', 'external-clamp', **{
        'Nome': 'Task29 frozen ionic and passive external-clamp current budget',
        'Esperimento': [experiment], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(protocol, ensure_ascii=False),
        'Criteri di successo': protocol['decision'],
        'Hash preregistrazione': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Technical failures do not count as scientific NO-GO; Task30 and Gate D promotion remain unauthorized.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task29_diagnostic_authorized': True, 'gate_d_performance_status': 'NO_GO_UNCHANGED', 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
