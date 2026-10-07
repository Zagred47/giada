"""Register the prospective passive RC/current/axial gate in local SQLite."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/iv_ad1_passive_axial_preregistration.json'
    spec = json.loads(path.read_text(encoding='utf-8'))
    assert spec['roadmap_ids'] == ['IV-A1', 'IV-A2', 'IV-D1']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-ad1-{suffix}-v1', **fields
        })['record_id']

    claim = put('claims', 'axial-interface', **{
        'Nome': 'Passive two-compartment axial interface matches independent RC/matrix solver and NEURON',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'Geometry-derived center-to-center axial conductance, capacitive/leak current balance and semi-implicit two-node voltage solve agree with NEURON across connected, disconnected and symmetric controls; a passing result is prerequisite to Task36.',
        'Limiti': 'Passive only; no ion channels, calcium, synapses, neural candidate, full cell or speedup.',
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'passive-preflight', **{
        'Nome': 'IV-A1/A2 then IV-D1 passive axial preflight',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['limits'],
        'Obiettivo informativo': 'Verify unit/sign/current interface and geometry-derived two-compartment axial solve before active Task36.'})
    put('protocols', 'v1', **{
        'Nome': 'IV-A1/A2/D1 sequential passive gate',
        'Esperimento': [experiment], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'A1/A2 must pass before D1 opens; D1 must pass before Task36. Never convert a passive failure into an active-model NO-GO.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_ad1_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
