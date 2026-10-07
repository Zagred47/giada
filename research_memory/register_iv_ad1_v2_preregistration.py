"""Register the prospective v2 passive preflight, preserving the v1 NO-GO."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/iv_ad1_passive_axial_preregistration_v2.json'
    spec = json.loads(path.read_text(encoding='utf-8'))
    prior = repo / 'experiments/results/iv_ad1_v1_final_report.json'
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == spec['prior_v1_result_sha256']
    assert spec['roadmap_ids'] == ['IV-A1', 'IV-A2', 'IV-D1']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-ad1-{suffix}-v2', **fields
        })['record_id']

    claim = put('claims', 'separated-error-contract', **{
        'Nome': 'Passive native/discrete identity and discrete/continuous convergence are distinct tests',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'The v1 single-compartment NO-GO resulted from applying an absolute continuous-time endpoint threshold to a fixed 0.1 ms backward-Euler step, despite NEURON/discrete identity. New held-out passive cases should show unchanged native/discrete agreement and systematic dt convergence before the unopened axial matrix is tested.',
        'Limiti': spec['limits'],
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'passive-preflight', **{
        'Nome': 'IV-A1/A2 then IV-D1 passive axial preflight v2',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['limits'],
        'Obiettivo informativo': 'Separate time-discretization convergence from native solver identity, then test the unopened passive axial matrix.'})
    put('protocols', 'preregistration', **{
        'Nome': 'IV-A1/A2/D1 sequential passive gate v2',
        'Esperimento': [experiment], 'Versione': 'v2', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'V1 NO-GO remains; new A1/A2 cases must pass before D1; D1 before Task36.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_ad1_v2_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
