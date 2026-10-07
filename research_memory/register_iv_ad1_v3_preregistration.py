"""Register the prospective v3 equal-current-density passive preflight."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/iv_ad1_passive_axial_preregistration_v3.json'
    spec = json.loads(path.read_text(encoding='utf-8'))
    prior = repo / 'experiments/results/iv_ad1_v2_final_report.json'
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == spec['prior_v2_result_sha256']
    assert spec['roadmap_ids'] == ['IV-A1', 'IV-A2', 'IV-D1']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-ad1-{suffix}-v3', **fields
        })['record_id']

    claim = put('claims', 'equal-density-control', **{
        'Nome': 'Equal density, not equal absolute current, is the asymmetric passive symmetry control',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'With identical specific passive parameters, equal initial V and equal injected current density imply equal V in compartments with different membrane area. All v2 native/discrete axial identities passed, but the equal-absolute-current negative control was physically invalid for unequal areas.',
        'Limiti': spec['limits'],
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'passive-preflight', **{
        'Nome': 'IV-A1/A2 then IV-D1 passive axial preflight v3',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['limits'],
        'Obiettivo informativo': 'Test the corrected symmetric negative control on new geometries without changing numerical tolerances.'})
    put('protocols', 'preregistration', **{
        'Nome': 'IV-A1/A2/D1 sequential passive gate v3',
        'Esperimento': [experiment], 'Versione': 'v3', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'V1/V2 NO-GO immutable; only full D1 pass authorizes Task36.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_ad1_v3_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
