"""Preregister IV-C1 locally; no Airtable calls."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/iv_c1_deterministic_synapse_preregistration.json'
    cfg = json.loads(path.read_text())
    assert cfg['roadmap_id'] == 'IV-C1'
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c1-{key}-v1', **fields
        })['record_id']

    claim = put('claims', 'receptor-kernel', **{
        'Nome': 'Canonical synaptic receptor kernels are sufficient under observed releases',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'Closed-form dual-exponential receptor states with the teacher Mg block reproduce canonical AMPA/NMDA/GABA point processes when every release event is observed.',
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['promotion']})
    experiment = put('experiments', 'deterministic-kernel', **{
        'Nome': 'GIADA IV-C1 deterministic synaptic receptor audit',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato',
        'Ipotesi': [claim], 'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Separate receptor kernel/timing from release probability, plasticity and RNG before Task33.'})
    put('protocols', 'native-formula', **{
        'Nome': 'IV-C1 native/formula paired receptor comparison',
        'Esperimento': [experiment], 'Versione': 'v1',
        'Modalità': 'Componente isolato',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['promotion'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Calibrate phase only on single-event schedule; if calibration fails, do not judge held-out schedules. IV-C1 does not authorize Task33 without IV-C2/C3.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'roadmap_id': 'IV-C1',
                      'task33_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
