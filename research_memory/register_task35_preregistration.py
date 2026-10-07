"""Register the prospective Task35 local-compartment test in SQLite only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/task35_full_local_compartment_preregistration.json'
    spec = json.loads(path.read_text(encoding='utf-8'))
    for source in ('parent_task33', 'parent_task34'):
        assert hashlib.sha256((repo / spec[f'{source}_report']).read_bytes()).hexdigest() == spec[f'{source}_sha256']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task35-{suffix}-v1', **fields
        })['record_id']

    claim = put('claims', 'local-panel-generalization', **{
        'Nome': 'Frozen local-compartment surrogate transfers across independent conductance panels',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'All six frozen Task32 candidates retain valid 40-ms autonomous V/Ca prediction in the declared full local mechanism inventory when calcium, sodium or potassium conductances are separately amplified fourfold, with independently replayed stochastic synapses.',
        'Limiti': 'Single cylindrical compartment, fully observed presynaptic schedule/RNG; no axial coupling, morphology, whole cell or speedup claim.',
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'independent-confirmation', **{
        'Nome': 'Task35 full declared local compartment, independent panel matrix',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['scope'],
        'Obiettivo informativo': 'Separate canonical-domain success from calcium/sodium/potassium conductance-panel generalization; test each family and seed without model selection.'})
    put('protocols', 'v1', **{
        'Nome': 'Task35 native-floor and six-checkpoint panel-wise gate',
        'Esperimento': [experiment], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Technical or formula/native floor failure blocks model judgment. With an admissible floor, a failed frozen candidate is a scientific NO-GO; thresholds and checkpoints stay fixed.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task35_preregistered': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
