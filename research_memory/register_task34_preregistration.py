"""Register Task34 prospective privileged-probe protocol in the SQLite mirror."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/task34_privileged_current_state_probes_preregistration.json'
    spec = json.loads(path.read_text(encoding='utf-8'))
    for field, digest in (('parent_report', 'parent_report_sha256'),
                          ('parent_traces', 'parent_traces_sha256')):
        assert hashlib.sha256((repo / spec[field]).read_bytes()).hexdigest() == spec[digest]
    parent = json.loads((repo / spec['parent_report']).read_text(encoding='utf-8'))
    assert parent['task33_passed'] and parent['code_revision'] == spec['parent_code_revision']
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-task34-{suffix}-v1', **fields
        })['record_id']

    claim = put('claims', 'error-attribution', **{
        'Nome': 'Privileged probes separate state, voltage and interaction errors',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'Matched teacher/model voltage and gate-state interventions, together with an exact per-channel current factorial, can identify whether the residual Task33 errors primarily arise in gate kinetics, voltage feedback or their interaction.',
        'Limiti': 'Existing 16 Task33 cases only; teacher-privileged arms are not deployable and no independent performance claim follows.',
        'Condizioni di falsificazione': spec['decision']})
    experiment = put('experiments', 'diagnostic', **{
        'Nome': 'Task34 privileged current and state probes',
        'Tipo': 'Esplorativo', 'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': spec['domain'],
        'Obiettivo informativo': 'Rank 11 channel-current and 18 gate-state errors and distinguish state, voltage, feedback and interaction without retraining or selecting checkpoints.'})
    put('protocols', 'v1', **{
        'Nome': 'Task34 matched causal probe matrix', 'Esperimento': [experiment],
        'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(spec, ensure_ascii=False),
        'Criteri di successo': spec['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Failure to reproduce Task33 or current identity is technical, not a model NO-GO. Oracle arms never select a model.'})
    put('decisions', 'launch', **{
        'Nome': 'Task33 confirmed; launch diagnostic Task34 before Task35',
        'Esito': 'Continuare',
        'Motivazione': 'Measure internal state/current attribution before changing the full-compartment architecture. Reuse Task33 cases diagnostically, not as a fresh test.',
        'Condizioni di revisione': 'Task34 report must reproduce Task33 baseline and preserve all oracle/exposure labels.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'task34_preregistered': True,
                      'airtable_accessed': False}))


if __name__ == '__main__':
    main()
