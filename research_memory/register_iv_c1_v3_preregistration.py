"""Record prospective IV-C1 v3 before its native run, SQLite only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    from src.giada_teacher.iv_c1_deterministic_synapse import load_v3_contract
    cfg = load_v3_contract(repo)
    spec = repo / 'experiments/iv_c1_deterministic_synapse_v3.json'
    forensic = repo / 'experiments/results/iv_c1_forensic_kaggle_b5550b6/state_timing_forensic.json'
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c1-v3-{suffix}-v1', **fields
        })['record_id']

    artifact = put('artifacts', 'forensic', **{
        'Nome': 'IV-C1 v2 per-event solver-clock forensic trace', 'Tipo': 'Report',
        'Percorso': forensic.relative_to(repo).as_posix(),
        'SHA-256': hashlib.sha256(forensic.read_bytes()).hexdigest(),
        'Dimensione byte': forensic.stat().st_size, 'Versione': 'forensic-v1'})
    claim = put('claims', 'solver-clock-hypothesis', **{
        'Nome': 'Actual solver clock reconciles deterministic receptor states',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'The event jump is visible iff public solver h.t has reached its NetCon timestamp; nominal n*dt can differ near the boundary.',
        'Limiti': cfg['scope'],
        'Condizioni di falsificazione': cfg['decision']})
    experiment = put('experiments', 'prospective', **{
        'Nome': 'IV-C1 v3 actual-clock prospective confirmation',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato',
        'Ipotesi': [claim], 'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Test actual public solver-clock event semantics on disjoint schedules; preserve state/current/charge gates and negative controls.'})
    put('protocols', 'prospective', **{
        'Nome': 'IV-C1 v3 solver-clock schedules', 'Esperimento': [experiment],
        'Versione': 'v3', 'Modalità': 'Componente isolato',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(spec.read_bytes()).hexdigest(),
        'Regole di arresto': 'Calibration before 12-cell confirmation. Diagnostic v1/v2/forensic schedules cannot be reused for promotion; no Task33 claim.'})
    put('decisions', 'prospective', **{
        'Nome': 'Advance IV-C1 to disjoint actual-clock confirmation',
        'Esito': 'Modificare',
        'Motivazione': 'Forensic trace links observed event-side A/B state to the sign of h.t minus scheduled time for both excitatory and inhibitory events.',
        'Condizioni di revisione': 'All fixed v3 gates on unopened schedules; then IV-C2 and IV-C3 still required.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'preregistered': True,
                      'forensic_artifact_recorded': bool(artifact),
                      'task33_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
