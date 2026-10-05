"""Preregister IV-B1/B2, without prematurely authorizing Task32 feedback."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/iv_b1_b2_calcium_prerequisite.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    assert cfg['roadmap_prerequisites'] == ['IV-B1', 'IV-B2']
    mirror = ValidatedBatchMirror()

    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-b-{key}-v1', **fields
        })['record_id']

    claim_b1 = put('claims', 'calcium-identity', **{
        'Nome': 'CaDynamics_E2 concentration dynamics match an independently coded exact step',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'Native CaDynamics_E2 and the exact constant-current ODE update agree across initial calcium, decay, gamma/depth and pulse schedules.',
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    claim_b2 = put('claims', 'sk-composition', **{
        'Nome': 'CaDynamics-to-SK composition preserves gate and current timing',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'At imposed voltage, native calcium, SK gate and SK current agree with a causal composed exact-step reference after IV-B1 passes.',
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    exp = put('experiments', 'calcium-prerequisite', **{
        'Nome': 'GIADA IV-B1/B2 calcium prerequisite matrix',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato',
        'Ipotesi': [claim_b1, claim_b2], 'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Separate calcium concentration solver/units from SK composition/timing before closing the electrical feedback loop in Task32.'})
    put('protocols', 'paired-native-matrix', **{
        'Nome': 'IV-B1/B2 native CaDynamics and SK matrix', 'Esperimento': [exp],
        'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'If IV-B1 fails, do not interpret IV-B2; if either fails, do not run Task32 feedback. Retain technical failure artifacts.'})
    put('decisions', 'withhold-feedback', **{
        'Nome': 'Task32 calcium feedback awaits IV-B1/B2', 'Esito': 'Modificare',
        'Motivazione': 'Original Section IV plan explicitly requires IV-B2 before Task32; Task20 validated SK only under imposed calcium.',
        'Condizioni di revisione': 'Both independent native prerequisites pass their registered thresholds and perturbation checks; no automatic claim about multicompartment or speed.'})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_b1_b2_preregistered': True,
                      'task32_feedback_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
