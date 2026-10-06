"""Preregister IV-C3 in the local SQLite mirror only."""

import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    from src.giada_teacher.iv_c3_integrated_synaptic_interface import load_contract
    cfg = load_contract(repo)
    path = repo / 'experiments/iv_c3_integrated_synaptic_interface_preregistration.json'
    mirror = ValidatedBatchMirror()

    def put(table, suffix, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-iv-c3-{suffix}-v1', **fields
        })['record_id']

    causal = put('claims', 'causal-online-shadow', **{
        'Nome': 'Explicit per-synapse state/RNG is sufficient for mixed-event current prediction',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'A causal online shadow driven by prescribed events and independent matched RNG streams reproduces four canonical mixed EMS synapses without teacher-realized future release.',
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    restart = put('claims', 'native-restart', **{
        'Nome': 'Mixed synaptic state plus RNG restores the native suffix',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'The per-synapse receptor A/B, plastic variables, NetCon tsyn, public solver clock and Random123 positions suffice to reproduce future native current under the same exogenous schedule.',
        'Limiti': cfg['scope'], 'Condizioni di falsificazione': cfg['decision']})
    experiment = put('experiments', 'prospective', **{
        'Nome': 'IV-C3 integrated mixed synaptic interface',
        'Tipo': 'Confermativo', 'Stato': 'Preregistrato',
        'Ipotesi': [causal, restart], 'Descrizione': cfg['scope'],
        'Obiettivo informativo': 'Test causal mixed-event composition, per-receptor current attribution, and full native restart in a 3x4x3 confirmation matrix.'})
    put('protocols', 'prospective', **{
        'Nome': 'IV-C3 four-synapse mixed-burst protocol',
        'Esperimento': [experiment], 'Versione': 'v1',
        'Modalità': 'Componente isolato',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': cfg['decision'],
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': cfg['selection_policy'] + ' Native/runtime failure is inconclusive.'})
    put('decisions', 'prospective', **{
        'Nome': 'Run IV-C3 before preparing Task33',
        'Esito': 'Continuare',
        'Motivazione': 'IV-C1 deterministic receptor kernels and IV-C2 stochastic plasticity/release both passed prospectively; mixed-event current interface remains untested.',
        'Condizioni di revisione': cfg['decision']})
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'iv_c3_preregistered': True,
                      'task33_authorized': False, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
