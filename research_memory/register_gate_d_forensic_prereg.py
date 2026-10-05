"""Preregister the diagnostic decomposition; preserve Gate D's NO-GO."""
import hashlib
import json

from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo = ROOT.parent
    path = repo / 'experiments/task28d_gate_d_bottleneck_forensic.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    from src.giada_teacher.gate_d_bottleneck_forensic import verify_parent
    verify_parent(repo, cfg)
    mirror = ValidatedBatchMirror()
    def put(table, key, **fields):
        return mirror.local_upsert(table, {
            'Codice stabile': f'{table}-giada-gated-forensic-{key}-v1', **fields
        })['record_id']
    claim = put('claims', 'neural-core-cost', **{
        'Nome': 'The five learned gate updates are the compiled latency bottleneck',
        'Tipo': 'Ipotesi', 'Stato': 'Aperta',
        'Enunciato': 'At batch 642 the same-backend compiled neural five-channel core is slower than the five canonical formulas, or adds launch and fusion costs when recomposed.',
        'Condizioni di falsificazione': 'The five-channel learned core is not slower and kernel evidence attributes the whole-block gap elsewhere.'
    })
    experiment = put('experiments', 'decomposition', **{
        'Nome': 'Gate D compiled bottleneck decomposition', 'Tipo': 'Confermativo',
        'Stato': 'Preregistrato', 'Ipotesi': [claim],
        'Descrizione': cfg['purpose'],
        'Obiettivo informativo': 'Distinguish neural core cost, six-formula tail, analytic current and recomposition/fusion effects without a new promotion test.'
    })
    put('protocols', 'same-backend', **{
        'Nome': 'Gate D same-input compiled factorial decomposition',
        'Esperimento': [experiment], 'Versione': 'v1', 'Modalità': 'Ricomposizione',
        'Procedura': json.dumps(cfg, ensure_ascii=False),
        'Criteri di successo': 'Valid output identity and paired component/full timings with CUDA profiler support.',
        'Hash preregistrazione': hashlib.sha256(path.read_bytes()).hexdigest(),
        'Regole di arresto': 'Diagnostic only; Task29 remains unauthorized regardless of latency attribution.'
    })
    mirror.commit_batch()
    write_json(ROOT / 'data/airtable_snapshot.json', mirror.export_snapshot())
    assert mirror.verify()['valid']
    print(json.dumps({'valid': True, 'diagnostic_only': True, 'airtable_accessed': False}))


if __name__ == '__main__':
    main()
