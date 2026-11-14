"""Prospective47 protocol and hypotheses, local SQLite only."""
import argparse,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task47_protocol_v1';folder.mkdir(exist_ok=True)
    p=json.loads((repo/'experiments/task47_preregistration_v1.json').read_text());m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    baseline={} if args.validate_only else json.loads((repo/'experiments/results/task46_retry_v1/result_memory_receipt.json').read_text())['verification']
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=table+'-giada-task47-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','confirmation',Nome='Task47 independent frozenP level-one confirmation',Tipo='Confermativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Independent spatial/input transfer, matched causal interventions and reproducibility of cost frontier; no refit or altered gates.'})
    protocol=put('protocols','confirmation',Nome='47 new geometry/input-seed matched confirmation',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':p['decision_rule'],'Regole di arresto':'Wrong immutable hash, nonfinite state, parity or native floor failure remains explicit; preserve partial output. No training/selection/threshold change.'})
    claims=[put('claims',h['id'],Nome='47 '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['contrast'],Limiti='Prospective, not established. Finite ensemble seeds reused; new input seeds.45e remains incomplete; no global biological equivalence.') for h in p['hypotheses']];put('experiments','confirmation',Ipotesi=claims)
    for arm in p['arms']:
        for mode in ('imposed','autonomous'):
            key=arm+'-'+mode;br=put('arms',key,Nome='47 '+key,Ruolo='Trattamento' if arm=='P' else 'Baseline',Protocollo=[protocol]);put('runs',key,Nome='47 '+key,Stato='Pianificata',Braccio=[br],Seed='Frozen model17/29/43; disjoint input9471103/9471127; two new geometries,24paired cases each')
    put('decisions','proceed',Nome='Proceed47 independent quality confirmation, cost separate',Esito='Continuare',Motivazione='User approved proceeding after successful46retry. P144/144 per mode/horizon in46, but coupled component step ~2.1x slower than formulas. No speedup or45ecompletion claim; immutable P and gates retained.',**{'Data':'2026-11-14'})
    m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('47 memory verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('47 schema valid; no writes')
