"""Prospective diagnostic48, local SQLite only; preserve47 NO-GO."""
import argparse,json,hashlib
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task48_protocol_v1';folder.mkdir(exist_ok=True)
    path=repo/'experiments/task48_preregistration_v1.json';p=json.loads(path.read_text());m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    baseline={} if args.validate_only else json.loads((repo/'experiments/results/task47_multi_event_v1/memory_receipt.json').read_text())['verification']
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=table+'-giada-task48-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','state-diagnostics',Nome='Task48 exploratory family state redundancy',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Distinguish instantaneous opening collisions, reachable-history redundancy, slow recovery observability and inherited P approximation; no training or compressed release.'})
    protocol=put('protocols','state-diagnostics',Nome='48 shared V/Ca clamp state probes',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Complete finite aligned diagnostics with verified unchanged frozenP, NumPy/Torch parity and identical-state controls. Diagnostic completion is not model qualification.','Regole di arresto':'Hash/parity/nonfinite/identity failure or runtime/output budget: preserve partial results; no training or threshold change.'})
    claims=[put('claims',h['id'],Nome='48 '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['interpretation']) for h in p['hypotheses']];put('experiments','state-diagnostics',Ipotesi=claims)
    for arm in ('canonical64','canonical32','P'):
        br=put('arms',arm,Nome='48 '+arm,Ruolo='Trattamento' if arm=='P' else 'Baseline',Protocollo=[protocol]);put('runs',arm,Nome='48 '+arm,Stato='Pianificata',Braccio=[br],Seed='Input9481103; frozenP model17/29/43; common initial canonical states and future drives')
    art=put('artifacts','plan',Nome='48 locked diagnostic plan',Tipo='Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    put('observations','inherited47limits',Nome='48 explicit inherited47 limits',Descrizione=p['limitations47'],**{'Artefatti dettagliati':[art]})
    put('decisions','exploratory-not-release',Nome='Proceed48 exploratorily without qualifying47',Esito='Continuare',Motivazione='Explicit user authorization. '+p['limitations47']+' '+p['decision'],**{'Data':'2026-11-15'})
    m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False,task47_passed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('48 SQLite verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('48 schema valid; no writes')
