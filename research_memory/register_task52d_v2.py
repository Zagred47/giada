import argparse,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task52d_protocol_v1/memory_receipt.json').read_text());p=json.loads((repo/'experiments/task52d_preregistration_v2.json').read_text());m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52d-v2-{key}';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','balance-trajectory',Nome='52d authorized decoder recalibration: balance x trajectory',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate net-current/charge objective and autonomous trajectory objective, conditional on shared trainable decoder and recurrence; frozen encoder.'})
    proto=put('protocols','factorial',Nome='52d four-arm decoder-recalibrated protocol',Versione='v2',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Original52/46quality and intervention gates unchanged; all individual channels retained. No passive-collapse promotion.','Regole di arresto':'Nonfinite/hash/encoder drift stops; checkpoints100/300/800retained; no automatic next-task approval.'})
    hypotheses=[put('claims',h['id'],Nome='52d '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','balance-trajectory',Ipotesi=hypotheses)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52d '+arm['id'],Ruolo='Trattamento',Protocollo=[proto]);put('runs',arm['id'],Nome='52d '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired')
    put('observations','authorization',Nome='User authorized decoder recalibration after52d preflight',Descrizione='User chose Ricalibrare il decoder. This supersedes frozen-decoder constraint for new four-arm training, not the recorded210/300preflight result. Encoder frozen; no physical gates restored online. Same10latent coordinates, same network architecture, same thresholds. Decoder andrecurrence trained in allarms; only extra loss terms differ.')
    finding=old['records']['findings-giada-task52d-preflight-not-established-v1']
    put('decisions','proceed',Nome='Proceed52d with explicitly authorized trainable decoder',Esito='Continuare',Motivazione='User approved recalibration. Do not pretend prior frozen decoder feasibility passed; retain failed/inconclusive preflight. Prepare/push/launch privateKaggle with external checkpoint andresult recovery. DB backup local.',Risultati=[finding])
    print('52d v2 staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];folder=repo/'experiments/results/task52d_protocol_v2';folder.mkdir(exist_ok=True);write_json(folder/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52d v2 DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52d v2 schema valid')
