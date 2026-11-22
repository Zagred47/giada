import argparse,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task52e_audit_v1/memory_receipt.json').read_text());p=json.loads((repo/'experiments/task52f_preregistration_v1.json').read_text());m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52f-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','encoder-credit',Nome='52f encoder alignment x temporal credit',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate encoder initialization mismatch and temporal gradient truncation without changing logical10state representation or scientific thresholds.'})
    proto=put('protocols','factorial',Nome='52f paired four-arm protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Original52quality/intervention gates unchanged. Same per-channel andseparateV/Ca nonregression guard against original52d reference; initial fallback is not improvement.','Regole di arresto':'Nonfinite/hash drift or incorrect masks stop. User cap52j; no52k; no automatic promotion to53.'})
    claims=[]
    for key,claim in [('encoder','Fit-only encoder calibration reduces initial misalignment and permits current improvements.'),('credit','512tick temporal credit improves dynamics compared with128tick truncation at identical forward horizon.'),('interaction','Encoder alignment andlonger temporal credit jointly reduce regressions.')]:claims.append(put('claims',key,Nome='52f '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=claim))
    put('experiments','encoder-credit',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52f '+arm['id'],Ruolo='Trattamento',Protocollo=[proto]);put('runs',arm['id'],Nome='52f '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired')
    put('observations','contract',Nome='52f calibration andtruncation contract',Descrizione='Allarms start52dtrajectory.80fit-only encoder steps (maskedzero updates forcontrols); decoder andrecurrence unchanged inwarmup. Allencoders frozen during800mainupdates. Same512tick forward, BPTT128vs512perarm; denser checkpoints50,100,150,200,300,400,600,800. No newinput dataset needed.')
    put('decisions','proceed',Nome='Launch52f onKaggle afterpreflight',Esito='Continuare',Motivazione='User authorized notebook,DB,push,Kaggle. Externalbackup supervisor; local DBbackup. Respect52j ceiling. Negative result never relabelled passed.')
    m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];folder=repo/'experiments/results/task52f_protocol_v1';write_json(folder/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52f DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52f schema valid')
