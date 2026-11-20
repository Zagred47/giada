import argparse,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;old=json.loads((repo/'experiments/results/task52_audit_v1/memory_receipt.json').read_text());p=json.loads((repo/'experiments/task52b_preregistration_v1.json').read_text())
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52b-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    e=put('experiments','decoder-memory',Nome='Task52b decoder x memory diagnosis',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate frozen-decoder range, slow-state retention, encoder error and finite optimization budget without changing latent10 or quality gates.'})
    protocol=put('protocols','factorial',Nome='52b paired four-arm matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Identical Task52/46 thresholds per case/channel and pooled voltage. Oracles are nonselectable. No success from lower training loss alone.','Regole di arresto':'Nonfinite parameters/gradients or hash failure stops. Preserve checkpoints at100/400/1200; no silent budget/scope changes.'})
    claims=[put('claims',h['id'],Nome='52b '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']]
    put('experiments','decoder-memory',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms',arm['id'],Nome='52b '+arm['id'],Ruolo='Trattamento',Protocollo=[protocol])
        put('runs',arm['id'],Nome='52b '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired; independent optimizer moments and clipping')
    put('observations','diagnostic-contract',Nome='52b initialization controls and inference limits',Descrizione='After development-only checkpoint freeze: normal encoder vs future-assisted h0 optimization vs physical-state re-encoding. Future oracles change only h0 and are not selectable. Full physical18state is only initialization input for actual candidate. Output state remains latent10; rate initialization adds no runtime equations. Budget1200 cannot establish convergence or sufficient1000ms training. Same discrete reference as52, not a new nativeNEURON audit. DB backup remains local.')
    put('decisions','authorization',Nome='Authorize52b controlled diagnosis',Esito='Continuare',Motivazione='Explicit user authorization for four arms, unchanged thresholds/abstraction, Kaggle execution and external result/checkpoint backups; local SQLite backup only.')
    print('52b schema staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid']
    folder=repo/'experiments/results/task52b_protocol_v1';folder.mkdir(exist_ok=True)
    write_json(folder/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52b DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52b schema valid')
