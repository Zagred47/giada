import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51c_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51d_preregistration_v1.json').read_text());archive=repo/'artifacts/task51c_kaggle_external/task51c_final.zip';r=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51c-hybrid-v1'];obs=[];dest=repo/'experiments/results/task51c_audit_v1';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts','51c-result',name,Nome='51c '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size});obs.append(put('observations','51c-result',name,Nome='51c full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    finding=put('findings','51c-result','tradeoff',Nome='51c log hybrid1 improves absolute accuracy but no full pass',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='Allarms0/24perseed. Reachable log_exp hybrid1 openingRMSE0.01218/logRMSE0.15483 vs logonly0.05804/0.09811; lambda10worse0.02106/0.83957. Tiny<1e-12 hybrid1logRMSE0.16761 vslogonly0.05338. Worst sampled Ca_LVAst opening absolute error0.47276. Multiple channels retain relative current error.',Limitazioni='Cross51b comparison confounded by new data and budget. Static sparse trajectories only; no temporal or coupled claim. Common selection objective fixed, no fresh tuning.')
    put('experiments','51c','hybrid',Stato='Concluso')
    for rep in ('sigmoid_zero','log_exp'):
        for loss in ('log_only','hybrid1','hybrid10'):put('runs','51c',rep+'_'+loss,Stato='Completata')
    put('lessons','51c-result','local-relative',Nome='51c inspect channel-local mapping and relative residuals',Lezione='Hybrid log+absolute improves compromise but cannot meet all channel gates. Investigate cross-channel interference and relative precision instead of assuming only tiny openings remain problematic.',Risultati=[finding],Limiti='Hypotheses not yet causal conclusions.')
    e=put('experiments','51d','local-relative',Nome='Task51d local readout and relative residuals',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'2x2channelmask/objective, budget ladder, nonselectable exact oracle.'})
    protocol=put('protocols','51d','local-relative',Nome='51dlocalrelative2x2',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Same current amplitude gates; separate synthetic/reachable; oracle not selectable.','Regole di arresto':'Nonfinite/hash/control failure, preserve checkpoints.'})
    claims=[put('claims','51d',h['id'],Nome='51d '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51d','local-relative',Ipotesi=claims)
    for rep in p['representations']:
        for loss in p['losses']:
            key=rep+'_'+loss;br=put('arms','51d',key,Nome='51d '+key,Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51d',key,Nome='51d '+key,Stato='Pianificata',Braccio=[br],Seed='17,29,43; paired matrices/streams')
    put('decisions','51d','proceed',Nome='Proceed51d residual diagnosis',Esito='Continuare',Motivazione='User authorized same thresholds, single notebook, privateKaggle and externalbackup.',Risultati=[finding],**{'Data':'2026-11-15'})
    print('51c result and51d plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51d_protocol_v1';folder.mkdir(exist_ok=True);receipt=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',receipt);write_json(dest/'memory_receipt.json',receipt);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51c/51d DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51c/51d schema valid')
