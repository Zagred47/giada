import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51b_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51c_preregistration_v1.json').read_text());archive=repo/'artifacts/task51b_kaggle_external/task51b_final.zip';receipt=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51b-small-openings-v1'];obs=[];dest=repo/'experiments/results/task51b_audit_v1';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts','51b-result',name,Nome='51b '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size});obs.append(put('observations','51b-result',name,Nome='51b full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    finding=put('findings','51b-result','tradeoff',Nome='51b log loss improves small openings but no quality pass',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='All6arms0/24trajectory amplitude passes perseed. Mean reachable logRMSE: sigmoid absolute3.542, sigmoid logbalanced0.183, fitbias logbalanced0.349, log_exp logbalanced0.162; synthetic log_exp0.301. Tiny<1e-12 log_exp0.075decades vs sigmoid0.167. Absolute opening error not improved consistently. No temporal test.',Limitazioni='Log loss and balanced sampling are combined; no recurrence/compression claim. FP32 underflow4extreme control entries is explicit, not general explanation.')
    put('experiments','51b','small-openings',Stato='Concluso')
    for r in ('sigmoid_zero','sigmoid_fit_bias','log_exp'):
        for l in ('absolute','log_balanced'):put('runs','51b',r+'_'+l,Stato='Completata')
    put('lessons','51b-result','dual-objective',Nome='51b small relative errors and large absolute errors need joint calibration',Lezione='Bias alone insufficient; log coordinate + balanced objective improves tiny openings but may sacrifice large-opening absolute accuracy. Test weights jointly with identical sampling, not by changing evaluation gates.',Risultati=[finding],Limiti='No release or temporal inference.')
    e=put('experiments','51c','hybrid',Nome='Task51c small-large opening hybrid loss',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Compare lambda0/1/10 across sigmoid and log output with matched balanced sampling.'})
    protocol=put('protocols','51c','hybrid',Nome='51c2x3hybridloss',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Original amplitude gates unchanged; separate absolute/log metrics, reachable/synthetic groups.','Regole di arresto':'Nonfinite/hash/control failures preserve checkpoints; no fresh selection.'})
    claims=[put('claims','51c',h['id'],Nome='51c '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51c','hybrid',Ipotesi=claims)
    for r in p['representations']:
        for l in p['losses']:
            key=r+'_'+l;br=put('arms','51c',key,Nome='51c '+key,Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51c',key,Nome='51c '+key,Stato='Pianificata',Braccio=[br],Seed='17,29,43; identical matrices and balanced sample stream')
    put('decisions','51c','proceed',Nome='Proceed51c hybrid objective',Esito='Continuare',Motivazione='Explicit user authorization; frozenP, no changed thresholds, privateKaggle backup.',Risultati=[finding],**{'Data':'2026-11-15'})
    print('51b results and51c plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51c_protocol_v1';folder.mkdir(exist_ok=True);r=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',r);write_json(dest/'memory_receipt.json',r);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51b/51c DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51b/51c schema valid')
