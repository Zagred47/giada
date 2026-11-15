import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51d_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51e_preregistration_v1.json').read_text());archive=repo/'artifacts/task51d_kaggle_external/task51d_final.zip';r=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51d-local-relative-v1'];obs=[];dest=repo/'experiments/results/task51d_audit_v1';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts','51d-result',name,Nome='51d '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size});obs.append(put('observations','51d-result',name,Nome='51d full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    finding=put('findings','51d-result','comparison',Nome='51d shared hybrid best; local and relative not resolved',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='Allarms0/24perseed. Reachable shared hybrid logRMSE0.104815/openingRMSE0.008796; sharedrelative0.383868/0.020341; localhybrid0.528507/0.055480; localrelative0.964921/0.046816. Exact nonselectable oracle maxerror1.11e-16. Sharedhybrid observed28%RMSE improvement vs51c but newdata/budget confound attribution.',Limitazioni='Local branch only2–3units/channel; capacity and locality confounded. Static/fullstate only. Oracle representability does not imply optimizer learnability. Most checkpoints selected5000, one3000; no general saturation proof.')
    put('experiments','51d','local-relative',Stato='Concluso')
    for rep in ('log_exp','log_local'):
        for loss in ('hybrid1','relative_hybrid'):put('runs','51d',rep+'_'+loss,Stato='Completata')
    e=put('experiments','51e','conditioning',Nome='Task51e readout capacity parameterization optimization',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Separate shared capacity, adequately wide local readout, raw log objective and learned linear solver.'})
    protocol=put('protocols','51e','conditioning',Nome='51e4x2readout matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Same amplitude gates; report raw/clipped errors and fresh separately.','Regole di arresto':'Hash/nonfinite/control failure preserves checkpoints; no changed gates.'})
    claims=[put('claims','51e',h['id'],Nome='51e '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51e','conditioning',Ipotesi=claims)
    for rep in p['representations']:
        for loss in p['losses']:
            key=rep+'_'+loss;br=put('arms','51e',key,Nome='51e '+key,Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51e',key,Nome='51e '+key,Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired sample stream')
    put('decisions','51e','proceed',Nome='Proceed51e readout diagnosis',Esito='Continuare',Motivazione='User authorized capacity/parameterization/optimization contrasts, same thresholds, privateKaggle and externalbackup.',Risultati=[finding],**{'Data':'2026-11-15'})
    print('51d results and51e plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51e_protocol_v1';folder.mkdir(exist_ok=True);receipt=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',receipt);write_json(dest/'memory_receipt.json',receipt);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51d/51e DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51d/51e schema valid')
