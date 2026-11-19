import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51e_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51f_preregistration_v1.json').read_text());archive=repo/'artifacts/task51e_kaggle_external/task51e_final.zip';r=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51e-conditioning-v1'];obs=[];dest=repo/'experiments/results/task51e_audit_v1';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts','51e-result',name,Nome='51e '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size});obs.append(put('observations','51e-result',name,Nome='51e full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    finding=put('findings','51e-result','comparison',Nome='51e local8 hybrid best but still no joint trajectory passes',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='Local8hybrid reachable openingRMSE0.002110/logRMSE0.088969; shared32hybrid0.0059999, shared88hybrid0.0069766. Allarms0/24perseed. Local8hybrid Ih72/72 per-channel passes, Ca_LVAst0/72; residual errors remain acrossNa,K. Worst opening sample Ca_HVA abs0.043846. Fit-only full linear control openingRMSE0.024167/logRMSE0.572628, rank19.',Limitazioni='Newdata/budget confound cross-task attribution. Floor1e-30 distorts extreme synthetic loglabels: hypothesis, not yet causal result. No temporal conclusion.')
    put('experiments','51e','conditioning',Stato='Concluso')
    for rep in ('shared32','shared88','local8','linear'):
        for loss in ('hybrid','raw_log'):put('runs','51e',rep+'_'+loss,Stato='Completata')
    put('lessons','51e-result','exact-targets',Nome='51e distinguish useful inductive bias from label distortion',Lezione='Local8 improves hybrid accuracy but not all channels pass. Inspect both capacity and loglabel clipping before inferring optimizer failure from linear regression. Exact label construction remains separate from inference architecture.',Risultati=[finding],Limiti='No compressed or recurrent release.')
    e=put('experiments','51f','local-targets',Nome='Task51f local capacity and target construction',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Local8/24 x floor/exact log labels plus legacy8; paired optimization and fit-only local linear controls.'})
    protocol=put('protocols','51f','local-targets',Nome='51f capacity-target matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Original sampled-current amplitude gates unchanged; exact-log residual separately; no temporal claim.','Regole di arresto':'Nonfinite/hash/zero-contract failure stops with checkpoints preserved.'})
    claims=[put('claims','51f',h['id'],Nome='51f '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51f','local-targets',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms','51f',arm['id'],Nome='51f '+arm['id'],Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51f',arm['id'],Nome='51f '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired matrices and stream')
    put('decisions','51f','proceed',Nome='Proceed51f residual diagnosis',Esito='Continuare',Motivazione='Explicit user approval: same thresholds, privateKaggle and externalbackups.',Risultati=[finding],**{'Data':'2026-11-19'})
    print('51e results and51f plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51f_protocol_v1';folder.mkdir(exist_ok=True);receipt=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',receipt);write_json(dest/'memory_receipt.json',receipt);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51e/51f DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51e/51f schema valid')
