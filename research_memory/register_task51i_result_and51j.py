import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51i_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51j_preregistration_v1.json').read_text())
    archive=repo/'artifacts/task51i_kaggle_external/task51i_final.zip';receipt=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51i-preservation-v1'];dest=repo/'experiments/results/task51i_audit_v1';dest.mkdir(exist_ok=True);obs=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as handle:handle.write(text)
            art=put('artifacts','51i-result',name,Nome='51i '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations','51i-result',name,Nome='51i full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    comparison=(dest/'comparison.json').read_text()
    obs.append(put('observations','51i-result','comparison',Nome='51i complete comparison with51h',Descrizione=comparison))
    finding=put('findings','51i-result','comparison',Nome='51i preservation confirmed within diagnostic sample',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='All checkpoints finite; final and latest checkpoint backup hash/CRC verified. All five arms final72/72; all selectedinitial0. Final openingRMSE frozen7.78885e-9, SGD7.75422e-9, Adamlargeeps1.10485e-8, Adamsmallstep3.67865e-7, Adamdefault1.51673e-5. Initial gradient norm8.78e-10 produces first defaultAdam update1.71786e-4 versuslargeeps1.5218e-8. Large epsilon suppresses amplification; SGD preserves numerical accuracy. No training benefit over calibrated initial weights.',Limitazioni='Static full-state readout, not recurrent/closed-loop qualification. No matched speedup claim. Numerical differences across51h data not causal, within51i comparisons paired.')
    put('experiments','51i','preservation',Stato='Concluso')
    for arm in ('frozen','adam_default','adam_small_step','adam_large_eps','sgd'):put('runs','51i',arm,Stato='Completata')
    put('lessons','51i-result','learnability',Nome='Avoid amplifying numerical residuals in solved readout',Lezione='SGD or Adam with epsilon1e-4 preserves fitted static readout far better than defaultAdam. Report final and selected separately: allselected0 while final errors differ. Freeze solved model when no adaptation is required; confirmation must use new data and seeds.',Risultati=[finding],Limiti='No inference about learning unsolved recurrent dynamics.')
    e=put('experiments','51j','confirmation',Nome='Task51j independent preservation confirmation',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Confirm fixed SGD and Adam epsilon with new streams/seeds and48freshhistories. Close51series thenTask52.'})
    protocol=put('protocols','51j','confirmation',Nome='51j projection and preservation matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Original amplitude gates unchanged; report initial/final/selected separately.','Regole di arresto':'Nonfinite/hash failure stops, preserve checkpoints.'})
    claims=[put('claims','51j',h['id'],Nome='51j '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51j','confirmation',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms','51j',arm['id'],Nome='51j '+arm['id'],Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51j',arm['id'],Nome='51j '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired data and initial matrices')
    put('decisions','51j','proceed',Nome='Proceed51j optimization diagnosis',Esito='Continuare',Motivazione='Explicit user request: unchanged thresholds and abstraction, privateKaggle and external backups.',Risultati=[finding])
    put('decisions','51j','hard-stop-51j',Nome='Stop static-readout detour by51j',Esito='Continuare',Motivazione=p['stop_policy'],Risultati=[finding])

    put('protocols','51i','preservation',Nome='51i optimizer preservation matrix')
    print('51i results and51j plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51j_protocol_v1';folder.mkdir(exist_ok=True);r=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',r);write_json(dest/'memory_receipt.json',r);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51i/51j DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51i/51j schema valid')
