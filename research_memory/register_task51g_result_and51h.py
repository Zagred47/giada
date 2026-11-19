import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51g_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51h_preregistration_v1.json').read_text())
    archive=repo/'artifacts/task51g_kaggle_external/task51g_final.zip';receipt=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51g-optimization-gap-v1'];dest=repo/'experiments/results/task51g_audit_v1';dest.mkdir(exist_ok=True);obs=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as handle:handle.write(text)
            art=put('artifacts','51g-result',name,Nome='51g '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations','51g-result',name,Nome='51g full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    comparison=(dest/'comparison.json').read_text()
    obs.append(put('observations','51g-result','comparison',Nome='51g complete comparison with51e',Descrizione=comparison))
    finding=put('findings','51g-result','comparison',Nome='51g separates discovery from preservation',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='All checkpoints finite and final archive verified. Standard constant reachable openingRMSE .000915627 and26/72 trajectories; standard decay .002379368 and0/72. Fit constant8.4534e-9 andfit decay9.5803e-9, both72/72; fiveof6fitinitialized checkpoints selectedat0. Same-run development score constant improves8.14x from5kto10k then mixed at15k. Late regularization remains untested. Different datasets limit causal comparison to51f.',Limitazioni='Static sampled-amplitude gates only. Fit initialization is estimated coefficients, not autonomous discovery by Adam. No recurrence or matched-workload speedup.')
    put('experiments','51g','optimization-gap',Stato='Concluso')
    for arm in ('standard_constant','standard_decay','fit_constant','fit_decay'):put('runs','51g',arm,Stato='Completata')
    put('lessons','51g-result','learnability',Nome='Reach and preserve require different optimization phases',Lezione='More constant-step training improves discovery; early decay stalls. Fit-estimated initialization starts essentially exact; constant Adam can degrade it, while decay preserves it. Test readout calibration and late decay separately, retaining before/after checkpoints.',Risultati=[finding],Limiti='No claim of recurrent compression or SGD discovering the fitted initialization.')
    e=put('experiments','51h','reach-preserve',Nome='Task51h reach and preserve',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'3x2 projection x late schedule, same architecture and thresholds; explicit before/after attribution.'})
    protocol=put('protocols','51h','reach-preserve',Nome='51h projection and preservation matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Original amplitude gates unchanged; report initial/final/selected separately.','Regole di arresto':'Nonfinite/hash failure stops, preserve checkpoints.'})
    claims=[put('claims','51h',h['id'],Nome='51h '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51h','reach-preserve',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms','51h',arm['id'],Nome='51h '+arm['id'],Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51h',arm['id'],Nome='51h '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired data and initial matrices')
    put('decisions','51h','proceed',Nome='Proceed51h optimization diagnosis',Esito='Continuare',Motivazione='Explicit user request: unchanged thresholds and abstraction, privateKaggle and external backups.',Risultati=[finding])
    put('decisions','51h','hard-stop-51j',Nome='Stop static-readout detour by51j',Esito='Continuare',Motivazione=p['stop_policy'],Risultati=[finding])
    print('51g results and51h plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51h_protocol_v1';folder.mkdir(exist_ok=True);r=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',r);write_json(dest/'memory_receipt.json',r);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51g/51h DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51g/51h schema valid')
