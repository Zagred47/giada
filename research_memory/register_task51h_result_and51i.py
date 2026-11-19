import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51h_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51i_preregistration_v1.json').read_text())
    archive=repo/'artifacts/task51h_kaggle_external/task51h_final.zip';receipt=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51h-reach-preserve-v1'];dest=repo/'experiments/results/task51h_audit_v1';dest.mkdir(exist_ok=True);obs=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as handle:handle.write(text)
            art=put('artifacts','51h-result',name,Nome='51h '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations','51h-result',name,Nome='51h full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    comparison=(dest/'comparison.json').read_text()
    obs.append(put('observations','51h-result','comparison',Nome='51h complete comparison with51g',Descrizione=comparison))
    finding=put('findings','51h-result','comparison',Nome='51h calibration and preservation',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='All checkpoints finite and final/partial archives verified. none_constant40/72 openingRMSE.000446824; none_late42/72 .000485255; residual_constant48/72 .000464938; residual_late43/72 .000493669. linearize constant/late both72/72 .8.39109e-9; allsix select postprojection10k. Linearize final exact raw log error constant.000357074 versuslate.00000267536; both degrade versuspostprojection.000000418632. Calibration keeping nonlinear residual is insufficient.',Limitazioni='Static sampled amplitude only; no recurrent or matched speedup validation. Passing selection arises from fit-only projection, not subsequent Adam. Different data from51g; internal paired contrasts primary.')
    put('experiments','51h','reach-preserve',Stato='Concluso')
    for arm in ('none_constant','none_late','residual_constant','residual_late','linearize_constant','linearize_late'):put('runs','51h',arm,Stato='Completata')
    put('lessons','51h-result','learnability',Nome='Preserve solved static readout without optimizer-induced drift',Lezione='Zero nonlinear output plus fit-only coefficient calibration reaches numerical precision. Continuing Adam perturbs that solution; late decay reduces but does not remove drift. Distinguish epsilon-driven normalization, step scale, SGD and frozen baseline, and evaluate final rather than only best checkpoint.',Risultati=[finding],Limiti='Not a learned recurrent memory result. No requirement to retain needless gradient updates after static calibration.')
    e=put('experiments','51i','preservation',Nome='Task51i preserve calibrated solution',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Paired frozen, Adam default, small step, larger epsilon and SGD; final-checkpoint preservation primary.'})
    protocol=put('protocols','51i','preservation',Nome='51i projection and preservation matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Original amplitude gates unchanged; report initial/final/selected separately.','Regole di arresto':'Nonfinite/hash failure stops, preserve checkpoints.'})
    claims=[put('claims','51i',h['id'],Nome='51i '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51i','preservation',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms','51i',arm['id'],Nome='51i '+arm['id'],Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51i',arm['id'],Nome='51i '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired data and initial matrices')
    put('decisions','51i','proceed',Nome='Proceed51i optimization diagnosis',Esito='Continuare',Motivazione='Explicit user request: unchanged thresholds and abstraction, privateKaggle and external backups.',Risultati=[finding])
    put('decisions','51i','hard-stop-51j',Nome='Stop static-readout detour by51j',Esito='Continuare',Motivazione=p['stop_policy'],Risultati=[finding])
    put('observations','51g-result','comparison',Nome='51g complete comparison with51f')
    print('51h results and51i plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51i_protocol_v1';folder.mkdir(exist_ok=True);r=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',r);write_json(dest/'memory_receipt.json',r);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51h/51i DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51h/51i schema valid')
