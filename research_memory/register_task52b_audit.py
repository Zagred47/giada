import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;dest=repo/'experiments/results/task52b_audit_v1';old=json.loads((repo/'experiments/results/task52b_protocol_v1/preregistration_receipt.json').read_text())
    receipt=json.loads((dest/'kaggle_receipt.clarified.json').read_text());archive=repo/'artifacts/task52b_kaggle_external'/receipt['archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={};obs=[]
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52b-audit-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode()
            target=dest/(name if name.endswith('.gz') else name+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts',name,Nome='52b original '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations',name,Nome='52b original '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    for filename in ('audit.json','kaggle_receipt.original.json','kaggle_receipt.clarified.json'):
        obs.append(put('observations',filename,Nome='52b audit '+filename,Descrizione=(dest/filename).read_text()))
    exp=old['records']['experiments-giada-task52b-decoder-memory-v1']
    finding=put('findings','result',Nome='52b passive collapse and fixed decoder infeasibility',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='Technically complete1200updates,12finite members,0/36strict cases each arm. Autonomous V-RMS1.877-1.887mV versus52=6.677, but zero-ionic baseline1.893mV and candidate distance to it0.0083-0.0364mV: apparent voltage improvement mostly compatible with suppression of ionic dynamics. Frozen decoder box bounds exclude36/36cases,333/396channelcases for all four arms. Short h0 fit helps less than0.1percent. Multiscale Jacobians0.88-0.97 vs0.02-0.07 without quality gain. All select final1200; convergence unproven.',Limitazioni='No general impossibility for latent10. h0 oracle only10ms locally optimized, evaluated100ms; teacher reencoding is not an ideal encoder. Jacobians local. Ca/Na pass counts include imposed blocked channels. Runtime-valid receipt not scientific success. 52-to52b retraining/loss changes confound single-factor attribution.')
    if not args.validate_only:
        m.local_upsert('experiments',{'Stato':'Concluso'},record_id=exp)
        for code,rid in old['records'].items():
            if code.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    design=json.loads((repo/'experiments/task52b_failure_hypotheses_v1.json').read_text())
    for h in design['hypotheses']:put('claims',h['id'],Nome='52b next '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']+' Falsifier: '+h['falsifier'])
    put('observations','next-matrix',Nome='52b proposed decoder-first encoder x recurrence matrix',Descrizione=json.dumps(design))
    put('lessons','passive-control',Nome='Low voltage RMS can conceal missing ionic dynamics',Lezione='Compare against no-ionic-current control and preserve current/charge/event gates. A large RMS improvement is not evidence that the intended mechanism is recovered. Establish frozen decoder feasibility before interpreting initialization or memory oracles.',Risultati=[finding],Limiti='Do not turn a local diagnostic into global impossibility.')
    put('decisions','analysis-only',Nome='52b no promotion; propose decoder-first factorial diagnosis',Esito='Continuare',Motivazione='User requests registration, receipt clarification and hypotheses only. No new training/run launched. Same thresholds and latent abstraction; proposed D/DE/DR/DER matrix requires next preregistration.',Risultati=[finding])
    print('52b audit staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];write_json(dest/'memory_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52b audit DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52b audit schema valid')
