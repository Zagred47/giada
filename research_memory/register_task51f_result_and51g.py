import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;old=json.loads((repo/'experiments/results/task51f_protocol_v1/preregistration_receipt.json').read_text());p=json.loads((repo/'experiments/task51g_preregistration_v1.json').read_text())
    archive=repo/'artifacts/task51f_kaggle_external/task51f_final.zip';receipt=json.loads((archive.parent/'kaggle_receipt.json').read_text())
    with archive.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=old['records']['experiments-giada-task51f-local-targets-v1'];dest=repo/'experiments/results/task51f_audit_v1';dest.mkdir(exist_ok=True);obs=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as handle:handle.write(text)
            art=put('artifacts','51f-result',name,Nome='51f '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations','51f-result',name,Nome='51f full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    comparison=(dest/'comparison.json').read_text()
    obs.append(put('observations','51f-result','comparison',Nome='51f complete comparison with51e',Descrizione=comparison))
    finding=put('findings','51f-result','comparison',Nome='Exact labels expose remaining optimization gap',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato='All checkpoints finite, final archive hash/CRC verified. All arms0/24 trajectories perseed. Reachable mean openingRMSE legacy8 .00309792; local8_floor .00397914; local8_exact .00393249; local24_floor .00414790; local24_exact .00267758. Best51e local8hybrid .00211006, different datasets prevent causal cross-task attribution. Fit-only local least squares exact logRMSE5.94e-15 vs floored1.27991. Exact24 beats exact8 by32% openingRMSE; all exact arms select5000. Ih/Im/SKv3_1 exact24 72/72 each, Ca_LVAst3/72.',Limitazioni='Static full-state map; no recurrent or temporal qualification. No matched workload speedup. Exact solver is fitted coefficients, not SGD discovery.')
    put('experiments','51f','local-targets',Stato='Concluso')
    for arm in ('legacy8','local8_floor','local8_exact','local24_floor','local24_exact'):put('runs','51f',arm,Stato='Completata')
    put('lessons','51f-result','learnability',Nome='Separate label distortion from optimizer reachability',Lezione='Exact labels recover a near-exact fit-only solution without supplied exponents. Capacity helps only when labels are correct; fixed-budget Adam remains far away. Diagnose initialization, step-size and nested budget without treating theoretical existence as learned success.',Risultati=[finding],Limiti='Do not conflate least-squares initialization with a spontaneously learned neural discovery.')
    e=put('experiments','51g','optimization-gap',Nome='Task51g exact-label optimization gap',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'2x2 fit-initialization x schedule plus nested budget, same architecture and unchanged thresholds.'})
    protocol=put('protocols','51g','optimization-gap',Nome='51g paired optimization matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Original amplitude gates unchanged; report initial/final/selected separately.','Regole di arresto':'Nonfinite/hash failure stops, preserve checkpoints.'})
    claims=[put('claims','51g',h['id'],Nome='51g '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','51g','optimization-gap',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms','51g',arm['id'],Nome='51g '+arm['id'],Ruolo='Trattamento',Protocollo=[protocol]);put('runs','51g',arm['id'],Nome='51g '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43 paired data and initial matrices')
    put('decisions','51g','proceed',Nome='Proceed51g optimization diagnosis',Esito='Continuare',Motivazione='Explicit user request: unchanged thresholds and abstraction, privateKaggle and external backups.',Risultati=[finding])
    print('51f results and51g plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task51g_protocol_v1';folder.mkdir(exist_ok=True);r=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',r);write_json(dest/'memory_receipt.json',r);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('51f/51g DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('51f/51g schema valid')
