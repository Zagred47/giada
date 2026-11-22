import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task52e_protocol_v1/preregistration_receipt.json').read_text());dest=repo/'experiments/results/task52e_audit_v1';design=json.loads((repo/'experiments/task52f_hypotheses_v1.json').read_text())
    receipt=json.loads((repo/'artifacts/task52e_kaggle_external/kaggle_receipt.json').read_text());archive=repo/'artifacts/task52e_kaggle_external'/receipt['archive']
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={};obs=[]
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52e-audit-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts',name,Nome='52e original '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations',name,Nome='52e original '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    obs.append(put('observations','audit',Nome='52e checkpoint rejection audit',Descrizione=(dest/'audit.json').read_text()))
    exp=old['records']['experiments-giada-task52e-support-family-v1']
    finding=put('findings','result',Nome='52e partial development gains fail joint non-regression guard',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(design['facts']),Limitazioni=design['interpretation'])
    if not args.validate_only:
        m.local_upsert('experiments',{'Stato':'Concluso'},record_id=exp)
        for key,rid in old['records'].items():
            if key.startswith('runs-'):m.local_upsert('runs',{'Stato':'Completata'},record_id=rid)
    for h in design['hypotheses']:put('claims',h['id'],Nome='52f proposed '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['test']+' Falsifier: '+h['falsifier'])
    put('observations','52f-proposal',Nome='52f encoder-alignment x temporal-credit proposal',Descrizione=json.dumps(design))
    put('lessons','joint-guard',Nome='Lower mean loss does not imply joint channel preservation',Lezione='52e combined300 improves most development metrics but differentseeds regress onCa_HVA,Nap_Et2,NaTa_t orCa. Preserve both per-seed andper-channel views. Fallback protects the deployedcandidate but hides partial training gains in final-only comparisons.',Risultati=[finding])
    put('decisions','budget-52j',Nome='User caps Task52 refinements at52j',Esito='Continuare',Motivazione=json.dumps(design['user_budget'])+' No52f launch authorized by this analysis turn.',Risultati=[finding])
    m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];write_json(dest/'memory_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52e audit and52f proposal verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52e audit schema valid')
