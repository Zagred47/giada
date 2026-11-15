"""Full49 findings and prospective50 in one local-only verified transaction."""
import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;old=json.loads((repo/'experiments/results/task49_protocol_v1/preregistration_receipt.json').read_text());audit=json.loads((repo/'experiments/results/task49_audit_v1/audit.json').read_text());p=json.loads((repo/'experiments/task50_preregistration_v1.json').read_text())
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();base={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==base['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    archive=repo/'artifacts/task49_kaggle_external/task49_final.zip'
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==audit['archive_sha256']
    exp=old['records']['experiments-giada-task49-initialization-v1'];obs=[];dest=repo/'experiments/results/task49_audit_v1'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not(name.endswith('.json') or name.endswith('.json.gz')):continue
            raw=z.read(name);text=gzip.decompress(raw).decode() if name.endswith('.gz') else raw.decode();target=dest/(name if name.endswith('.gz') else name+'.gz')
            with gzip.open(target,'wt',encoding='utf-8') as f:f.write(text)
            art=put('artifacts','49-result',name,Nome='49 raw '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size})
            obs.append(put('observations','49-result',name,Nome='49 full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    obs.append(put('observations','49-result','audit',Nome='49 all initialization/cost comparisons',Descrizione=json.dumps(audit)))
    finding=put('findings','49-result','initialization',Nome='49 short warmup and linear codes fail; exact restart works',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(audit['summary']),Limitazioni=audit['scope']+' P still fails with exact canonical initialization; no coupled release. PCA failure does not disprove nonlinear compression.')
    put('findings','49-result','cost',Nome='49 initialization cost and slow memory',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(dict(cost=audit['cost_median_ms'],slow=audit['slow_gate_rmse'])),Limitazioni=audit['cost_scope'])
    put('experiments','49','initialization',Stato='Concluso')
    for arm in ['P_history_oracle','P_restart_copy','physical_identity','family_one','family_medium','warmup_0','warmup_1','warmup_8','warmup_40','warmup_100','warmup_300','warmup_400','shuffled_physical','canonical64','canonical32']:put('runs','49',arm,Stato='Completata')
    put('lessons','49-result','preserve-memory',Nome='49 preserve history; learn task-relevant nonlinear memory',Lezione='Exact state restart is bitwise stable. Short warmup loses slow memory; post-rest does not guarantee forgetting. PCA5/10 insufficient; larger linear code is not uniformly better. Distinguish initialization loss from inherited P dynamics and do not claim initialization latency as end-to-end speedup.',Risultati=[finding],Limiti=audit['scope'])
    e=put('experiments','50','compact-memory',Nome='Task50 learned compact family recurrence',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Orthogonal latent/trunk capacity and budget comparisons; nonlinear state-available encoding, no online gate decoder.'})
    protocol=put('protocols','50','compact-memory',Nome='50 paired five-arm three-seed matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[e],Procedura=json.dumps(p),**{'Criteri di successo':'Unchanged current quality thresholds per case; exact frozen P; development selection before fresh; no coupled release.','Regole di arresto':'Nonfinite parameters, changed P hash or runtime failure preserves checkpoints and partial reports.'})
    claims=[put('claims','50',h['id'],Nome='50 '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']) for h in p['hypotheses']];put('experiments','50','compact-memory',Ipotesi=claims)
    for arm in p['arms']:
        br=put('arms','50',arm['id'],Nome='50 '+arm['id'],Ruolo='Trattamento',Protocollo=[protocol]);put('runs','50',arm['id'],Nome='50 '+arm['id'],Stato='Pianificata',Braccio=[br],Seed='17,29,43; shared minibatches, separate parameters')
    put('decisions','50','proceed',Nome='Proceed50 nonlinear family memory after49',Esito='Continuare',Motivazione='User explicitly authorized learned compact recurrence, frozenP, same thresholds, single Kaggle notebook and external backup.',Risultati=[finding],**{'Data':'2026-11-15'})
    print('49 results and50 plan staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(base);assert v['valid'];folder=repo/'experiments/results/task50_protocol_v1';folder.mkdir(exist_ok=True);receipt=dict(valid=True,verification=v,records=ids,airtable_accessed=False);write_json(folder/'preregistration_receipt.json',receipt);write_json(dest/'memory_receipt.json',receipt);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('49/50 DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('49/50 schema valid')
