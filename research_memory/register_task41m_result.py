"""Complete41m original evidence and scoped conclusions; SQLite only."""
import argparse,hashlib,json,zipfile
from datetime import datetime,timezone
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

STATEMENTS={
 'quality':'Same compared autonomous+imposed-stress/recurrence evaluations: H primary3 failures,Ca-only primary2,Ca-only voltage/solve64 zero.1320/1320 autonomous trajectories,840/840responses,72/72stress and144/144recurrence for combined diagnostic. Imposed stress/recurrence reused byte-identically, not newly independent tests.',
 'separation':'Frozen swap000 and swap100 share all saved tensors except a two-gate Ca_HVA selection mask; donor tensors equal K_full. Ca-only transplant removes both Ca_HVA boundary stress failures and avoids donor Nap/K_Tst imposed regressions. Numerical intervention removes remaining autonomous response failures. No new training or abstraction change.',
 'reference':'Ca-only voltage/solve64 has0failed responses versus BOTH originalFP32 and matched canonical reference. Recentered32 has3versus original but0versus matched: cannot silently promote a matched-reference diagnostic to original-contract GO.',
 'cost':'At642sites/seed integrated Ca-only mixed precision takes2.014x original formula and1.697x original H. Two-network transplant primary takes1.532x H; mixed precision adds1.108x over that transplant. At4096 candidate takes5.216x formula. No qualified acceleration; exact execution optimization remains separate from learned compression.',
 'scope':'Technically complete in668.17seconds, all saved tensors finite, final701195672-byte external archive SHA/CRC verified. Already-opened diagnostic panels, fixed modelseeds17/29/43; not fresh independent confirmation. OriginalFP32 release contract remains unpassed; Task42 not authorized.',
 'decision':'User authorizes fresh independent confirmation of the frozen selective/mixed-precision function and same-notebook output-preserving execution optimization. No retraining, no new fitting or model selection on fresh outcomes. Keep original thresholds and original/matched canonical references; backup DB local only.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41m_audited_v1';audit=json.loads((folder/'audit.json').read_text());assert audit['valid'] and audit['complete']
    archive=repo/audit['archive'];baseline=json.loads((repo/'experiments/results/task41m_protocol_v1/launch_memory_receipt.json').read_text())['verification']
    if (folder/'memory_receipt.json').exists() and not args.validate_only:print('41m results already registered');return
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41m-result-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code;return rows[0]['record_id']
    aid=put('artifacts','archive',Nome='Task41m complete final external archive',Tipo='Report',Versione='v1',Percorso=audit['archive'],**{'SHA-256':audit['sha256'],'Dimensione byte':audit['bytes']})
    raw_reports=[]
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==audit['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in audit['original_reports_sha256'].items():
            raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==sha
            art=put('artifacts','member-'+name,Nome='Task41m original '+name,Tipo='Report',Versione='v1',Percorso=audit['archive']+'#'+name,**{'SHA-256':sha,'Dimensione byte':len(raw)})
            obs=put('observations','raw-'+name,Nome='Task41m complete original JSON '+name,Descrizione='Schema preview only' if args.validate_only else raw.decode('utf-8'),**{'Artefatti dettagliati':[aid,art]})
            raw_reports.append(dict(member=name,observation=obs,sha256=sha,bytes=len(raw)))
    arts=[aid]
    for filename in ('scripts/audit_task41m_final.py','experiments/results/task41m_audited_v1/audit.json'):
        raw=(repo/filename).read_bytes();arts.append(put('artifacts',filename,Nome='Task41m '+filename,Tipo='Codice' if filename.endswith('.py') else 'Report',Versione='v1',Percorso=filename,**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)}))
    obs=put('observations','audit',Nome='Task41m complete paired/frozen audit',Descrizione=json.dumps(audit),**{'Artefatti dettagliati':arts})
    exp=ref('experiments-giada-task41m-frozen-contract-v1');limits='Opened diagnostic panels, not independent confirmation. Failed evaluations not unique biological cases; model weights frozen; no qualified speedup, threshold waiver or automaticTask42.'
    findings={k:put('findings',k,Nome='Task41m '+k,Esito='Negativo' if k=='cost' else 'Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=v,Limitazioni=limits) for k,v in STATEMENTS.items()}
    for name,key,outcome in [('selective_transfer','quality','Contraddice'),('numerical_contract','separation','Sostiene'),('reference_confound','reference','Sostiene'),('fp32_algebra','reference','Contraddice')]:
        put('evidence',name,Nome='Task41m scoped '+name,Esito=outcome,**{'Affermazione valutata':[ref('claims-giada-task41m-'+name+'-v1')],'Risultati a sostegno':[findings[key]],'Argomentazione':STATEMENTS[key],'Limiti e spiegazioni alternative':limits+' Selective-transfer/FP32 contradictions scoped to original-reference primary contract; successful combined FP64 diagnostic does not contradict original failure.','Data valutazione':datetime.now(timezone.utc).date().isoformat()})
    runs=[]
    spec=json.loads((repo/'experiments/task41m_preregistration_v1.json').read_text())
    for model in spec['models']:
        for policy in spec['policies']:
            code='runs-giada-task41m-'+model+'/'+policy+'-v1';runs.append(ref(code));m.local_upsert('runs',{'Codice stabile':code,'Stato':'Completata','Artefatti prodotti':[aid],'Validità tecnica':'Complete frozen evaluation,exit0,finite tensors,all canonical floors passed;source hashes/parity verified. No training or selection.'})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41m-frozen-contract-v1','Stato':'Concluso'})
    put('analyses','full',Nome='Task41m full frozen numerical/reference and cost audit',Origine='Post hoc',Versione='v1',Procedura='scripts/audit_task41m_final.py',Configurazione=limits,**{'Run analizzati':runs,'Input artefatti':arts,'12 · Risultati scientifici':list(findings.values())})
    put('decisions','confirm-independently',Nome='Preregister41n frozen independent confirmation and exact execution',Esito='Continuare',Risultati=list(findings.values()),Motivazione=STATEMENTS['decision'],**{'Data':datetime.now(timezone.utc).date().isoformat()})
    print('Committing64original reports plus scoped evidence; full verification follows',flush=True)
    change=m.commit_batch()
    if args.validate_only:return
    verified=m.verify_incremental(baseline);assert verified['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=verified,change=change,records=ids,raw_reports=raw_reports,airtable_accessed=False,DB_backup='local refresh with41n preregistration'))
    print('Complete41m evidence registered and globally verified',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Schema preview passed; no writes')
