"""Bounded batches of complete available 41g JSON, linked findings and history.

No Airtable calls, no scientific mutation, no experiment launch. Original raw
JSON text is retained byte-for-byte (UTF-8) and queried with SQLite JSON1.
"""
import gc,hashlib,json,sys,zipfile
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41g_audited_v1';ext=repo/'artifacts/task41g_primary_external'
    inv=json.loads((folder/'evidence_inventory.json').read_text());s=json.loads((folder/'synthesis.json').read_text())
    report=json.loads((folder/'final_report_recovered.json').read_text());spec=json.loads((repo/'experiments/task41g_preregistration_v1.json').read_text())
    assert inv['valid'] and inv['all_saved_tensors_finite'] and inv['updates']==60000
    assert not s['thresholds_changed'] and not s['next_experiment']['launched']
    preview='--validate-only' in sys.argv
    progress_path=folder/'registration_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() and not preview else dict(raw_reports=[],batches=[])
    baseline=progress.get('verification',json.loads((repo/'experiments/results/task41g_protocol_v1/launch_memory_receipt.json').read_text())['verification'])
    m=ResultPreview() if preview else IncrementalBatchMirror();ids={};arts={};compact=[]
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code.replace("'","''")+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41g-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def flush():
        nonlocal m,baseline
        if preview:return
        print('Committing bounded raw batch',flush=True)
        change=m.commit_batch();v=m.verify_incremental(baseline);assert v['valid']
        baseline=v;progress['verification']=v;progress['batches'].append(change)
        write_json(progress_path,progress);m=IncrementalBatchMirror();gc.collect()
        print('Batch verified '+str(change['changed_records']),flush=True)
    archive=Path(inv['archive']['local'])
    archive_id=put('artifacts','archive',Nome='Task41g verified intermediate archive, not complete final container',Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix(),**{'SHA-256':inv['archive']['remote']['sha256'],'Dimensione byte':archive.stat().st_size})
    done={x['member'] for x in progress['raw_reports']};staged_bytes=0
    with zipfile.ZipFile(archive) as z:
        for r in inv['raw_json_reports']:
            if r['member'] in done:continue
            if staged_bytes and staged_bytes+r['bytes']>60_000_000:flush();staged_bytes=0
            raw=b'{}' if preview else z.read(r['member']) if r['source']=='archive' else Path(r['path']).read_bytes()
            if not preview:assert hashlib.sha256(raw).hexdigest()==r['sha256']
            location=archive.relative_to(repo).as_posix()+'#'+r['member'] if r['source']=='archive' else Path(r['path']).relative_to(repo).as_posix()
            a=put('artifacts','member-'+r['member'],Nome='Task41g original '+r['member'],Tipo='Report',Versione='v1',Percorso=location,**{'SHA-256':r['sha256'],'Dimensione byte':r['bytes']})
            kind='cost' if 'cost' in r['member'] or 'execution_matrix' in r['member'] else 'diagnostics' if 'frozen_audit' in r['member'] or 'probe' in r['member'] else 'selection' if any(k in r['member'] for k in ('freeze','curves','contrasts')) else 'quality'
            oid=put('observations','raw-'+r['member'],Nome='Task41g complete original JSON '+r['member'],Descrizione=raw.decode('utf-8'),**{'Artefatti dettagliati':[a,archive_id] if r['source']=='archive' else [a], 'Specifica di valutazione':[ref(f'evaluations-giada-task41g-{kind}-v1')]})
            progress['raw_reports'].append(dict(member=r['member'],observation=oid,sha256=r['sha256'],bytes=r['bytes'],source=r['source']))
            staged_bytes+=r['bytes'];del raw
            print('Registered original '+r['member']+' '+str(r['bytes']),flush=True)
        if staged_bytes:flush()
    exp=ref('experiments-giada-task41g-factorial-v1')
    run_ids=[ref(f"runs-giada-task41g-{a['name']}-{seed}-v1") for a in spec['arms'] for seed in spec['seeds']]
    files=[folder/n for n in ('evidence_inventory.json','final_report_recovered.json','synthesis.json','AUDIT.md','memory_queries.json','memory_queries.sql')]
    files += [ext/n for n in ('partial_scientific_audit.json','job_status.json','salvage_receipt.json','external_backup_receipt.json','remote_recovery_manifest.json')]
    files += [repo/'scripts'/n for n in ('query_task41g_memory.py','assemble_task41g_evidence.py','synthesize_task41g_memory.py','salvage_task41g_partial_zip.py')]+[Path(__file__)]
    for path in files:
        raw=path.read_bytes();a=put('artifacts',path.name,Nome='Task41g audit '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        arts[path.name]=a
        if path.suffix=='.json':compact.append(put('observations',path.name,Nome='Task41g audit evidence '+path.name,Descrizione=raw.decode('utf-8'),**{'Artefatti dettagliati':[a]}))
    audit=json.loads((ext/'partial_scientific_audit.json').read_text())
    for name,summary in report['summaries'].items():
        compact.append(put('observations','summary-'+name,Nome='Task41g aggregate and exact failure identities '+name,Descrizione=json.dumps(dict(summary=summary,failures=audit['failures'].get(name),changes_vs_41f=audit['deltas_vs_41f_nominee'].get(name)),ensure_ascii=False),**{'Specifica di valutazione':[ref('evaluations-giada-task41g-quality-v1')],'Artefatti dettagliati':[arts['partial_scientific_audit.json'],arts['final_report_recovered.json']]}))
    findings={k:put('findings',k,Nome='Task41g '+k,Esito='Negativo' if k=='outcome' else 'Misto',Esperimenti=[exp],Osservazioni=compact,Risultato=v,Limitazioni=s['limits']) for k,v in s['findings'].items()}
    put('analyses','audit',Nome='Task41g complete available reports, precision, cost and historical SQL exploration',Origine='Post hoc',Versione='v1',Procedura='assemble_task41g_evidence.py;query_task41g_memory.py;synthesize_task41g_memory.py',Configurazione=json.dumps(dict(thresholds_changed=False,retraining=False,new_experiment_launched=False,final_archive_complete=False)),**{'Run analizzati':run_ids,'Input artefatti':[archive_id]+list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for e in s['original_claim_evidence']:
        code='claims-giada-task41f-result-'+e['id']+'-v1';argument=s['findings'][e['finding']]
        m.local_upsert('claims',{'Codice stabile':code,'Stato':e['state'],'Limiti':argument+' '+s['limits']})
        put('evidence','registered-'+e['id'],Nome='Task41g scoped assessment '+e['id'],Esito=e['outcome'],**{'Affermazione valutata':[ref(code)],'Risultati a sostegno':[findings[e['finding']]],'Argomentazione':argument,'Limiti e spiegazioni alternative':s['limits'],'Data valutazione':'2026-10-23'})
    for h in s['hypotheses']:
        claim=put('claims',h['id'],Nome='Task41g residual hypothesis '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['statement'],**{'Condizioni di falsificazione':h['falsifier']},Limiti=h['test']+' '+s['limits'])
        put('evidence',h['id'],Nome='Task41g hypothesis motivation '+h['id'],Esito='Indeterminata',**{'Affermazione valutata':[claim],'Risultati a sostegno':[findings[h['evidence']]],'Argomentazione':h['test'],'Limiti e spiegazioni alternative':h['falsifier'],'Data valutazione':'2026-10-23'})
    for pattern in s['patterns']:
        historical=[ref(code) for code in pattern['sources'] if code.startswith('findings-')]
        put('lessons',pattern['id'],Nome='Task41g recurring pattern '+pattern['id'],Lezione=json.dumps(pattern,ensure_ascii=False),Risultati=historical+list(findings.values()),Limiti=pattern['transfer']+' '+s['limits'])
    decision=put('decisions','propose41h',Nome='Retain41g NO-GO; propose targeted41h diagnosis and repair',Esito='Continuare',Motivazione=json.dumps(s['next_experiment'],ensure_ascii=False),Risultati=list(findings.values()),**{'Data':'2026-10-23'})
    put('actions','propose41h',Nome='Task41h proposed, not launched',Stato='Proposta',Descrizione=json.dumps(s['next_experiment'],ensure_ascii=False),**{'Decisioni di origine':[decision]})
    put('actions','final-container-recovery',Nome='Task41g final ZIP container recovery outstanding',Stato='Proposta',Descrizione='Final report and precision cost recovered with memberCRC;latest complete intermediate archive and checkpoints verified. Exact remote final container SHA unavailable locally;runtime assignment absent. Do not rerun training or label partialZIP complete.',**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41g-factorial-v1','Stato':'Concluso','Descrizione':s['findings']['outcome']+' '+s['limits']})
    for a in spec['arms']:
        for seed in spec['seeds']:m.local_upsert('runs',{'Codice stabile':f"runs-giada-task41g-{a['name']}-{seed}-v1",'Stato':'Completata','Artefatti prodotti':[archive_id,arts['final_report_recovered.json']], 'Validità tecnica':'60000updates;allsevenavailablecheckpointsfinite;freezeSHAverified;finalreportCRCvalid;scientificNO-GO;finalarchivecontainerincomplete.'})
    if preview:m.commit_batch();return
    flush()
    print('Verifying all65 original JSON byte hashes and preserved41f',flush=True)
    with m.connect() as con:
        for r in progress['raw_reports']:
            rowid=con.execute('SELECT rowid FROM observations WHERE _record_id=?',(r['observation'],)).fetchone()[0]
            digest=hashlib.sha256()
            with con.blobopen('observations','Descrizione',rowid,readonly=True) as blob:
                while block:=blob.read(1024*1024):digest.update(block)
            assert digest.hexdigest()==r['sha256'],r['member']
        old=json.loads((repo/'experiments/results/task41f_audited_v1/memory_receipt.json').read_text())
        for r in old['raw_reports']:
            rowid=con.execute('SELECT rowid FROM observations WHERE _record_id=?',(r['observation'],)).fetchone()[0]
            digest=hashlib.sha256()
            with con.blobopen('observations','Descrizione',rowid,readonly=True) as blob:
                while block:=blob.read(1024*1024):digest.update(block)
            assert digest.hexdigest()==r['canonical_json_sha256'],r['member']
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=baseline,raw_reports=progress['raw_reports'],records=ids,
        raw_original_bytes_verified=True,prior41f_raw_reports_preserved=len(old['raw_reports']),completed_runs=36,airtable_accessed=False,DB_backup='localonly',new_experiment_launched=False,final_archive_complete=False))
    print('Task41g evidence and six hypotheses verified in SQLite',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Preview passed; no DB writes')
