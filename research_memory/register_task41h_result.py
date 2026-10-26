"""Complete Task41h original JSON and source-linked conclusions in SQLite only.

Resume bounded transactions from a verified receipt; never train, select,
change scientific reports, or contact Airtable. Binary artifacts remain in the
verified complete ZIP. Each original JSON is retained exactly inside SQLite.
"""
import argparse
import gc
import hashlib
import json
import zipfile
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT, write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview, PreviewComplete


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('raw','complete'),required=True);ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    assert not (args.validate_only and args.phase=='raw')
    repo=ROOT.parent;folder=repo/'experiments/results/task41h_recovery_v1'
    complete=json.loads((folder/'audit_completion.json').read_text(encoding='utf-8'))
    assert complete['valid'] and complete['complete'] and not complete['thresholds_changed']
    external=repo/'artifacts/task41h_complete_evaluation_external'
    receipt=json.loads((external/'external_backup_receipt.json').read_text(encoding='utf-8'))
    archive=external/'recovery_final_merged.zip'
    assert archive.stat().st_size==complete['final_backup']['bytes']
    progress_path=folder/'registration_progress.json'
    progress=json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else dict(raw_reports=[],batches=[],records={})
    baseline=progress.get('verification',json.loads((repo/'experiments/results/task41h_protocol_v1/launch_memory_receipt.json').read_text(encoding='utf-8'))['verification'])
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror()
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    ids=progress['records']

    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        table={'finding':'findings','decision':'decisions'}.get(code.split('-')[0],code.split('-')[0])
        assert table in m.contract.tables
        rows=m.query('SELECT _record_id id FROM ['+table+"] WHERE [Codice stabile]='"+code.replace("'","''")+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['id']

    def put(table,suffix,**fields):
        code=f'{table}-giada-task41h-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid

    def flush():
        nonlocal m,baseline
        if args.validate_only:m.commit_batch();return
        print('Commit bounded batch',flush=True)
        change=m.commit_batch();v=m.verify_incremental(baseline);assert v['valid']
        baseline=v;progress['verification']=v;progress['batches'].append(change)
        write_json(progress_path,progress);m=IncrementalBatchMirror();gc.collect()
        print('Verified '+str(change['changed_records'])+' changed records; '+str(len(progress['raw_reports']))+' reports',flush=True)

    archive_id=put('artifacts','archive',Nome='Task41h complete immutable recovery archive',Tipo='Report',Versione='v1',
        Percorso=archive.relative_to(repo).as_posix(),**{'SHA-256':complete['final_backup']['sha256'],'Dimensione byte':archive.stat().st_size})
    if args.phase=='raw':
        digest=hashlib.sha256()
        with archive.open('rb') as stream:
            while block:=stream.read(8*1024*1024):digest.update(block)
        assert digest.hexdigest()==complete['final_backup']['sha256']
        done={x['member'] for x in progress['raw_reports']};staged=0
        with zipfile.ZipFile(archive) as z:
            members=sorted((x for x in z.infolist() if x.filename.endswith('.json')),key=lambda x:x.file_size)
            for member in members:
                name=member.filename
                if name in done:continue
                if staged and staged+member.file_size>120_000_000:flush();staged=0
                raw=z.read(name);sha=hashlib.sha256(raw).hexdigest()
                aid=put('artifacts','member-'+name,Nome='Task41h original '+name,Tipo='Report',Versione='v1',
                    Percorso=archive.relative_to(repo).as_posix()+'#'+name,**{'SHA-256':sha,'Dimensione byte':len(raw)})
                kind='cost' if 'cost' in name or 'execution_matrix' in name else 'diagnostics' if ('audit' in name or 'oracle' in name or 'probe' in name) else 'selection' if any(k in name for k in ('freeze','curves','contrasts','development_')) else 'quality'
                oid=put('observations','raw-'+name,Nome='Task41h complete original JSON '+name,Descrizione=raw.decode('utf-8'),
                    **{'Artefatti dettagliati':[aid,archive_id],'Specifica di valutazione':[ref(f'evaluations-giada-task41h-{kind}-v1')]})
                progress['raw_reports'].append(dict(member=name,observation=oid,sha256=sha,bytes=len(raw)))
                staged+=len(raw);del raw
                print('Staged '+name+' '+str(member.file_size),flush=True)
            if staged:flush()
            assert {x['member'] for x in progress['raw_reports']}=={x.filename for x in members}
        print('All original JSON reports registered',flush=True)
        return

    synthesis=json.loads((folder/'residual_hypotheses.json').read_text(encoding='utf-8'))
    comparison=json.loads((folder/'comparison_with_41g.json').read_text(encoding='utf-8'))
    spec=json.loads((repo/'experiments/task41h_preregistration_v1.json').read_text(encoding='utf-8'))
    assert synthesis['valid'] and not synthesis['next_experiment']['launched']
    if not args.validate_only:
        with zipfile.ZipFile(archive) as z:
            assert {x['member'] for x in progress['raw_reports']}=={x for x in z.namelist() if x.endswith('.json')}
    arts={};observations=[]
    files=[folder/n for n in ('audit_completion.json','comparison_with_41g.json','final_report.json','LATEST.json','memory_queries.json','memory_queries.sql','residual_hypotheses.json','ANALISI_RESIDUI.md','residual_probe.json')]
    files += [external/'external_backup_receipt.json',external/'job_status.json',repo/'scripts/query_task41h_memory.py',repo/'scripts/probe_task41h_residuals.py',repo/'scripts/synthesize_task41h_hypotheses.py',Path(__file__)]
    for path in files:
        raw=path.read_bytes();aid=put('artifacts',path.name,Nome='Task41h audit '+path.name,
            Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),
            **{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        arts[path.name]=aid
        if path.suffix=='.json':observations.append(put('observations',path.name,Nome='Task41h audit evidence '+path.name,Descrizione=raw.decode('utf-8'),**{'Artefatti dettagliati':[aid,archive_id]}))
    exp=ref('experiments-giada-task41h-factorial-v1')
    run_ids=[ref(f"runs-giada-task41h-{a['name']}-{seed}-v1") for a in spec['arms'] for seed in spec['seeds']]
    for name,summary in comparison['report']['summaries'].items():
        contributing=[ref(f'runs-giada-task41h-{name}-{seed}-v1') for seed in spec['seeds']] if name in {a['name'] for a in spec['arms']} else []
        observations.append(put('observations','summary-'+name,Nome='Task41h full model summary '+name,
            Descrizione=json.dumps(dict(summary=summary,failures=comparison['failures'].get(name),contributing_runs=contributing),ensure_ascii=False),
            **{'Artefatti dettagliati':[archive_id,arts['comparison_with_41g.json']],'Specifica di valutazione':[ref('evaluations-giada-task41h-quality-v1')]}))
    for group,rows in comparison['failures'][comparison['nominee']].items():
        for index,row in enumerate(rows):
            observations.append(put('observations','failure-'+group+'-'+str(index),Nome='Task41h remaining failure '+group+' '+str(index),
                Descrizione=json.dumps(row,ensure_ascii=False),**{'Artefatti dettagliati':[arts['comparison_with_41g.json']],'Specifica di valutazione':[ref('evaluations-giada-task41h-quality-v1')]}))
    findings={key:put('findings',key,Nome='Task41h '+key,Esito='Negativo' if key=='outcome' else 'Misto',
        Esperimenti=[exp],Osservazioni=observations,Risultato=value,Limitazioni=synthesis['limits']) for key,value in synthesis['findings'].items()}
    put('analyses','completion',Nome='Task41h complete audit and falsifiable residual analysis',Origine='Post hoc',Versione='v1',
        Procedura='query_task41h_memory.py;probe_task41h_residuals.py;synthesize_task41h_hypotheses.py',
        Configurazione=json.dumps(dict(thresholds_changed=False,new_training=False,new_selection=False)),
        **{'Run analizzati':run_ids,'Input artefatti':[archive_id]+list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for e in synthesis['original_claim_evidence']:
        # Keep historical statement and limits; append a dated, scoped assessment.
        claim=ref(e['code'])
        put('evidence','prior-'+e['id'],Nome='Task41h assessment '+e['id'],Esito=e['outcome'],
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[findings[e['finding']]],'Argomentazione':e['argument'],
            'Limiti e spiegazioni alternative':e['limits'],'Data valutazione':synthesis['date']})
    for h in synthesis['hypotheses']:
        claim=put('claims',h['id'],Nome=h['title'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['statement'],
            **{'Condizioni di falsificazione':h['falsifier']},Limiti=h['test']+' '+h['limits'])
        put('evidence',h['id'],Nome='Task41h scoped motivation '+h['id'],Esito='Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[findings[k] for k in h['evidence']],
            'Argomentazione':h['test'],'Limiti e spiegazioni alternative':h['limits'],'Data valutazione':synthesis['date']})
    for pattern in synthesis['patterns']:
        put('lessons',pattern['id'],Nome=pattern['title'],Lezione=json.dumps(pattern,ensure_ascii=False),
            Risultati=[ref(x) for x in pattern['sources']]+[findings[k] for k in pattern['current']],Limiti=pattern['limits'])
    decision=put('decisions','retain-no-go',Nome='Task41h NO-GO; propose selective residual tests',Esito='Continuare',
        Motivazione=json.dumps(synthesis['next_experiment'],ensure_ascii=False),Risultati=list(findings.values()),**{'Data':synthesis['date']})
    put('actions','proposed-next',Nome='Next residual experiment proposed, not launched',Stato='Proposta',
        Descrizione=json.dumps(synthesis['next_experiment'],ensure_ascii=False),**{'Decisioni di origine':[decision]})
    # The registered design remains historical evidence, not a mutable result
    # summary. Conclusions live in source-linked findings, not over that text.
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41h-factorial-v1','Stato':'Concluso'})
    for arm in spec['arms']:
        for seed in spec['seeds']:
            code=f"runs-giada-task41h-{arm['name']}-{seed}-v1"
            old=m.query("SELECT fields_json FROM _remote_records WHERE table_key='runs' AND record_id='"+ref(code)+"'")['rows']
            assert len(old)==1
            fields=json.loads(old[0]['fields_json'])
            artifact_field=next(f['id'] for f in m.contract.tables['runs']['fields'] if f['name']=='Artefatti prodotti')
            produced=list(dict.fromkeys(fields.get(artifact_field,[])+[archive_id,arts['final_report.json']]))
            m.local_upsert('runs',{'Codice stabile':code,'Stato':'Completata',
                'Artefatti prodotti':produced,
                'Validità tecnica':'60000 original updates; finite original checkpoints unchanged. Frozen selection recovered once before final evaluation-only request. Complete Colab/Kaggle evaluation recovered without training, reselection or threshold changes. Scientific NO-GO.'})
    flush()
    print('Readback of scientific records and evidence chains',flush=True)
    counts={}
    with m.connect() as con:
        for table,suffix,expected in [
            ('findings','',len(synthesis['findings'])),
            ('claims','',len(synthesis['hypotheses'])),
            ('evidence','',len(synthesis['hypotheses'])+len(synthesis['original_claim_evidence'])),
            ('lessons','',len(synthesis['patterns'])),
            ('observations','failure-',6),
            ('observations','summary-',len(comparison['report']['summaries'])),
        ]:
            prefix=f'{table}-giada-task41h-result-{suffix}'
            count=con.execute(f'SELECT count(*) FROM [{table}] WHERE [Codice stabile] LIKE ?', (prefix+'%',)).fetchone()[0]
            assert count==expected,(table,suffix,count,expected)
            counts[table+(':'+suffix if suffix else '')]=count
        assert con.execute('SELECT Stato FROM experiments WHERE _record_id=?',(exp,)).fetchone()[0]=='Concluso'
        for run in run_ids:
            assert con.execute('SELECT Stato FROM runs WHERE _record_id=?',(run,)).fetchone()[0]=='Completata'
        for target in ('claims','findings'):
            links=con.execute('SELECT sql_table FROM _mirror_relations WHERE source_table=? AND target_table=?',('evidence',target)).fetchall()
            assert len(links)==1
            sql_table=links[0][0]
            count=con.execute(f'SELECT count(DISTINCT l.source_id) FROM [{sql_table}] l JOIN evidence e ON e._record_id=l.source_id WHERE e.[Codice stabile] LIKE ?',('evidence-giada-task41h-result-%',)).fetchone()[0]
            assert count==counts['evidence'],(target,count)
            counts['evidence_linked_to_'+target]=count
    print('Readback of all original JSON byte hashes',flush=True)
    with m.connect() as con:
        for row in progress['raw_reports']:
            rid=con.execute('SELECT rowid FROM observations WHERE _record_id=?',(row['observation'],)).fetchone()[0]
            assert con.execute('SELECT json_valid(Descrizione) FROM observations WHERE rowid=?',(rid,)).fetchone()[0]==1,row['member']
            hh=hashlib.sha256()
            with con.blobopen('observations','Descrizione',rid,readonly=True) as blob:
                while block:=blob.read(1024*1024):hh.update(block)
            assert hh.hexdigest()==row['sha256'],row['member']
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=baseline,records=ids,raw_reports=progress['raw_reports'],
        raw_original_bytes_verified=True,raw_reports_json1_queryable=True,completed_runs=len(run_ids),final_archive_complete=True,airtable_accessed=False,
        new_experiment_launched=False,scientific_readback_counts=counts,DB_backup='local only; refreshed backup receipt required'))
    print('Exporting complete local JSON snapshot',flush=True)
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Task41h complete memory verified and exported',flush=True)


if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Schema/relations preview passed; no DB writes',flush=True)
