"""Complete Task41f results and discriminating hypotheses, local SQLite only."""
import hashlib,json,sys,zipfile
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41f_audited_v1';ext=repo/'artifacts/task41f_reserve_external'
    synthesis=json.loads((folder/'synthesis.json').read_text(encoding='utf-8'))
    audit=json.loads((ext/'final_detailed_audit.json').read_text())
    receipt=json.loads((ext/'final_recovery_receipt.json').read_text());archive=Path(receipt['local'])
    assert receipt['valid'] and receipt['runtime_independent'] and receipt['byte_identical_to_remote'] and receipt['crc_all_members_valid']
    assert receipt['all_training_updates_completed']==60000 and all(receipt['parameters_finite'].values())
    assert digest(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    assert synthesis['valid'] and not synthesis['thresholds_changed'] and not synthesis['next_experiment']['launched']
    m=ResultPreview() if '--validate-only' in sys.argv else ValidatedBatchMirror()
    ids={};arts={};obs=[];raw_manifest=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41f-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid
        return rid
    exp=ref('experiments-giada-task41f-factorial-v1')
    archive_id=put('artifacts','archive',Nome='Task41f complete byte-identical final backup',Tipo='Report',Versione='v1',
        Percorso=archive.relative_to(repo).as_posix(),**{'SHA-256':receipt['sha256'],'Dimensione byte':receipt['bytes']})
    with zipfile.ZipFile(archive) as z:
        report=json.loads(z.read('final_report.json'));spec=json.loads(z.read('preregistration.json'))
        assert report['valid'] and report['decision']=='PRECISION_RELAXATION_NO_GO' and not report['passing_arms']
        assert not report['thresholds_changed'] and not report['task42_authorized']
        runs={f"{a['name']}-{seed}":ref(f"runs-giada-task41f-{a['name']}-{seed}-v1") for a in spec['arms'] for seed in spec['seeds']}
        for name in sorted(n for n in z.namelist() if n.endswith('.json')):
            raw=z.read(name);serialized=json.dumps(json.loads(raw),ensure_ascii=False,separators=(',',':'))
            kind=('cost' if ('cost' in name or 'execution_matrix' in name) else 'diagnostics' if ('frozen_audit/' in name or 'probe' in name)
                else 'selection' if any(k in name for k in ('freeze','curves','contrasts')) else 'quality')
            art=put('artifacts','member-'+name,Nome='Task41f original '+name,Tipo='Report',Versione='v1',
                Percorso=archive.relative_to(repo).as_posix()+'#'+name,
                **{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
            oid=put('observations','raw-'+name,Nome='Task41f complete '+name,Descrizione=serialized,
                **{'Artefatti dettagliati':[art,archive_id],'Specifica di valutazione':[ref(f'evaluations-giada-task41f-{kind}-v1')]})
            obs.append(oid);raw_manifest.append(dict(member=name,observation=oid,original_bytes=len(raw),canonical_json_sha256=hashlib.sha256(serialized.encode()).hexdigest()))
            print('Staged full original report '+name,flush=True)
    files=[ext/'final_detailed_audit.json',ext/'final_recovery_receipt.json',ext/'remote_recovery_manifest.json',
        ext/'supervisor_receipt.json',ext/'comparison_41e_41f.json',folder/'synthesis.json',folder/'AUDIT.md',
        repo/'scripts/audit_task41f_final.py',repo/'scripts/synthesize_task41f_audit.py',repo/'tests/test_task41f_audit.py',Path(__file__)]
    for path in files:
        raw=path.read_bytes();art=put('artifacts',path.name,Nome='Task41f audit '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',
            Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        arts[path.name]=art
        if path.suffix=='.json':
            obs.append(put('observations',path.name,Nome='Task41f evidence '+path.name,
                Descrizione=json.dumps(json.loads(raw),ensure_ascii=False,separators=(',',':')),**{'Artefatti dettagliati':[art]}))
    for name,summary in report['summaries'].items():
        obs.append(put('observations','summary-'+name,Nome='Task41f summary and case transitions '+name,
            Descrizione=json.dumps(dict(summary=summary,failures=audit['failure_details'][name],
                comparisons={b:audit['regressions'][b][name] for b in audit['regressions']},
                contributing_runs=[rid for k,rid in runs.items() if k.startswith(name+'-')]),ensure_ascii=False,separators=(',',':')),
            **{'Artefatti dettagliati':[archive_id,arts['final_detailed_audit.json']],
               'Specifica di valutazione':[ref('evaluations-giada-task41f-quality-v1')]}))
    findings={key:put('findings',key,Nome='Task41f '+key,Esito='Negativo' if key=='outcome' else 'Misto',
        Esperimenti=[exp],Osservazioni=obs,Risultato=value,Limitazioni=synthesis['limits']) for key,value in synthesis['findings'].items()}
    put('analyses','full-audit',Nome='Task41f complete paired audit and causal diagnostic probes',Origine='Post hoc',Versione='v1',
        Procedura='audit_task41f_final.py; synthesize_task41f_audit.py',Configurazione=json.dumps(dict(thresholds_changed=False,retraining=False,new_selection=False)),
        **{'Run analizzati':list(runs.values()),'Input artefatti':[archive_id]+list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for e in synthesis['original_claim_evidence']:
        code=f"claims-giada-task41e-result-{e['id']}-v1"
        m.local_upsert('claims',{'Codice stabile':code,'Stato':e['state'],'Limiti':e['argument']+' '+synthesis['limits']})
        put('evidence','registered-'+e['id'],Nome='Task41f evidence for '+e['id'],Esito=e['outcome'],
            **{'Affermazione valutata':[ref(code)],'Risultati a sostegno':[findings[e['finding']]],
               'Argomentazione':e['argument'],'Limiti e spiegazioni alternative':synthesis['limits'],'Data valutazione':'2026-10-22'})
    for h in synthesis['hypotheses']:
        claim=put('claims',h['id'],Nome='Task41f residual hypothesis '+h['id'],Tipo='Ipotesi',Stato='Aperta',
            Enunciato=h['statement'],**{'Condizioni di falsificazione':h['falsifier']},Limiti=h['test']+' '+synthesis['limits'])
        put('evidence',h['id'],Nome='Task41f scoped evidence '+h['id'],Esito='Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[findings[h['evidence']]],'Argomentazione':h['test'],
               'Limiti e spiegazioni alternative':h['falsifier'],'Data valutazione':'2026-10-22'})
    for key,text in synthesis['lessons'].items():
        put('lessons',key,Nome='Task41f '+key,Lezione=text,Risultati=list(findings.values()),Limiti=synthesis['limits'])
    decision=put('decisions','propose41g',Nome='Keep41f NO-GO; propose targeted41g',Esito='Continuare',
        Motivazione=json.dumps(synthesis['next_experiment'],ensure_ascii=False),Risultati=list(findings.values()),**{'Data':'2026-10-22'})
    put('actions','propose41g',Nome='Task41g proposed, not launched',Stato='Proposta',
        Descrizione=json.dumps(synthesis['next_experiment'],ensure_ascii=False),**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41f-factorial-v1','Stato':'Concluso','Descrizione':synthesis['findings']['outcome']+' '+synthesis['findings']['quality']})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task41f-launch-v1','Stato':'Conclusa','Descrizione':'Completed60000updates;finalarchiveSHA/CRCverifiedlocally;scientificNO-GO;resultauditregistered.'})
    for key in runs:
        m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41f-{key}-v1','Stato':'Completata','Artefatti prodotti':[archive_id],
            'Validità tecnica':'60000updates;allrecoveredTorchfilesfinite;freezeandfullfinalbackupverified;scientificNO-GO;thresholdsunchanged;notindependentconfirmation.'})
    print(json.dumps(dict(stage='staged',records=len(ids),raw_reports=len(raw_manifest),completed_runs=len(runs))),flush=True)
    m.commit_batch();print('Committed41f; exporting snapshot',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot())
    print('Verifying mirror and full raw JSON readback',flush=True)
    verification=m.verify();assert verification['valid']
    with m.connect() as con:
        for r in raw_manifest:
            row=con.execute('SELECT Descrizione FROM observations WHERE _record_id=?',(r['observation'],)).fetchone()
            assert row and hashlib.sha256(row[0].encode()).hexdigest()==r['canonical_json_sha256'],r['member']
    count=m.query("SELECT count(*) n FROM runs WHERE [Codice stabile] LIKE 'runs-giada-task41f-%' AND Stato='Completata'")['rows'][0]['n'];assert count==36
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=verification,raw_reports=raw_manifest,completed_runs=count,
        airtable_accessed=False,DB_backup='localonly',new_experiment_launched=False,task42_authorized=False,original_freeze_unchanged=True))
    print('Complete41f result, raw evidence and hypotheses verified inSQLite',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Field contracts, cardinalities and references valid; no DB writes',flush=True)
