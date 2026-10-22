"""Register complete recovered 41e results, caveats and proposed41f locally.

All JSON report rows are preserved and independently read back by hash.
No Airtable, Drive upload, retraining, threshold change or new launch.
"""
import hashlib
import json
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror
from .register_task41d_result import ResultPreview, PreviewComplete


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):sha.update(chunk)
    return sha.hexdigest()


def main():
    repo=ROOT.parent
    folder=repo/'experiments/results/task41e_audited_v1'
    ext=repo/'artifacts/task41e_reserve_external'
    s=json.loads((folder/'synthesis.json').read_text(encoding='utf-8'))
    audit=json.loads((ext/'final_detailed_audit.json').read_text())
    receipt=json.loads((ext/'final_recovery_receipt.json').read_text())
    archive=Path(receipt['local'])
    assert s['valid'] and not s['thresholds_changed'] and not s['next_experiment']['launched']
    assert receipt['valid'] and receipt['runtime_independent'] and receipt['crc_all_members_valid']
    assert receipt['finished_status']['worker_exit_code']==0 and receipt['finished_status']['state']=='finished'
    assert all(receipt['parameters_finite'].values()) and 'checkpoint_60000.pt' in receipt['parameters_finite']
    assert digest(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    preview='--validate-only' in sys.argv
    m=ResultPreview() if preview else ValidatedBatchMirror()
    ids={};arts={};observations=[];raw_manifest=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41e-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id']
        ids[code]=rid
        return rid
    exp=ref('experiments-giada-task41e-factorial-v1')
    archive_id=put('artifacts','archive',Nome='Task41e complete recovered final archive',Tipo='Report',Versione='v1-recovered',
        Percorso=archive.relative_to(repo).as_posix(),**{'SHA-256':receipt['sha256'],'Dimensione byte':receipt['bytes']})
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'))
        spec=json.loads(z.read('preregistration.json'))
        freeze=json.loads(z.read('selection_freeze.json'))
        assert report['valid'] and report['decision']=='TRANSFER_DIAGNOSIS_NO_GO' and not report['thresholds_changed']
        assert report['nominee']=='Q0' and not report['passing_arms'] and not freeze['opened_regressions_accessed']
        runs={f"{a['name']}-{seed}":ref(f"runs-giada-task41e-{a['name']}-{seed}-v1") for a in spec['arms'] for seed in spec['seeds']}
        for name in sorted(n for n in z.namelist() if n.endswith('.json')):
            raw=z.read(name)
            value=json.loads(raw)
            serialized=json.dumps(value,ensure_ascii=False,separators=(',',':'))
            kind=('cost' if 'cost' in name else 'diagnostics' if 'coupling_probe/' in name else
                'selection' if any(k in name for k in ('freeze','curves','contrasts')) else 'quality')
            art=put('artifacts','member-'+name,Nome='Task41e original '+name,Tipo='Report',Versione='v1',
                Percorso=archive.relative_to(repo).as_posix()+'#'+name,
                **{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
            oid=put('observations','raw-'+name,Nome='Task41e complete '+name,Descrizione=serialized,
                **{'Artefatti dettagliati':[art,archive_id],
                   'Specifica di valutazione':[ref(f'evaluations-giada-task41e-{kind}-v1')]})
            observations.append(oid)
            raw_manifest.append(dict(member=name,observation=oid,original_bytes=len(raw),
                canonical_json_sha256=hashlib.sha256(serialized.encode()).hexdigest()))
            print('Staged full original report '+name,flush=True)
    files=[ext/'final_detailed_audit.json',ext/'recovered_summary_audit.json',ext/'final_recovery_receipt.json',
        ext/'final_delta_manifest.json',folder/'residual_probe.json',folder/'synthesis.json',
        folder/'AUDIT.md',repo/'scripts/probe_task41e_residuals.py',repo/'scripts/synthesize_task41e_audit.py',
        repo/'tests/test_task41e_audit.py',repo/'experiments/results/task39c_kaggle_v1/result_audit.json',
        repo/'src/giada_teacher/task39c_dynamic_identification.py',repo/'src/giada_teacher/task41b_evaluation.py',
        repo/'scripts/finalize_task41e_recovery.py',repo/'scripts/audit_task41e_recovered_summary.py',
        repo/'scripts/audit_task41e_final.py',Path(__file__)]
    for path in files:
        raw=path.read_bytes()
        art=put('artifacts',path.name,Nome='Task41e audit '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',
            Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        arts[path.name]=art
        if path.suffix=='.json':
            oid=put('observations',path.name,Nome='Task41e evidence '+path.name,
                Descrizione=json.dumps(json.loads(raw),ensure_ascii=False,separators=(',',':')),
                **{'Artefatti dettagliati':[art]})
            observations.append(oid)
    for name,summary in report['summaries'].items():
        rr=[rid for key,rid in runs.items() if key.startswith(name+'-')]
        put('observations','summary-'+name,Nome='Task41e summary, failures and case transitions '+name,
            Descrizione=json.dumps(dict(summary=summary,failures=audit['failure_details'][name],
                regressions_vs_P110=audit['regressions_vs_P110'][name],contributing_runs=rr),ensure_ascii=False,separators=(',',':')),
            **{'Artefatti dettagliati':[arts['final_detailed_audit.json'],archive_id],
               'Specifica di valutazione':[ref('evaluations-giada-task41e-quality-v1')]})
    findings={key:put('findings',key,Nome='Task41e '+key,Esito='Negativo' if key=='outcome' else 'Misto',
        Esperimenti=[exp],Osservazioni=observations,Risultato=value,Limitazioni=s['limits']) for key,value in s['findings'].items()}
    put('analyses','residual-audit',Nome='Task41e full-result audit and frozen numerical/selection probes',Origine='Post hoc',Versione='v1',
        Procedura='audit_task41e_final.py; probe_task41e_residuals.py; synthesize_task41e_audit.py',
        Configurazione=json.dumps(dict(no_training=True,original_freeze_unchanged=True,new_nominee=False,thresholds_changed=False)),
        **{'Run analizzati':list(runs.values()),'Input artefatti':[archive_id]+list(arts.values()),
           '12 · Risultati scientifici':list(findings.values())})
    for e in s['original_claim_evidence']:
        code=f"claims-giada-task41d-result-{e['id']}-v1"
        m.local_upsert('claims',{'Codice stabile':code,'Stato':e['state'],'Limiti':e['argument']+' '+s['limits']})
        put('evidence','registered-'+e['id'],Nome='Task41e evidence for registered '+e['id'],Esito=e['outcome'],
            **{'Affermazione valutata':[ref(code)],'Risultati a sostegno':[findings[e['finding']]],
               'Argomentazione':e['argument'],'Limiti e spiegazioni alternative':s['limits'],'Data valutazione':'2026-10-22'})
    for h in s['hypotheses']:
        claim=put('claims',h['id'],Nome='Task41e residual hypothesis '+h['id'],Tipo='Ipotesi',Stato=h['status'],
            Enunciato=h['statement'],**{'Condizioni di falsificazione':h['falsifier']},Limiti=h['priority']+'; '+h['test']+' '+s['limits'])
        put('evidence',h['id'],Nome='Task41e scoped evidence '+h['id'],Esito='Indeterminata',
            **{'Affermazione valutata':[claim],'Risultati a sostegno':[findings[h['evidence']]],
               'Argomentazione':h['test'],'Limiti e spiegazioni alternative':h['falsifier']+' '+s['limits'],'Data valutazione':'2026-10-22'})
    lessons={
        'same_predicate':'Una soglia unica richiede un unico predicato numerico tra ranking e accettazione;0.3 e0.30000000000000004 non sono due livelli scientifici di errore. Testare confini prima del training.',
        'reference_contract':'Un probe deve riprodurre i fallimenti senza interventi prima di spiegarli. Cambiare riferimento,dtype o batch può cambiare risposte minuscole; non chiamare questo una riparazione.',
        'relaxation':'Correggere solo equilibrio non è equivalente a poter correggere il rilassamento; separare u,z e gate m/h con oracoli diagnostici prima di investire altro budget.',
        'scope_8x':'Conservare il risultato8x nel suo scope residente isolato. Misurare il trasferimento con pesi,fisica,batch e strategiaCUDAgraph appaiati,senza cancellare il progresso né estenderlo al neurone completo.',
        'backup':'StatoCLI/ContentsAPI fallito non dimostra worker morto. Verificare status finale e checkpoint da canale indipendente; ricostruzioneZIP percontenuti richiede inventario completo,CRC eSHA con distinzione dal containerremoto.',
    }
    for key,value in lessons.items():
        put('lessons',key,Nome='Task41e '+key,Lezione=value,Risultati=list(findings.values()),Limiti=s['limits'])
    put('lessons','memory-patterns',Nome='Task41e previous numerical and learnability patterns',
        Lezione=json.dumps(s['memory_links']),Risultati=[ref(k) for k in s['memory_links']]+list(findings.values()),
        Limiti='Analogies guide falsifiable tests; no automatic causal transfer from previous experiments.')
    decision=put('decisions','propose41f',Nome='Keep41e NO-GO; propose41f discriminating audit and repair',Esito='Continuare',
        Motivazione=json.dumps(s['next_experiment'],ensure_ascii=False),Risultati=list(findings.values()),**{'Data':'2026-10-22'})
    put('actions','propose41f',Nome='Task41f proposed, not launched',Stato='Proposta',
        Descrizione=json.dumps(s['next_experiment'],ensure_ascii=False),**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41e-factorial-v1','Stato':'Concluso',
        'Descrizione':s['findings']['outcome']+' '+s['findings']['quality']+' Technical audit caveats: '+s['findings']['selection']})
    for key in runs:
        m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41e-{key}-v1','Stato':'Completata',
            'Artefatti prodotti':[archive_id],
            'Validità tecnica':'60000updates,exit0,allrecoveredtensorsfinite,completeexternalfinalbackup. ScientificNO-GO. Originalmetricsunchanged. Posthocselection-boundaryandprobe-referencecaveatsregistered;notanindependentconfirmation.'})
    print(json.dumps(dict(stage='staged',records=len(ids),raw_reports=len(raw_manifest),completed_runs=len(runs))),flush=True)
    m.commit_batch()
    print('Committed complete41e result; exporting snapshot',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot())
    print('Verifying mirror and every raw report',flush=True)
    verification=m.verify()
    assert verification['valid']
    with m.connect() as con:
        for row in raw_manifest:
            stored=con.execute('SELECT Descrizione FROM observations WHERE _record_id=?',(row['observation'],)).fetchone()
            assert stored and hashlib.sha256(stored[0].encode()).hexdigest()==row['canonical_json_sha256'],row['member']
    count=m.query("SELECT count(*) n FROM runs WHERE [Codice stabile] LIKE 'runs-giada-task41e-%' AND Stato='Completata'")['rows'][0]['n']
    assert count==36
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=verification,raw_reports=raw_manifest,
        completed_runs=count,airtable_accessed=False,DB_backup='localonly',new_experiment_launched=False,
        task42_authorized=False,original_freeze_unchanged=True))
    print('Complete41e result, raw reports, audit and hypotheses verified in SQLite',flush=True)


if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Field contracts, cardinalities and references valid; no DB writes',flush=True)
