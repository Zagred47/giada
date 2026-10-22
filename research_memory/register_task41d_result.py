"""Complete 41d results, typed audit evidence and proposed transfer contrasts.

SQLite only. Original reports are stored without dropping rows; numerical arrays
remain in the SHA/CRC-verified external archive. No experiment is launched here.
"""
import hashlib, json, sys, zipfile
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT, write_json
from .validated_batch import ValidatedBatchMirror
from .register_task41d import Preview, PreviewComplete


class ResultPreview(Preview):
    def local_upsert(self, table, fields):
        patch = self.contract.normalize(table, fields, partial=True)
        for field_id, value in patch.items():
            if field_id in self.contract.link_fields:
                link, reverse = self.contract.link_fields[field_id]
                if link['single'] and not reverse and len(value) > 1:
                    raise ValueError(f'Single-target link violated: {table}/{field_id}')
        return super().local_upsert(table, fields)


def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41d_audited_v1';external=repo/'artifacts/task41d_primary_external'
    s=json.loads((folder/'synthesis.json').read_text());audit=json.loads((external/'detailed_audit.json').read_text())
    assert s['valid'] and not s['thresholds_changed'] and not s['next_experiment']['task42_authorized']
    receipt=json.loads((external/'external_backup_receipt.json').read_text());archive=Path(receipt['local'])
    assert receipt['valid'] and receipt['runtime_independent'] and receipt['remote']['label']=='final'
    sha=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
    assert sha.hexdigest()==receipt['remote']['sha256']==audit['backup']['sha256']
    assert archive.stat().st_size==audit['backup']['bytes']
    preview='--validate-only' in sys.argv
    m=ResultPreview() if preview else ValidatedBatchMirror();ids={};arts={};observations=[];raw_manifest=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41d-result-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=ref('experiments-giada-task41d-factorial-v1')
    runs={a+'-'+str(seed):ref(f'runs-giada-task41d-{a}-{seed}-v1') for a in ('C00','C01','C10','C11','N000','N001','N010','N011','N100','N101','N110','N111') for seed in (17,29,43)}
    archive_id=put('artifacts','archive',Nome='Task41d final complete archive outside Colab',Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix(),**{'SHA-256':sha.hexdigest(),'Dimensione byte':archive.stat().st_size})
    observation_ids={}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'));assert report['valid'] and not report['thresholds_changed'] and not report['passing_arms']
        for name in sorted(n for n in z.namelist() if n.endswith('.json')):
            content=z.read(name);value=json.loads(content);serialized=json.dumps(value,ensure_ascii=False,separators=(',',':'))
            kind='cost' if 'cost' in name else ('selection' if ('freeze' in name or 'curves' in name or 'contrasts' in name) else 'quality')
            art=put('artifacts','member-'+name,Nome='Task41d archive member '+name,Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix()+'#'+name,**{'SHA-256':hashlib.sha256(content).hexdigest(),'Dimensione byte':len(content)})
            oid=put('observations','raw-'+name,Nome='Task41d full '+name,Descrizione=serialized,**{'Artefatti dettagliati':[art,archive_id],'Specifica di valutazione':[ref(f'evaluations-giada-task41d-{kind}-v1')]})
            observations.append(oid);observation_ids[name]=oid
            raw_manifest.append(dict(member=name,observation=oid,bytes=len(content),canonical_json_sha256=hashlib.sha256(serialized.encode()).hexdigest()))
            print('Staged raw report '+name,flush=True)
    files=[external/'detailed_audit.json',external/'external_backup_receipt.json',external/'supervisor_receipt.json',
           external/'job_status.json',external/'freeze_recovery_receipt.json',folder/'transfer_probe.json',folder/'synthesis.json',
           repo/'scripts/audit_task41d_result.py',repo/'scripts/probe_task41d_transfer.py',repo/'scripts/synthesize_task41d_audit.py',Path(__file__)]
    for path in files:
        data=path.read_bytes();art=put('artifacts',path.name,Nome='Task41d '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(data).hexdigest(),'Dimensione byte':len(data)})
        arts[path.name]=art
        if path.suffix=='.json':
            oid=put('observations',path.name,Nome='Task41d audit '+path.name,Descrizione=json.dumps(json.loads(data),ensure_ascii=False,separators=(',',':')),**{'Artefatti dettagliati':[art]});observations.append(oid)
    for name,summary in report['summaries'].items():
        rr=[rid for key,rid in runs.items() if key.startswith(name+'-')]
        # An aggregate is not a single-run observation. Keep its contributing
        # run IDs explicitly; the analysis below relates all runs to findings.
        put('observations','summary-'+name,Nome='Task41d aggregate and failures '+name,Descrizione=json.dumps(dict(summary=summary,failures=audit['failure_details'][name],contributing_runs=rr),ensure_ascii=False,separators=(',',':')),**{'Artefatti dettagliati':[arts['detailed_audit.json'],archive_id],'Specifica di valutazione':[ref('evaluations-giada-task41d-quality-v1')]})
    findings={key:put('findings',key,Nome='Task41d '+key,Esito='Negativo' if key=='outcome' else 'Misto',Esperimenti=[exp],Osservazioni=observations,Risultato=value,Limitazioni=s['limits']) for key,value in s['findings'].items()}
    put('analyses','transfer',Nome='Task41d current composition, objective geometry and gradient audit',Origine='Post hoc',Versione='v1',Procedura='audit_task41d_result.py; probe_task41d_transfer.py; synthesize_task41d_audit.py',Configurazione=json.dumps(dict(scope=s['limits'],read_only=True,no_training=True)),**{'Run analizzati':list(runs.values()),'Input artefatti':[archive_id]+list(arts.values()),'12 · Risultati scientifici':list(findings.values())})
    for e in s['original_claim_evidence']:
        code=f"claims-giada-task41d-H{e['id']}-v1"
        m.local_upsert('claims',{'Codice stabile':code,'Stato':e['state'],'Limiti':e['argument']+' '+s['limits']})
        put('evidence','original-H'+str(e['id']),Nome='Task41d registered H'+str(e['id']),Esito=e['outcome'],**{'Affermazione valutata':[ref(code)],'Risultati a sostegno':[findings[e['finding']]],'Argomentazione':e['argument'],'Limiti e spiegazioni alternative':s['limits'],'Data valutazione':'2026-10-22'})
    for h in s['hypotheses']:
        claim=put('claims',h['id'],Nome='Task41d transfer hypothesis '+h['id'],Tipo='Ipotesi',Stato=h['status'],Enunciato=h['statement'],**{'Condizioni di falsificazione':h['falsifier']},Limiti=h['test']+' '+s['limits'])
        put('evidence',h['id'],Nome='Task41d scoped evidence '+h['id'],Esito='Indeterminata',**{'Affermazione valutata':[claim],'Risultati a sostegno':[findings[h['evidence']]],'Argomentazione':'Motivazione empirica disponibile; causalità da distinguere. '+h['test'],'Limiti e spiegazioni alternative':h['falsifier']+' '+s['limits'],'Data valutazione':'2026-10-22'})
    for key,lesson in {
        'objective':'Verificare che trasformazioni/clamp delle etichette restino coerenti con la precisione e con la correzione massima rappresentabile; ispezionare loss e gradienti per regime, non solo media.',
        'coupling':'Registrare errori con segno e identità dei casi. Due correzioni singolarmente accettabili possono comporsi male; distinguere V/Ca, arrotondamento e sensibilità prima di attribuire il fallimento a capacità.',
        'gradient_balance':'La normalizzazione alla sola inizializzazione non garantisce un bilanciamento durante il training. Misurare le norme pesate, i coseni e gli effetti degli aggiornamenti nel tempo.',
        'selection':'Separare metriche diagnostiche del bersaglio dai gate globali di accettazione: il massimo su canali immutati può nascondere progressi. Tutti i gate restano richiesti per il GO.',
        'cost':'Comporre correttori riusando un unico calcolo del padre, previo controllo di equivalenza. Una composizione diagnostica con tre chiamate non dimostra un limite fondamentale dell astrazione.',
    }.items():put('lessons',key,Nome='Task41d '+key,Lezione=lesson,Risultati=list(findings.values()),Limiti=s['limits'])
    put('lessons','historical-patterns',Nome='Task41d links to earlier selection and precision patterns',Lezione=json.dumps(s['memory_links']),Risultati=[ref(code) for code in s['memory_links']]+list(findings.values()),Limiti='Collegamenti documentali e analogie, non prova automatica di identità causale.')
    decision=put('decisions','propose41e',Nome='Retain41d NO-GO; propose41e transfer and learnability tests',Esito='Continuare',Motivazione=json.dumps(s['next_experiment'],ensure_ascii=False),Risultati=list(findings.values()),**{'Data':'2026-10-22'})
    put('actions','propose41e',Nome='Task41e proposed shared-control notebook, not launched',Stato='Proposta',Descrizione=json.dumps(s['next_experiment'],ensure_ascii=False),**{'Decisioni di origine':[decision]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41d-factorial-v1','Stato':'Concluso','Descrizione':s['findings']['outcome']+' '+s['findings']['quality']})
    for key,rid in runs.items():
        m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41d-{key}-v1','Stato':'Completata','Artefatti prodotti':[archive_id],
                             'Validità tecnica':'Exit0,60k updates, finite frozen weights and checkpoints, valid numerical floors. Complete raw reports and archive preserved. Scientific NO-GO; thresholds unchanged.'})
    print(json.dumps(dict(stage='staged',records=len(ids),raw_reports=len(raw_manifest))),flush=True)
    m.commit_batch();print('Committed Task41d result and audit; exporting and verifying',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot())
    verification=m.verify();assert verification['valid']
    # Independent readback of every raw observation, including large reports.
    with m.connect() as con:
        for item in raw_manifest:
            row=con.execute('SELECT Descrizione FROM observations WHERE _record_id=?',(item['observation'],)).fetchone()
            assert row and hashlib.sha256(row[0].encode()).hexdigest()==item['canonical_json_sha256'],item['member']
    count=m.query("SELECT count(*) n FROM runs WHERE [Codice stabile] LIKE 'runs-giada-task41d-%' AND Stato='Completata'")['rows'][0]['n'];assert count==36
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=verification,raw_reports=raw_manifest,completed_runs=count,airtable_accessed=False,DB_backup='localonly',new_experiment_launched=False,task42_authorized=False))
    print('Task41d complete audit, raw reports, hypotheses and links registered and verified',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('All field contracts and existing references valid; no DB writes',flush=True)
