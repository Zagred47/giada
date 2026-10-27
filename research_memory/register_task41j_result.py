"""Register every original41j JSON and the verified NO-GO; SQLite only."""
import argparse,gc,hashlib,json,zipfile
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror


def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41j_audited_v1'
    audit=json.loads((folder/'comparison_with_H.json').read_text());assert audit['valid'] and audit['complete']
    assert audit['report']['training_updates_performed']==0 and not audit['thresholds_changed']
    archive=Path(audit['backup']['local'])
    with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==audit['backup']['remote']['sha256']
    progress_path=folder/'registration_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(raw=[],records={},batches=[])
    if progress.get('complete'):print('Task41j result already registered');return
    baseline=progress.get('verification',json.loads((repo/'experiments/results/task41j_protocol_v1/launch_memory_receipt.json').read_text())['verification'])
    m=IncrementalBatchMirror();ids=progress['records'];assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41j-result-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def flush():
        nonlocal m,baseline
        print('Task41j committing verified bounded result batch',flush=True)
        change=m.commit_batch();baseline=m.verify_incremental(baseline);assert baseline['valid']
        progress.update(verification=baseline);progress['batches'].append(change);write_json(progress_path,progress)
        m=IncrementalBatchMirror();gc.collect();print('Task41j original reports registered: '+str(len(progress['raw'])),flush=True)
    aid=put('artifacts','archive',Nome='Task41j complete immutable final recovery',Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix(),
            **{'SHA-256':audit['backup']['remote']['sha256'],'Dimensione byte':archive.stat().st_size})
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;done={row['member'] for row in progress['raw']};staged=0
        members=sorted((x for x in z.infolist() if x.filename.endswith('.json')),key=lambda x:x.file_size)
        for member in members:
            if member.filename in done:continue
            if staged and staged+member.file_size>420_000_000:flush();staged=0
            raw=z.read(member);name=member.filename;digest=hashlib.sha256(raw).hexdigest()
            art=put('artifacts','member-'+name,Nome='Task41j original '+name,Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix()+'#'+name,
                    **{'SHA-256':digest,'Dimensione byte':len(raw)})
            obs=put('observations','raw-'+name,Nome='Task41j complete original JSON '+name,Descrizione=raw.decode('utf-8'),**{'Artefatti dettagliati':[aid,art]})
            progress['raw'].append(dict(member=name,observation=obs,sha256=digest,bytes=len(raw)));staged+=len(raw);del raw
            print('Task41j staged '+name+' '+str(member.file_size),flush=True)
        if staged:flush()
        assert {r['member'] for r in progress['raw']}=={x.filename for x in members}
    obs=put('observations','paired-audit',Nome='Task41j same-panel full audit',Descrizione=json.dumps(audit,ensure_ascii=False),**{'Artefatti dettagliati':[aid]})
    exp=ref('experiments-giada-task41j-comparison-v1')
    findings=[]
    for suffix,title,statement in [
        ('outcome','41j scientific NO-GO','Matched failed evaluations: H6,chain6,compact64 435,compact128 477,restricted8. No model qualifies; Task42 remains unauthorized.'),
        ('selection','Selection rejected later joint/compact refinement','All chain/compact seeds select step0; compact step0 follows4096 distillation updates. Training changed parameters but later checkpoints lost the registered comparison. Do not infer no training from identical frozen H.'),
        ('cost','Compact operators are faster but scientifically unqualified','Same-panel642 per seed ionic median ms: formula69.2163,H82.231,chain82.633,compact64 17.125,compact128 17.929,restricted100.505. Integrated formula104.653,H125.773,chain125.923,compact64 52.548,compact128 53.401,restricted144.090. No qualified speedup.'),
        ('objective','Coefficient improvement conflicts with observed development dynamics','Chain point loss decreases from approx0.024 to0.009 while auxiliary dynamics grows from0.00005-0.0002 to0.096-0.147. Development failures at60000 total25 versus3 initially. Supports testing objective alignment, does not establish unique cause; separate repair from compression.')]:
        findings.append(put('findings',suffix,Nome=title,Esito='Negativo' if suffix=='outcome' else 'Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=statement,Limitazioni='Opened regression diagnosis, not independent confirmation. Thresholds unchanged; original training and frozen selection retained.'))
    put('decisions','retain-H',Nome='Retain H; separate repair and compression in41k',Esito='Continuare',Risultati=findings,Motivazione='No41j arm promoted. User approved41k at same thresholds and abstraction; investigate conflicting objectives,retention,coverage and pure compression.')
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41j-comparison-v1','Stato':'Concluso'})
    spec=json.loads((repo/'experiments/task41j_preregistration_v1.json').read_text())
    for arm in spec['arms']:
        for seed in spec['seeds']:
            m.local_upsert('runs',{'Codice stabile':f"runs-giada-task41j-{arm['name']}-{seed}-v1",'Stato':'Completata',
                'Artefatti prodotti':[aid],'ValiditÃ  tecnica':'60000 finite canonical updates after4096 distillation; complete full-suite frozen evaluation. Scientific OPERATOR_RESET_NO_GO.'})
    flush();progress.update(complete=True);write_json(progress_path,progress)
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=baseline,records=ids,raw_reports=progress['raw'],
        raw_original_bytes_verified=True,completed_runs=12,airtable_accessed=False,DB_backup='local refresh pending'))
    print('Task41j all original JSON and source-linked result registered',flush=True)


if __name__=='__main__':main()
