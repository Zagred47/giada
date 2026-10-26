"""Register every original41i JSON and the verified NO-GO; SQLite only."""
import argparse,gc,hashlib,json,zipfile
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror


def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41i_recovery_v1'
    audit=json.loads((folder/'comparison_with_41h.json').read_text());assert audit['valid'] and audit['complete']
    assert audit['training_updates_performed']==0 and not audit['thresholds_changed']
    archive=Path(audit['backup']['local'])
    with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==audit['backup']['remote']['sha256']
    progress_path=folder/'registration_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(raw=[],records={},batches=[])
    if progress.get('complete'):print('Task41i result already registered');return
    baseline=progress.get('verification',json.loads((repo/'experiments/results/task41i_protocol_v1/launch_memory_receipt.json').read_text())['verification'])
    m=IncrementalBatchMirror();ids=progress['records'];assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41i-result-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def flush():
        nonlocal m,baseline
        print('Task41i committing verified bounded result batch',flush=True)
        change=m.commit_batch();baseline=m.verify_incremental(baseline);assert baseline['valid']
        progress.update(verification=baseline);progress['batches'].append(change);write_json(progress_path,progress)
        m=IncrementalBatchMirror();gc.collect();print('Task41i original reports registered: '+str(len(progress['raw'])),flush=True)
    aid=put('artifacts','archive',Nome='Task41i complete immutable final recovery',Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix(),
            **{'SHA-256':audit['backup']['remote']['sha256'],'Dimensione byte':archive.stat().st_size})
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;done={row['member'] for row in progress['raw']};staged=0
        members=sorted((x for x in z.infolist() if x.filename.endswith('.json')),key=lambda x:x.file_size)
        for member in members:
            if member.filename in done:continue
            if staged and staged+member.file_size>420_000_000:flush();staged=0
            raw=z.read(member);name=member.filename;digest=hashlib.sha256(raw).hexdigest()
            art=put('artifacts','member-'+name,Nome='Task41i original '+name,Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix()+'#'+name,
                    **{'SHA-256':digest,'Dimensione byte':len(raw)})
            obs=put('observations','raw-'+name,Nome='Task41i complete original JSON '+name,Descrizione=raw.decode('utf-8'),**{'Artefatti dettagliati':[aid,art]})
            progress['raw'].append(dict(member=name,observation=obs,sha256=digest,bytes=len(raw)));staged+=len(raw);del raw
            print('Task41i staged '+name+' '+str(member.file_size),flush=True)
        if staged:flush()
        assert {r['member'] for r in progress['raw']}=={x.filename for x in members}
    obs=put('observations','paired-audit',Nome='Task41i same-panel full audit',Descrizione=json.dumps(audit,ensure_ascii=False),**{'Artefatti dettagliati':[aid]})
    exp=ref('experiments-giada-task41i-factorial-v1')
    findings=[]
    for suffix,title,statement in [
        ('outcome','41i scientific NO-GO','I010 has6 failures versus4 for the SAME-panel H control. No primary cases resolved; additional Ca stress and Nap autonomous response failure. Old H development totals are not matched.'),
        ('numerics','Frozen numerical policy resolves response failures','Joint64/increment32 have4 total failures: all autonomous responses pass, but3stress and1development fail. Diagnostic, not a selectable promotion.'),
        ('cost','Whole operator slowed','At642perseed primary ionic time+22.7%,integrated+14.5% versus H. Compiled integrated still+8.9% versus formula and6quality failures. No speedup claim.'),
        ('strategy','Restricted repair did not unlock the system','Parent frozen; only2equilibria trainable, all z frozen. Negative evidence about THIS repair space; not a disproof of abstraction. DB precedent is hypothesis motivation, not causal proof.')]:
        findings.append(put('findings',suffix,Nome=title,Esito='Negativo' if suffix=='outcome' else 'Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=statement,Limitazioni='Opened regression diagnosis, not independent confirmation. Thresholds unchanged; original training and frozen selection retained.'))
    put('decisions','retain-H',Nome='Retain H; change learning strategy in41j',Esito='Continuare',Risultati=findings,Motivazione='Do not promote I010 or a numerical sidecar. User approved joint relearning versus compact operator at same abstraction and thresholds.')
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41i-factorial-v1','Stato':'Concluso'})
    spec=json.loads((repo/'experiments/task41i_preregistration_v1.json').read_text())
    for arm in spec['arms']:
        for seed in spec['seeds']:
            m.local_upsert('runs',{'Codice stabile':f"runs-giada-task41i-{arm['name']}-{seed}-v1",'Stato':'Completata',
                'Artefatti prodotti':[aid],'Validità tecnica':'60000 original finite updates; evaluation recovered without retraining/reselection; full final archive verified. Scientific SELECTIVE_REPAIR_NO_GO.'})
    flush();progress.update(complete=True);write_json(progress_path,progress)
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=baseline,records=ids,raw_reports=progress['raw'],
        raw_original_bytes_verified=True,completed_runs=24,airtable_accessed=False,DB_backup='local refresh pending'))
    print('Task41i all original JSON and source-linked result registered',flush=True)


if __name__=='__main__':main()
