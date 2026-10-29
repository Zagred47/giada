"""Lossless41l evidence and scoped conclusions; single local SQLite writer."""
import gc,hashlib,json,zipfile
from datetime import datetime,timezone
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror

STATEMENTS={
 'outcome':'All8 repair arms fail global qualification. Same-panel H has3 failures (2Ca_HVA stress,1Nap autonomous); nominee L_A1D1P0 has6 (2Nap stress,3Nap autonomous,1Nap recurrence). Historical failures are additionally counted in the paired audit; L_A0D1P1 adds2. No Task42 or qualified speedup.',
 'local_separability':'Frozen K Ca_HVA-only transplant swap100 repairs both slow_boundary_return stress41c failures and preserves the measured stress41 and recurrence panels. Nap transplant induces Nap stress; K_Tst transplant induces K_Tst recurrence. These local harms are separable, but autonomous effects interact non-additively. Not a full-suite repair claim.',
 'feedback_numerics':'Selected frozen H/K responses fail in autonomous mode and pass with V imposed; Ca-only forcing is insufficient. Joint voltage-state/membrane-solve64 passes selected responses against matched-policy canonical reference. Numerical sensitivity was already known since41f: new information is channel-factorial attribution and matched-reference confounding, not first discovery. Full-suite and original-reference verification still required.',
 'objective_mismatch':'Nominee selection objective decreases52.8569% on average, but behavior regresses. Training feedback rollouts cover4ms while opened evaluation covers40ms. Training recurrent/solver path usesFP64 while primary inference isFP32. These are concrete contract differences, not yet causal proof of the training-failure mechanism.',
 'step_coverage_protection':'Lower LR reduces early finite-step overshoot, but no arm passes globally. Nested4x support repairs Ca stress in some arms with other regressions. Canonical worst-group protection is insufficient here. Gradient cosines are small/variable; no proof of universally conflicting gradients or impossible representation.',
 'cost':'Nominee paired median saved fraction: batch1 ionic+.42368/integrated+.28013; batch642 ionic-.19842/integrated-.205385; batch4096 ionic-2.5639/integrated-1.74494. Thus no qualified practical acceleration; do not combine fast inaccurate compact arms with accurate slow repair arms.',
 'recovery':'Final1,330,465,519-byte archive verified SHA256/ZIP CRC outside runtime; frozen checkpoint tensors finite. Colab recovery performed0 training updates and no reselection. Scientific NO-GO is not infrastructure failure.',
 'stop_policy':'User requests material progress or actionable discriminating information, not an indefinite series. Next41m is bounded frozen verification, no new training or compression. If it supplies neither full-panel repair nor a new concrete causal correction, stop this repair family and discuss direction; no automatic training extension.'}

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41l_audited_v1'
    audit=json.loads((folder/'audit.json').read_text());assert audit['valid']
    archive=repo/audit['source'];report=audit['report']
    progress_path=folder/'registration_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(raw=[],records={},batches=[])
    if progress.get('complete'):print('Already registered');return
    baseline=progress.get('verification',json.loads((repo/'experiments/results/task41l_protocol_v1/launch_submitted_memory_receipt.json').read_text())['verification'])
    m=IncrementalBatchMirror();assert m.baseline_hash==baseline['snapshot_hash'];ids=progress['records']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41l-result-{suffix}-v1'
        ids[code]=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];return ids[code]
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def flush():
        nonlocal m,baseline
        print('Commit and global integrity/reciprocal hash verification',flush=True)
        change=m.commit_batch();baseline=m.verify_incremental(baseline);assert baseline['valid']
        progress['verification']=baseline;progress['batches'].append(change);write_json(progress_path,progress)
        m=IncrementalBatchMirror();gc.collect();print('Verified reports '+str(len(progress['raw'])),flush=True)
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==audit['sha256']
    aid=put('artifacts','archive',Nome='Task41l complete final external archive',Tipo='Report',Versione='v1',Percorso=audit['source'],**{'SHA-256':audit['sha256'],'Dimensione byte':audit['bytes']})
    with zipfile.ZipFile(archive) as z:
        done={x['member'] for x in progress['raw']};staged=0
        members=sorted((x for x in z.infolist() if x.filename.endswith('.json')),key=lambda x:x.file_size)
        for member in members:
            if member.filename in done:continue
            if staged and staged+member.file_size>300_000_000:flush();staged=0
            raw=z.read(member);name=member.filename;sha=hashlib.sha256(raw).hexdigest()
            art=put('artifacts','member-'+name,Nome='Task41l original '+name,Tipo='Report',Versione='v1',Percorso=audit['source']+'#'+name,**{'SHA-256':sha,'Dimensione byte':len(raw)})
            obs=put('observations','raw-'+name,Nome='Task41l complete JSON '+name,Descrizione=raw.decode('utf-8'),**{'Artefatti dettagliati':[aid,art]})
            progress['raw'].append(dict(member=name,observation=obs,sha256=sha,bytes=len(raw)));staged+=len(raw);del raw
        if staged:flush()
        assert {x['member'] for x in progress['raw']}=={x.filename for x in members}
    obs=put('observations','audit',Nome='Task41l full paired audit and frozen causal probes',Descrizione=json.dumps(audit,ensure_ascii=False),**{'Artefatti dettagliati':[aid]})
    exp=ref('experiments-giada-task41l-regression-attribution-v1')
    limits='Opened diagnostic panels, not independent confirmation. Failures are evaluations, not unique biological cases. New development differs; same-run controls required. No threshold/abstraction waiver. Negative finite-budget evidence is not impossibility proof.'
    findings={key:put('findings',key,Nome='Task41l '+key,Esito='Negativo' if key=='outcome' else 'Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=text,Limitazioni=limits) for key,text in STATEMENTS.items()}
    date=datetime.now(timezone.utc).date().isoformat()
    for hypothesis,key in [('step','step_coverage_protection'),('coverage','step_coverage_protection'),('protection','step_coverage_protection'),('feedback','feedback_numerics'),('numerical-interaction','feedback_numerics')]:
        put('evidence',hypothesis,Nome='Task41l scoped assessment '+hypothesis,Esito='Supporta' if hypothesis in ('feedback','numerical-interaction') else 'Indeterminata',**{'Affermazione valutata':[ref('claims-giada-task41l-'+hypothesis+'-v1')],'Risultati a sostegno':[findings[key]],'Argomentazione':STATEMENTS[key],'Limiti e spiegazioni alternative':limits+' Support restricted to measured probes, not global sufficiency.','Data valutazione':date})
    runs=[]
    for arm in audit['chosen_steps']:
        for seed in (17,29,43):
            code=f'runs-giada-task41l-{arm}-{seed}-v1';runs.append(ref(code))
            m.local_upsert('runs',{'Codice stabile':code,'Stato':'Completata','Artefatti prodotti':[aid],'Validità tecnica':'4096 updates; frozen tensors finite; complete evaluation recovered with0new training/no reselection. Scientific NO-GO.'})
        put('observations','aggregate-'+arm,Nome='Task41l aggregate '+arm,Descrizione=json.dumps(dict(summary=report['summaries'][arm],failures=audit['failures'][arm],chosen_steps=audit['chosen_steps'][arm])),**{'Artefatti dettagliati':[aid],'Specifica di valutazione':[ref('evaluations-giada-task41l-quality-v1')]})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41l-regression-attribution-v1','Stato':'Concluso'})
    put('analyses','complete',Nome='Task41l complete audit',Origine='Post hoc',Versione='v1',Procedura='scripts/audit_task41l_final.py',Configurazione=limits,**{'Run analizzati':runs,'Input artefatti':[aid],'12 · Risultati scientifici':list(findings.values())})
    put('decisions','bounded-next',Nome='Bounded frozen41m; no automatic additional training',Esito='Continuare',Risultati=list(findings.values()),Motivazione=STATEMENTS['stop_policy'],**{'Data':date})
    put('lessons','contract-before-more-training',Nome='Local improvement is not coupled repair',Lezione='Test full-panel transfer of selective learned correction and both original/matched numerical references before more training. Query-derived patterns motivate tests but do not guarantee improvement. Preserve negative findings and separate compression.',Risultati=list(findings.values()),Limiti=limits)
    flush();progress['complete']=True;write_json(progress_path,progress)
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=baseline,records=ids,raw_reports=progress['raw'],completed_runs=24,airtable_accessed=False,backup='local refresh after41m preregistration'))
    print('41l evidence registered and globally verified',flush=True)

if __name__=='__main__':main()
