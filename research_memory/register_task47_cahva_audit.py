"""Complete47 raw results and saved-trace Ca_HVA diagnosis; SQLite only."""
import argparse,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task47_cahva_audit_v1';runfolder=repo/'experiments/results/task47_retry_v1'
    audit=json.loads((folder/'audit.json').read_text());tests=json.loads((folder/'tests.json').read_text());assert audit['valid'] and tests['valid']
    assert not any(audit[k] for k in ('thresholds_changed','training_performed','selection_performed','model_outputs_modified','evaluator_code_modified'))
    old=json.loads((repo/'experiments/results/task47_protocol_v1/preregistration_receipt.json').read_text());prior=json.loads((runfolder/'preregistration_receipt.json').read_text())
    receipt=json.loads((repo/'artifacts/task47_retry_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent'] and not receipt['technical_failure']
    archive=repo/'artifacts/task47_retry_external/task47_final.zip'
    with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==receipt['remote']['sha256']==audit['archive_sha256'] and archive.stat().st_size==receipt['remote']['bytes']
    print('47 final archive SHA verified; validating CRC and staging complete reports',flush=True)
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();baseline={} if a.validate_only else prior['verification'];ids={};observations=[]
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=table+'-giada-task47-cahva-audit-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    art=put('artifacts','final-archive',Nome='47 corrected run final complete verified archive',Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix(),**{'SHA-256':digest,'Dimensione byte':archive.stat().st_size})
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'));assert report['task']=='47' and report['executed_protocol_verified'] and report['all_parameters_finite'] and report['all_traces_finite']
        assert not report['independent_quality_confirmation_passed'] and report['summary']['1000.0']['P']['autonomous']==dict(total=144,eligible=144,passed=143)
        for name in sorted(z.namelist()):
            if name.endswith('.json'):
                observations.append(put('observations','raw-'+name,Nome='47 corrected run full '+name,Descrizione=z.read(name).decode('utf-8'),**{'Artefatti dettagliati':[art]}))
    artifact_ids=[art]
    names=['audit.json','trace_audit_initial.json','critical_traces.json','tests.json','cahva_threshold_audit.png','case8_node4_evidence.npz']
    for name in names:
        path=folder/name;data=path.read_bytes();aid=put('artifacts',name,Nome='47 Ca_HVA forensic '+name,Tipo='Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(data).hexdigest(),'Dimensione byte':len(data)});artifact_ids.append(aid)
        if name.endswith('.json'):observations.append(put('observations',name,Nome='47 Ca_HVA forensic '+name,Descrizione=data.decode('utf-8'),**{'Artefatti dettagliati':[aid,art]}))
    comparison=(runfolder/'comparison.json').read_text();observations.append(put('observations','paired-comparison',Nome='47 complete geometry/current/intervention/cost paired audit',Descrizione=comparison,**{'Artefatti dettagliati':[art]}))
    observations.append(put('observations','verified-external-backup',Nome='47 final archive SHA and ZIP CRC verified locally',Descrizione=json.dumps(dict(receipt=receipt,sha256_recomputed=digest,all_zip_members_crc_valid=True,archive_runtime_independent=True)),**{'Artefatti dettagliati':[art]}))
    exp=old['records']['experiments-giada-task47-confirmation-v1']
    limits='Post-hoc diagnostic, unchanged original143/144NO-GO at500/1000ms per mode. No threshold/evaluator/weight change, training, reselection, automatic144/144, task48authorization or speedup claim. Small model amplitude error remains real; local peak agreement is not complete biological equivalence.'
    findings={}
    findings['outcome']=put('findings','outcome',Nome='47 technically valid independent run; one original timing case fails',Esito='Negativo',Esperimenti=[exp],Osservazioni=observations,Risultato=json.dumps(report),Limitazioni=limits)
    findings['mechanism']=put('findings','mechanism',Nome='Ca_HVA44.9ms reflects global-event support switch, not44.9ms delay',Esito='Misto',Esperimenti=[exp],Osservazioni=observations,Risultato=json.dumps(audit),Limitazioni=limits)
    findings['cost']=put('findings','cost',Nome='47 accuracy transfer mostly retained; coupled execution still about2.07x slower',Esito='Negativo',Esperimenti=[exp],Osservazioni=observations,Risultato=comparison,Limitazioni='Fixed common-state component timing is nonadditive, not whole autonomous rollout runtime. Implementation still computes masked canonical ionic branches; not a lower bound on abstraction. '+limits)
    facts=[
        ('physical-delay','Contraddetta nel dominio','The44.9ms value denotes a real temporal displacement of the final Ca_HVA response.','The saved final weak peaks coincide at198.3ms for all3seeds in imposed/autonomous mode; the two dominant active islands have identical boundaries. The original offset subtracts198.3-153.4ms between different islands.'),
        ('support-switch','Supportata nel dominio','A marginal amplitude mismatch switches the aggregate last-active index between separate response islands.','Exactly one active-mask disagreement, sample1983: target fraction0.1000071256 of global peak; P0.0999374383. Copying only that candidate sample from the saved reference removes the44.9ms aggregate offset, without changing any threshold.'),
        ('roundoff-only','Contraddetta nel dominio','The failure is solely float32 rounding in the evaluator.','The amplitude difference is6884referencefloat32ULPs in imposed mode; float64 evaluation reproduces44.9ms. Thus a real small approximation error, not solely evaluator arithmetic roundoff.'),
        ('new-coupling-error','Contraddetta nel dominio','The discrepancy is newly introduced by P repair or axial feedback.','The failed case has coupling0 and fails also under imposed V/Ca. Seed17 imposed Ca_HVA current arrays are bitwise identical across C1/P/R41.'),
        ('small-amplitude-error','Supportata nel dominio','A small real amplitude approximation is sufficient to toggle support for the final borderline response.','Weak peak amplitude error is about0.06968percent; onset/global peak timing unchanged. A temporal-support discontinuity amplifies this into the reported44.9ms.')]
    for key,state,claim,argument in facts:
        rid=put('claims',key,Nome='47 Ca_HVA '+key,Tipo='Ipotesi',Stato=state,Enunciato=claim,Limiti=argument+' '+limits)
        put('evidence',key,Nome='47 scoped saved-trace evidence '+key,Esito='Sostiene' if state.startswith('Supportata') else 'Contraddice',**{'Affermazione valutata':[rid],'Risultati a sostegno':[findings['mechanism']],'Argomentazione':argument,'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-11-14'})
    m.local_upsert('claims',{'Codice stabile':'claims-giada-task47-H1-v1','Stato':'Contraddetta nel dominio','Limiti':'Strict all-case confirmation did not pass:143/144at500/1000ms. The saved-trace audit explains one borderline support case, not a waived gate. '+limits})
    put('analyses','saved-trace-forensic',Nome='47 Ca_HVA saved-trace event-identity forensic',Origine='Post hoc',Versione='v1',Procedura='audit_task47_cahva.py; finalize_task47_cahva_audit.py; validate_task47_cahva_audit.py',Configurazione=json.dumps(audit['numerical_definition']),**{'Input artefatti':artifact_ids,'12 · Risultati scientifici':list(findings.values())})
    put('lessons','multi-event-support',Nome='Separate aggregate threshold support from matched-event timing',Lezione='A one-transient global first/last-active-index applied to a multi-pulse trajectory can turn a tiny amplitude error near the threshold into a large apparent delay by switching event identity. Audit masks, signed threshold margins, pulse-local peaks and known-delay controls before adding training or capacity. Preserve missing/support-mismatched events explicitly; do not silently remove them or change acceptance thresholds.',Risultati=[findings['mechanism']],Limiti=limits)
    put('decisions','retain-original-gates',Nome='Retain47originalNO-GO; no training from apparent44.9ms delay',Esito='Continuare',Motivazione='User requested saved-trace audit only. ArchiveSHA/CRC and raw evidence verified, diagnostic complete. Keep original gates and outputs. A prospective matched-event evaluator with separate support mismatch requires explicit specification and testing; not implemented or used to grant a GO here.',Risultati=list(findings.values()),**{'Data':'2026-11-14'})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task47-confirmation-v1','Stato':'Concluso'})
    for arm in ('none','C1','P','R41'):
        for mode in ('imposed','autonomous'):
            m.local_upsert('runs',{'Codice stabile':'runs-giada-task47-'+arm+'-'+mode+'-v1','Stato':'Completata','Artefatti prodotti':[art],'Validità tecnica':'Corrected47entrypoint; phase identities, native floors, null/parity and finite checks valid. First misrouted attempt retained separately as technical invalidity. Scientific original gates unchanged.'})
    print('47 raw results, traces, hypotheses and audit staged',flush=True);m.commit_batch()
    if a.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid'];write_json(folder/'memory_receipt.json',dict(valid=True,verification=verification,records=ids,airtable_accessed=False,original_task47_verdict_unchanged=True));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('47 forensic audit SQLite verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('47 Ca_HVA audit schema valid; no writes')
