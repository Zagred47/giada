"""Full immutable-cohort multi-event comparison, controls and scoped conclusions."""
import argparse,gzip,hashlib,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task47_multi_event_v1'
    s=json.loads((folder/'summary.json').read_text());i=json.loads((folder/'interpretation.json').read_text());freeze=json.loads((folder/'freeze.json').read_text());plan=json.loads((repo/'experiments/task47_multi_event_evaluation_v1.json').read_text())
    assert s['valid'] and i['valid'] and freeze['valid'] and s['complete'] and not s['thresholds_changed'] and not s['training_performed'] and s['original_events_and_counts_reproduced']
    for name,digest in freeze['source_sha256'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest
    for name,proof in s['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==proof['sha256'] and (folder/name).stat().st_size==proof['bytes']
    prior=json.loads((repo/'experiments/results/task47_cahva_audit_v1/memory_receipt.json').read_text());old=json.loads((repo/'experiments/results/task47_protocol_v1/preregistration_receipt.json').read_text())
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();baseline={} if args.validate_only else prior['verification'];ids={};obs=[];arts=[]
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=table+'-giada-task47-multi-event-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    names=['freeze.json','summary.json','case_results.json','failure_audit.json','channel_results.json.gz','event_details.json.gz','saved_reference_controls.json.gz','new_support_audit.json','interpretation.json']
    for name in names:
        path=folder/name;raw=path.read_bytes();art=put('artifacts',name,Nome='47 multi-event '+name,Tipo='Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)});arts.append(art)
        text=gzip.decompress(raw).decode('utf-8') if name.endswith('.gz') else raw.decode('utf-8')
        obs.append(put('observations',name,Nome='47 multi-event full '+name,Descrizione=text,**{'Artefatti dettagliati':[art]}))
    obs.append(put('observations','locked-analysis-plan',Nome='47 post-hoc evaluator plan fixed before full cohort',Descrizione=json.dumps(plan)))
    for key,row in s['summary'].items():obs.append(put('observations','case-summary-'+key,Nome='47 original versus multi-event '+key,Descrizione=json.dumps(row)))
    for key,row in s['negative_controls'].items():obs.append(put('observations','control-'+key,Nome='47 saved-reference control '+key,Descrizione=json.dumps(row)))
    exp=old['records']['experiments-giada-task47-confirmation-v1'];limits=s['limitations']+' '+i['interpretation']
    good=put('findings','implementation-controls',Nome='47 multi-event implemented controls pass on the specified domain',Esito='Positivo',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(dict(identities=i['identity_controls_passed'],known_bad_rejected=i['negative_mutants_rejected'],synthetic_tests=10,formula_null_control=s['formula_null_control_passed'],complete_original_reproduction=s['original_events_and_counts_reproduced'])),Limitazioni=limits)
    comparison=put('findings','comparison',Nome='47 multi-event changes interpretation, not a repaired model',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(i),Limitazioni='Original P143/144at500/1000ms per mode retained. Separate multi-event P143/144imposed and140/144autonomous. No old result overwritten or new training. '+limits)
    for key,state,text,reason in [
        ('control-rejection','Supportata nel dominio','Frozen episode evaluator rejects all validly constructed known-bad controls and accepts all identity controls.','5206/5206negative mutants rejected;1313/1313identity controls accepted;10synthetic tests pass; full formula-null lanes pass. Original also rejected all constructed bad mutants, so this is not evidence that original lacked basic mutation sensitivity.'),
        ('support-not-offset','Supportata nel dominio','Ca_HVA remaining mismatch is classified as episode support, not44.9ms displacement between matched events.','Matched event errors0;1unmatched reference single-sample episode preserved as failure; no event dropped to obtain a pass.'),
        ('automatic-quality-go','Contraddetta nel dominio','Changing to the frozen multi-event evaluator qualifies P on all saved47cases.','P remains143/144imposed and becomes140/144autonomous at500/1000ms under the new observable; no144/144or task48qualification.'),
        ('threshold-topology','Supportata nel dominio','Counting connected threshold episodes can split a near-threshold response without proving a new biological event.','SK_E2case9autonomous has2/1/1mask-disagreement samples for17/29/43, producing a short initial component plus the main component. Main matched onset shift0.3ms meets original tolerance; strict spurious-component gate still fails.')]:
        claim=put('claims',key,Nome='47 multi-event '+key,Tipo='Ipotesi',Stato=state,Enunciato=text,Limiti=reason+' '+limits)
        put('evidence',key,Nome='47 multi-event evidence '+key,Esito='Sostiene' if state.startswith('Supportata') else 'Contraddice',**{'Affermazione valutata':[claim],'Risultati a sostegno':[good,comparison],'Argomentazione':reason,'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-11-15'})
    put('analyses','full-cohort-replay',Nome='47 all-trace original versus frozen multi-event replay',Origine='Post hoc',Versione='v1',Procedura='freeze_task47_multi_event.py; run_task47_multi_event.py; audit_task47_multi_event_changes.py; summarize_task47_multi_event.py',Configurazione=json.dumps(plan),**{'Input artefatti':arts,'12 · Risultati scientifici':[good,comparison]})
    put('lessons','do-not-confuse-topology-with-physiology',Nome='Mutation-test success does not settle biological event identity',Lezione='A stricter multi-event metric may uncover support fragmentation missed by global extrema without any model regression. Distinguish implementation correctness, threshold-connected components, amplitude approximation and actual biological responses. Keep known bad/identity controls and all case identities; do not tune smoothing or erase single-sample excursions after inspecting the full cohort.',Risultati=[good,comparison],Limiti=limits)
    put('decisions','retain-separate-verdicts',Nome='Retain original47verdict and separate multi-event interpretation',Esito='Continuare',Motivazione='User authorized all-saved-trace evaluator comparison and controls with frozenP, unchanged tolerances, no training. Completed and recorded without automatically qualifying48. Event-identity contract remains a scoped scientific decision; no post-result metric adjustment performed.',Risultati=[good,comparison],**{'Data':'2026-11-15'})
    print('47 multi-event full reports and scoped findings staged',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/'memory_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False,original_verdict_retained=True,thresholds_changed=False,training_performed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('47 multi-event SQLite verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('47 multi-event schema valid; no writes')
