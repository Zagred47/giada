"""One verified SQLite batch: full48 results/limits and prospective49 protocol."""
import argparse,gzip,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent
    old=json.loads((repo/'experiments/results/task48_protocol_v1/preregistration_receipt.json').read_text());audit_path=repo/'experiments/results/task48_audit_v1/audit.json';audit=json.loads(audit_path.read_text());plan=json.loads((repo/'experiments/task49_preregistration_v1.json').read_text());archive=repo/'artifacts/task48_kaggle_external/task48_final.zip'
    with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==audit['archive_sha256']
    assert archive.stat().st_size==audit['archive_bytes'] and audit['valid'];folder=repo/'experiments/results/task49_protocol_v1';folder.mkdir(exist_ok=True)
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();baseline={} if args.validate_only else old['verification'];ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,task,key,**fields):
        code=f'{table}-giada-task{task}-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp48=old['records']['experiments-giada-task48-state-diagnostics-v1'];obs=[];arts=[]
    arts.append(put('artifacts','48-result','archive',Nome='48 complete runtime-independent archive',Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix(),**{'SHA-256':audit['archive_sha256'],'Dimensione byte':audit['archive_bytes']}))
    dest=repo/'experiments/results/task48_audit_v1'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        report=json.loads(z.read('final_report.json'));assert report['complete'] and report['valid']
        for name in z.namelist():
            if not name.endswith('.json'):continue
            raw=z.read(name);target=dest/(name+'.gz')
            with target.open('wb') as f:
                with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz:gz.write(raw)
            art=put('artifacts','48-result',name,Nome='48 raw '+name,Tipo='Report',Versione='v1',Percorso=target.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(target.read_bytes()).hexdigest(),'Dimensione byte':target.stat().st_size});arts.append(art)
            obs.append(put('observations','48-result',name,Nome='48 full raw '+name,Descrizione=raw.decode(),**{'Artefatti dettagliati':[art]}))
    obs.append(put('observations','48-result','audit',Nome='48 complete collision/recovery/family/precision audit',Descrizione=json.dumps(audit),**{'Artefatti dettagliati':arts}))
    obs.append(put('observations','48-result','canonical-slow-timescales',Nome='48 canonical slow gate times at -65mV',Descrizione=json.dumps(dict(voltage_mv=-65.,calcium_mM=5e-5,tau_ms={'Nap_Et2.h':2192.013379649131,'Ca_HVA.h':449.16658819863034,'K_Pst.h':409.4405797955013},source='ionic_block_teacher_forced.rates; canonical formulas, not fitted P estimates'))))
    for model,channels in audit['collision_summary'].items():
        for channel,value in channels.items():obs.append(put('observations','48-result','collision-'+model+'-'+channel,Nome='48 collision '+model+' '+channel,Descrizione=json.dumps(value)))
    for key,value in audit['local_observability_spectra'].items():obs.append(put('observations','48-result','sensitivity-'+key,Nome='48 sensitivity '+key,Descrizione=json.dumps(value)))
    limits=' '.join(audit['limitations'])
    finding=put('findings','48-result','state-information',Nome='48 instantaneous openings lose future-relevant state',Esito='Positivo',Esperimenti=[exp48],Osservazioni=obs,Risultato=json.dumps(dict(constructed_pairs=21,pair_drive_combinations=63,canonical_combinations_above_shared_10percent=23,collision_summary=audit['collision_summary'],local_spectra=audit['local_observability_spectra'])),Limitazioni=limits)
    uncertainty=put('findings','48-result','scope-and-precision',Nome='48 canonical corroboration and unresolved reachable compression',Esito='Misto',Esperimenti=[exp48],Osservazioni=obs,Risultato=json.dumps(dict(paired_endpoint_errors=audit['paired_endpoint_errors'],history_error=audit['history_gate_max_rmse'],recovery=audit['recovery_endpoint_gaps'],no_nontrivial_exact_reachable_collision_established=True)),Limitazioni=limits)
    for key,statement,state in [('H1','Constructed exact opening collisions have different futures in all seven two-gate channels;23/63 pair/drive combinations exceed10percent shared-scale gap.','Supportata nel dominio'),('H2','Exact coincident reachable representations did not reveal distinct futures in this bank. Approximate neighbours are not an impossibility proof; claim remains open.','Aperta'),('H3','Recovery/sensitivity probes expose slow persistent memory, especially Nap_Et2.h and K_Pst.h. Endpoint PCA after rest is not global state dimension.','Supportata nel dominio'),('H4','FP64/FP32 agree on qualitative collisions; P broadly agrees but has quantitative deviations, including5.738percent diagnostic endpoint error in hyperpolarization. This is not a registered full-trajectory failure rate.','Supportata nel dominio')]:
        claim=put('claims','48',key,Stato=state,Limiti=statement+' '+limits)
        put('evidence','48-result',key,Nome='48 evidence '+key,Esito='Indeterminata' if key=='H2' else 'Sostiene',**{'Affermazione valutata':[claim],'Risultati a sostegno':[finding,uncertainty],'Argomentazione':statement,'Limiti e spiegazioni alternative':limits,'Data valutazione':'2026-11-15'})
    put('experiments','48','state-diagnostics',Stato='Concluso')
    for arm in ('canonical64','canonical32','P'):put('runs','48',arm,Stato='Completata',**{'Artefatti prodotti':arts,'Validità tecnica':'Finite states, unchanged P hash, exact restart-state control and canonical parity; archive SHA and every ZIP CRC verified.'})
    put('lessons','48-result','washout-and-memory',Nome='48 do not compress from post-rest PCA alone',Lezione='A long shared rest erases fast-gate/SK variability. Sample active states before rest and separate true history sufficiency from privileged physical-state encoding. Instantaneous opening equality does not imply a Markov state; preserve slow recovery and Ca explicitly.',Risultati=[finding,uncertainty],Limiti=limits)
    exp=put('experiments','49','initialization',Nome='Task49 causal initialization before and after rest',Tipo='Esplorativo',Stato='Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Separate state-available encoding loss, finite causal warmup, inherited P error and rest-induced apparent redundancy; no compressed recurrent release.'})
    protocol=put('protocols','49','initialization',Nome='49 matched initialization matrix with frozen P',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Criteri di successo':'Complete all fixed arms/roles/cuts with unchanged P, exact restart/full-prefix controls and copied quality limits. No fresh-based arm choice; no coupled/event release.','Regole di arresto':'Hash/parity/nonfinite/control failure or runtime/output budget preserves partial reports; no threshold adjustment.'})
    claims=[put('claims','49',h['id'],Nome='49 '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['claim'],Limiti=h['contrast']+' '+plan['limitations']) for h in plan['hypotheses']];put('experiments','49','initialization',Ipotesi=claims)
    for arm in plan['arms']+['canonical64','canonical32']:
        br=put('arms','49',arm,Nome='49 '+arm,Ruolo='Baseline' if arm in ('canonical64','canonical32','P_history_oracle') else 'Trattamento',Protocollo=[protocol]);put('runs','49',arm,Nome='49 '+arm,Stato='Pianificata',Braccio=[br],Seed='P17/29/43; independent fit9491103/development9491127/fresh9491141; cuts300/350/400ms')
    put('decisions','49','proceed',Nome='Proceed49 with pre-rest causal initialization matrix',Esito='Continuare',Motivazione='Explicit user approval after48 audit. Keep frozenP and original limits; fit fixed PCA widths on fit states only, no P training/selection, warmup only past V/Ca, physical identity privileged. Dynamic compressed recurrence stays inTask50.',Risultati=[finding,uncertainty],**{'Data':'2026-11-15'})
    print('48 full results and49 protocol staged in one SQLite transaction',flush=True);m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];receipt=dict(valid=True,verification=v,records=ids,airtable_accessed=False,task48_complete=True,task49_preregistered=True,thresholds_changed=False)
    write_json(folder/'preregistration_receipt.json',receipt);write_json(dest/'memory_receipt.json',receipt);m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('48 results +49 preregistration SQLite verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('48-result/49 schema valid; no DB writes')
