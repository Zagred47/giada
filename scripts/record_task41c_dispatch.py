"""Record evidenced launch, extra provenance and scoped41b evidence; SQLite only."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import ROOT as MEMORY,write_json
from research_memory.validated_batch import ValidatedBatchMirror

def main():
    folder=ROOT/'experiments/results/task41c_protocol_v1';external=ROOT/'artifacts/task41c_primary_external'
    request=json.loads((folder/'dispatch_request.json').read_text());status=json.loads((external/'job_status.json').read_text())
    backup=json.loads((external/'external_backup_receipt.json').read_text());spec=json.loads((ROOT/'experiments/task41c_preregistration_v1.json').read_text())
    assert request['revision']==status['revision'] and backup['valid'] and backup['runtime_independent']
    assert not status.get('technical_failure',False)
    archive=Path(backup['local']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==backup['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('source_checkpoint.pt')).hexdigest()==spec['sources'][spec['checkpoint']]
        contract=json.loads(z.read('candidate_contract.json'));assert not contract['fresh_accessed'] and not contract['training_performed']
    launch=dict(valid=True,launched=True,session='giada-task41c-primary',revision=request['revision'],status=status,external_backup=backup,
        DBbackup='localonly',thresholds_changed=False,result_pending_audit=True)
    if (folder/'launcher_note.json').exists():launch['launcher_history']=json.loads((folder/'launcher_note.json').read_text())
    write_json(folder/'launch.json',launch)
    m=ValidatedBatchMirror();arts=[]
    paths=[folder/'dispatch_request.json',folder/'launch.json',folder/'gpu_preflight.json',folder/'launcher_note.json',ROOT/'scripts/record_task41c_dispatch.py',ROOT/'scripts/prepare_task41c_dispatch.py',ROOT/'scripts/task41c_gpu_probe.py',ROOT/'scripts/prepare_task41c_preflight.py',ROOT/'scripts/launch_task41c_console.ps1']
    for path in paths:
        arts.append(m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task41c-dispatch-'+path.name+'-v1','Nome':'Task41c '+path.name,
            'Tipo':'Codice' if path.suffix=='.py' else 'Report','Versione':'v1','Percorso':path.relative_to(ROOT).as_posix(),
            'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})['record_id'])
    done=status.get('state')=='finished'
    for name in ('P110','B6','formula','LUT_control'):
        for seed in spec['model_seeds']:
            m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41bc-41c-{name}-{seed}-v1','Stato':'Completata' if done else 'In corso','Artefatti prodotti':arts,
                'Configurazione effettiva':json.dumps(dict(model=name,seed=seed,seed_kind='original frozen model seed' if name in ('P110','B6') else 'deterministic replica label,not new training',modes=['imposed','autonomous'],training=False,selection=False,input_seeds=spec['native_seeds'],stress_seeds=spec['stress_seeds'],thresholds_changed=False)),
                'Hardware e ambiente':json.dumps(dict(gpu='TeslaT4',session=launch['session'],revision=request['revision'])),
                'Validità tecnica':'Actualconsoleworkerstarted;frozencandidatecheckpointrecoveredoutsideVM SHA/CRCvalid. State sampled inlaunchreceipt;scientificresultpendingaudit.'})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41bc-41c-v1','Stato':'Concluso' if done else 'In corso'})
    m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41bc-41c-v1','Commit':request['revision']})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task41bc-41c-launch-v1','Stato':'Conclusa','Descrizione':'Verified consolelaunch andexternalfrozencheckpointbackup;actualtimestampedstatus inlaunchreceipt. No scientificpromotions.'})
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    finding=ref('findings-giada-task41bc-41b-result-v1')
    for name in ('coverage','capacity','objective_conditioning','budget'):
        m.local_upsert('evidence',{'Codice stabile':'evidence-giada-task41bc-'+name+'-v1','Nome':'Task41b scoped factorial '+name,'Esito':'Indeterminata',
            'Affermazione valutata':[ref('claims-giada-task41-audit-'+name+'-v1')],'Risultati a sostegno':[finding],
            'Argomentazione':'P110 passescompleteopenedbattery;no12conditionalcontrastimprovesworstseverityall3seedsunderregisteredcriterion. Noisolatedmaincausalattribution;budgetcurvesnotuniversalproof.',
            'Limiti e spiegazioni alternative':'SupportchangesFITrealization;selectionranktimingquantized;otherfactorsinteract;newindependentconfirmationpending.','Data valutazione':'2026-10-21'})
    print('Task41c verified launch staged',flush=True)
    m.commit_batch();write_json(MEMORY/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    check=m.query("SELECT count(*) n FROM v_records WHERE table_key='runs' AND stable_code LIKE 'runs-giada-task41bc-41c-%'")['rows'][0]
    assert check['n']==12
    write_json(folder/'launch_memory_receipt.json',dict(valid=True,verification=v,launched=True,run_readback=check,revision=request['revision'],result_pending_audit=True,airtable_accessed=False,DBbackup='localonly'))
    print('Task41c launch registered and verified',flush=True)
if __name__=='__main__':main()
