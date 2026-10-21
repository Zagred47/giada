"""Preserve complete automatic confirmation results; attribution audit pending."""
import hashlib,json,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41c_completed_v1';folder.mkdir(exist_ok=True)
    external=repo/'artifacts/task41c_primary_external'
    receipt=json.loads((external/'external_backup_receipt.json').read_text());archive=Path(receipt['local'])
    assert receipt['valid'] and receipt['runtime_independent'] and receipt['remote']['label']=='final'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        read=lambda n:json.loads(z.read(n))
        report=read('final_report.json');spec=read('preregistration.json');freeze=read('candidate_contract.json')
        assert report['valid'] and not report['training_performed'] and not report['thresholds_changed']
        assert not report['quality_passed'] and report['decision']=='FROZEN_P110_INDEPENDENT_SCIENTIFIC_NO_GO'
        assert read('native_interface_report.json')['valid'] and read('gpu_preflight.json')['valid']
        assert hashlib.sha256(z.read('source_checkpoint.pt')).hexdigest()==freeze['checkpoint_sha256']==spec['sources'][spec['checkpoint']]
        assert hashlib.sha256((repo/'experiments/task41c_preregistration_v1.json').read_bytes()).hexdigest()==freeze['protocol_sha256']
        assert not freeze['fresh_accessed'] and not freeze['model_selection_performed']
        assert all(r['passed'] for r in read('native_current_phase_floor.json'))
        assert all(r['passed'] for mode in read('formula_floor.json').values() for rows in mode.values() for r in rows)
        assert all(r['passed'] for r in read('fresh_stress_results.json')['formula_floor'])
        assert all(all(row['finite'].values()) for row in read('cost.json'))
        for name in ('final_report.json','revision.json','candidate_contract.json','gpu_preflight.json'):(folder/name).write_bytes(z.read(name))
        write_json(folder/'backup_receipt.json',receipt)
        m=ValidatedBatchMirror();ids={};obs=[]
        def put(table,suffix,**fields):
            code=f'{table}-giada-task41c-completed-{suffix}-v1'
            rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
        def ref(code):
            rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
            return rows[0]['record_id']
        art=put('artifacts','archive',Nome='Task41c final external archive',Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix(),
            **{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':archive.stat().st_size})
        recorder=Path(__file__)
        put('artifacts','recorder',Nome='Task41c full automatic result registrar',Tipo='Codice',Versione='v1',Percorso=recorder.relative_to(repo).as_posix(),
            **{'SHA-256':hashlib.sha256(recorder.read_bytes()).hexdigest(),'Dimensione byte':recorder.stat().st_size})
        obs.append(put('observations','external-backup',Nome='Task41c verified final external backup',Descrizione=json.dumps(receipt),**{'Artefatti dettagliati':[art]}))
        for name in z.namelist():
            if not name.endswith('.json'):continue
            value=read(name)
            obs.append(put('observations',name,Nome='Task41c full '+name,Descrizione=json.dumps(value,ensure_ascii=False,separators=(',',':')),**{'Artefatti dettagliati':[art]}))
    exp=ref('experiments-giada-task41bc-41c-v1')
    finding=put('findings','automatic-no-go',Nome='Task41c valid independent P110 scientific NO-GO',Esito='Negativo',Esperimenti=[exp],Osservazioni=obs,
        Risultato=json.dumps(report,ensure_ascii=False),Limitazioni='Automatic registered evaluation,not detailed attribution audit. Native/interface/FP32-64 floors pass;candidate and protocol hashes verified. No thresholds changed,no training or selection. All cases retained. Single-compartment conditional teacher confirmation;noTask42authorization. Full JSON metrics queryable inside SQLite;arrays inverifiedexternalarchive.')
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41bc-41c-v1','Stato':'Concluso','Descrizione':'Completed independently;technical controls pass;P110528/528bothmodes,336/336imposedresponses,334/336autonomousresponses,31/36stress. ScientificNO_GO;detailedfailureauditpending.'})
    for name in ('P110','B6','formula','LUT_control'):
        for seed in spec['model_seeds']:m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41bc-41c-{name}-{seed}-v1','Stato':'Completata','Artefatti prodotti':[art],
            'Validità tecnica':'Complete exit0;native/interface/formulafloorsvalid;frozencheckpointandprotocolhashesvalid;no training;fullrawmetrics stored. ScientificNO_GO,P110notpromoted.'})
    put('decisions','retain-no-go',Nome='Task41c NO-GO retained; audit before next intervention',Esito='Continuare',Risultati=[finding],
        Motivazione='Do not advance Task42 or silently waive2autonomousresponsefailures/5stressfailures. Preserveallresultsandprobe causesbefore any further experiment.',**{'Data':'2026-10-21'})
    print(json.dumps(dict(stage='staged',records=len(ids),raw_reports=len(obs))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,full_report_observations=len(obs),verification=v,airtable_accessed=False,DBbackup='localonly',detailed_attribution_audit_pending=True))
    print('Task41c complete automatic results registered and verified',flush=True)
if __name__=='__main__':main()
