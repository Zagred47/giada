"""Preserve protocol erratum and verified post-training freeze recovery."""
import hashlib,json
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41d_protocol_v1';path=folder/'metadata_erratum.json';err=json.loads(path.read_text())
    assert hashlib.sha256((repo/'experiments/task41d_preregistration_v1.json').read_bytes()).hexdigest()==err['original_protocol_sha256']
    assert not err['model_or_training_changed'] and not err['thresholds_changed'] and not err['scientific_run_restarted']
    m=ValidatedBatchMirror();ids={}
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41d-metadata-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    art=put('artifacts','erratum',Nome='Task41d immutable protocol metadata erratum',Tipo='Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    observation=put('observations','erratum',Nome='Task41d old inactive metadata versus actual pinned runtime bindings',Descrizione=json.dumps(err),**{'Artefatti dettagliati':[art]})
    finding=put('findings','documentary',Nome='Task41d inherited narrative defect;source and active preregistration unchanged',Esito='Misto',Esperimenti=[ref('experiments-giada-task41d-factorial-v1')],Osservazioni=[observation],Risultato='Inactive41bdescriptivefields retainedbydictinheritance. Actual41dsource reads new_seeds,arms,training_policy/objectives andfixednewselection. Fullbindingmap preserved;originalprotocolSHA unchanged.',Limitazioni='Documentarycorrection afterlaunch;no retrospective changeofcandidate,arms,seeds,thresholds,training,selectionoroutcomes. Audit mustconsider originalprotocolandthiserratum together.')
    put('lessons','whitelist',Nome='Whitelist shared protocol fields; preserve dispatched protocol with explicit errata',Lezione=err['future_rule'],Risultati=[finding],Limiti='Source-preregistered runtime was unambiguous; narrative ambiguity is still a documentation defect that must not be silently erased.')
    recovery=repo/'artifacts/task41d_primary_external/freeze_recovery_receipt.json'
    rr=json.loads(recovery.read_text());assert rr['valid'] and rr['training_updates']==60000 and all(v['all_finite'] for v in rr['files'].values())
    ra=put('artifacts','freeze-recovery',Nome='Task41d local60k/frozen checkpoint verification receipt',Tipo='Report',Versione='v1',Percorso=recovery.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(recovery.read_bytes()).hexdigest(),'Dimensione byte':recovery.stat().st_size})
    ro=put('observations','freeze-recovery',Nome='Task41d60k and selected weights finite,off-runtime',Descrizione=json.dumps(rr),**{'Artefatti dettagliati':[ra],'Specifica di valutazione':[ref('evaluations-giada-task41d-optimizer-v1')]})
    put('findings','training-complete',Nome='Task41d60k training complete;frozen artifacts verified externally',Esito='Positivo',Esperimenti=[ref('experiments-giada-task41d-factorial-v1')],Osservazioni=[ro],Risultato='36members completed60kupdates;156frozencheckpointtensorsand4optimizer/checkpointtensorsfinite;freezeandcheckpointSHAverified;bothweightsand60kstateonthePC.',Limitazioni='Technicaltrainingcompletion,notqualityGO. Openedregressionsandcostevaluationmustfinishandbeaudited. NoTask42authorization.')
    for name,kind in [('training_curves.json','selection'),('development_60000.json','quality'),('fit_calibration.json','optimizer')]:
        file=recovery.parent/name;value=json.loads(file.read_text())
        artifact=put('artifacts',name,Nome='Task41d externally recovered '+name,Tipo='Report',Versione='v1',Percorso=file.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(file.read_bytes()).hexdigest(),'Dimensione byte':file.stat().st_size})
        put('observations',name,Nome='Task41d full '+name,Descrizione=json.dumps(value,separators=(',',':')),**{'Artefatti dettagliati':[artifact],'Specifica di valutazione':[ref('evaluations-giada-task41d-'+kind+'-v1')]})
    recorder=Path(__file__)
    put('artifacts','recorder',Nome='Task41d metadata and post-training recovery registrar',Tipo='Codice',Versione='v1',Percorso=recorder.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(recorder.read_bytes()).hexdigest(),'Dimensione byte':recorder.stat().st_size})
    print('Task41d metadata erratum staged',flush=True);m.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'metadata_memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False,DBbackup='localonly',scientific_protocol_unchanged=True,freeze_recovery=rr))
    print('Task41d metadata erratum registered and verified',flush=True)
if __name__=='__main__':main()
