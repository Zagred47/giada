"""Preserve original dispatched protocol and explain unused inherited fields."""
import hashlib,json
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
    print('Task41d metadata erratum staged',flush=True);m.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'metadata_memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False,DBbackup='localonly',scientific_protocol_unchanged=True))
    print('Task41d metadata erratum registered and verified',flush=True)
if __name__=='__main__':main()
