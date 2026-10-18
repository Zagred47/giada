"""Preserve failed attempt, register technical retry; SQLite only."""
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher.task39e_tail_support_heads import arms,CONFIG

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task39e_retry_v1'
    launch=json.loads((folder/'launch.json').read_text());spec=json.loads((repo/CONFIG).read_text())
    m=ValidatedBatchMirror();ids={}
    def put(table,code,**fields):
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    artifact=put('artifacts','artifacts-giada-task39e-retry1-console-v1',Nome='Task39e technical retry console and external recovery',Tipo='Log',Versione='retry1',Percorso='experiments/results/task39e_retry_v1/launch.json')
    for arm in arms():
        for seed in spec['seeds']:
            old=f'runs-giada-task39e-tail-{arm["name"]}-seed{seed}-v1'
            put('runs',old,Stato='Fallita',**{'Validità tecnica':'Native exit1 at60000 updates. Original exact traceback unavailable after VM release; no scientific NO-GO. CLI history retained locally. Do not treat as completed.'})
            put('runs',old+'-retry1',Nome=f'Task39e retry1 {arm["name"]} seed{seed}',Stato='In corso',Seed=str(seed),**{'Artefatti prodotti':[artifact],'Configurazione effettiva':json.dumps(dict(arm=arm,seed=seed,launch=launch)),'Hardware e ambiente':'Colab T4 official CLI, PC-side SHA/CRC recovery supervisor','Validità tecnica':'Technical retry; unchanged scientific preregistration. Native worker cause not yet attributed. No completion claimed.'})
    put('analyses','analyses-giada-task39e-technical-failure-v1',Nome='Task39e technical failure, not scientific NO-GO',Origine='Post hoc',Versione='v1',Procedura='experiments/task39e_technical_retry.md',Configurazione=json.dumps(dict(original_exit=1,last_update=60000,original_cause='undetermined',scientific_no_go=False,diagnostic_fp64_only=True,preregistration_unchanged=True)))
    print(json.dumps(dict(staged=len(ids))),flush=True);m.commit_batch()
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
