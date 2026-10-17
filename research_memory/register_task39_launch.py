"""Record verified actual dispatch without creating scientific observations."""
import hashlib
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task39_protocol_v1';path=folder/'kaggle_launch.json'
    launch=json.loads(path.read_text());assert launch['status_at_verification']=='RUNNING' and launch['published_source_pin_verified']
    m=ValidatedBatchMirror()
    table=m.contract.tables['observations']['name']
    sql=f'''SELECT count(*) AS channels,
        count(DISTINCT o._record_id || '/' || a.key) AS cases
        FROM "{table}" o,
        json_each(CASE WHEN o."Codice stabile" LIKE 'observations-giada-task38-joint-transition-%'
            AND json_valid(o."Descrizione") THEN o."Descrizione" ELSE '{{}}' END, '$.cases') a,
        json_each(a.value, '$.metrics.channels') c'''
    coverage=m.query(sql)['rows'][0]
    assert coverage['channels']==4752 and coverage['cases']==432,coverage
    code='artifacts-giada-task39-learnability-kaggle-launch-v1'
    art=m.local_upsert('artifacts',{'Codice stabile':code,'Nome':'Task39 actual verified Kaggle dispatch','Tipo':'Log',
        'Percorso':path.relative_to(repo).as_posix(),'Versione':'v1','SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'Dimensione byte':path.stat().st_size})['record_id']
    from src.giada_teacher.task39_learnability import arms
    run_ids=[]
    for a in arms():
        run_ids.append(m.local_upsert('runs',{'Codice stabile':f'runs-giada-task39-learnability-{a["name"]}-v1',
            'Nome':'Task39 running '+a['name'],'Stato':'In corso','Artefatti prodotti':[art],
            'Configurazione effettiva':json.dumps(dict(launch=launch,arm=a,seeds=[17,29,43])),
            'Hardware e ambiente':'Kaggle v1 T4, full revision pinned; same parent runtime image. Actual torch/device in worker execution.json.',
            'Validità tecnica':'RUNNING/source pin/GPU verified at dispatch; scientific result still pending.'})['record_id'])
    m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task39-learnability-worker-v1',
        'Nome':'Task39 paired30-member worker','Commit':launch['revision']})
    print(json.dumps(dict(stage='validated_batch_ready',runs=len(run_ids))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'launch_memory_receipt.json',dict(valid=True,run_ids=run_ids,artifact=art,
        verification=v,task38_queryable_detail=coverage,airtable_accessed=False,scientific_result_claimed=False))
    print(json.dumps(dict(valid=True,runs=len(run_ids))),flush=True)


if __name__=='__main__':main()
