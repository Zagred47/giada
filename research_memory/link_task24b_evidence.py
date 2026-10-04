"""Link recorded observations to the six causal findings, plus confirmation."""
import json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror


def main():
    m=ValidatedBatchMirror();table=m.contract.tables['observations']['name']
    patterns={
      'annealing':['Task24b paired_contrasts %','Task24b development-%'],
      'clipping':['Task24b gradient_probes %','Task24b paired_contrasts %'],
      'budget_capacity':['Task24b paired_contrasts %','Task24b development-%'],
      'rate_attribution':['Task24b frozen_rate_precision_attribution %'],
      'precision':['Task24b frozen_rate_precision_attribution %'],
      'selection':['Task24b development-%'],
      'repair-and-sharing':['Task24b fresh-%']}
    counts={}
    for key,likes in patterns.items():
        query='SELECT _record_id AS id FROM "'+table+'" WHERE '+' OR '.join('Nome LIKE '+repr(p) for p in likes)
        ids=[];offset=0
        while True:
            rows=m.query(query+' ORDER BY _record_id LIMIT 10000 OFFSET '+str(offset),limit=10000)['rows']
            ids.extend(r['id'] for r in rows)
            if len(rows)<10000:break
            offset+=10000
        assert ids
        m.local_upsert('findings',{'Codice stabile':f'findings-giada-roadmap-task24b-{key}-v1','Osservazioni':ids});counts[key]=len(ids)
    for arm in ('frozen_shared_heads','frozen_conditioned'):
        m.local_upsert('arms',{'Codice stabile':f'arms-giada-roadmap-task24b-{arm}-v1','Ruolo':'Baseline'})
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());assert m.verify()['valid']
    print(json.dumps(dict(valid=True,linked_observation_counts=counts)))


if __name__=='__main__':main()
