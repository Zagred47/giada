"""SQL readback of immutable original and separate multi-event case counts."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.payload_store import open_readonly
from research_memory.mirror import write_json
F=ROOT/'experiments/results/task47_multi_event_v1'
if __name__=='__main__':
    receipt=json.loads((F/'memory_receipt.json').read_text());assert receipt['valid']
    queries={
        'P_original_vs_multi':('SELECT Nome,json_extract(Descrizione,\'$.original_passed\') AS original_passed,json_extract(Descrizione,\'$.multi_event_passed\') AS multi_event_passed,json_extract(Descrizione,\'$.total\') AS total FROM observations WHERE "Codice stabile" LIKE ? ORDER BY Nome',('observations-giada-task47-multi-event-case-summary-%/P-v1',)),
        'negative_controls':('SELECT Nome,json_extract(Descrizione,\'$.total\') AS total,json_extract(Descrizione,\'$.as_expected\') AS as_expected,json_extract(Descrizione,\'$.unexpected\') AS unexpected FROM observations WHERE "Codice stabile" LIKE ? ORDER BY Nome',('observations-giada-task47-multi-event-control-%-v1',)),
        'scoped_claims':('SELECT Nome,Stato FROM claims WHERE "Codice stabile" LIKE ? ORDER BY Nome',('claims-giada-task47-multi-event-%-v1',))}
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as con:data={k:dict(sql=sql,parameters=parameters,rows=[dict(r) for r in con.execute(sql,parameters)]) for k,(sql,parameters) in queries.items()}
    assert len(data['P_original_vs_multi']['rows'])==6 and len(data['negative_controls']['rows'])==5 and len(data['scoped_claims']['rows'])==4
    assert all(r['unexpected']==0 for r in data['negative_controls']['rows'])
    write_json(F/'query_readback.json',dict(valid=True,airtable_accessed=False,queries=data));print(json.dumps(data,ensure_ascii=False),flush=True)
