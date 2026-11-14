"""Read exact scientific facts through the payload-v2 SQL adapter."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.payload_store import open_readonly
from research_memory.mirror import write_json
if __name__=='__main__':
    folder=ROOT/'experiments/results/task47_cahva_audit_v1';receipt=json.loads((folder/'memory_receipt.json').read_text());assert receipt['valid']
    queries={
        'hypotheses':('SELECT "Codice stabile", Nome, Stato FROM claims WHERE "Codice stabile" LIKE ? ORDER BY "Codice stabile"',('claims-giada-task47-cahva-audit-%',)),
        'critical_samples':('SELECT json_extract(Descrizione,\'$.reference[1983]\') AS reference_ma_cm2, json_extract(Descrizione,\'$.imposed_seed17[1983]\') AS imposed_ma_cm2, json_extract(Descrizione,\'$.autonomous_seed17[1983]\') AS autonomous_ma_cm2 FROM observations WHERE "Codice stabile"=?',('observations-giada-task47-cahva-audit-critical_traces.json-v1',)),
        'diagnosis':('SELECT json_extract(Risultato,\'$.factual_diagnosis\') AS diagnosis,json_extract(Risultato,\'$.threshold_samples[0].weak_peak_relative_error\') AS weak_peak_relative_error,json_extract(Risultato,\'$.official_task47_verdict_unchanged\') AS original_verdict FROM findings WHERE "Codice stabile"=?',('findings-giada-task47-cahva-audit-mechanism-v1',))}
    with open_readonly(ROOT/'research_memory/data/research_memory.sqlite') as con:
        data={key:dict(sql=sql,parameters=params,rows=[dict(row) for row in con.execute(sql,params)]) for key,(sql,params) in queries.items()}
    assert len(data['hypotheses']['rows'])==5 and len(data['critical_samples']['rows'])==1 and len(data['diagnosis']['rows'])==1
    point=data['critical_samples']['rows'][0];assert abs(point['reference_ma_cm2']-(-7.18801748007536e-5))<1e-15
    assert 'NO_GO' in data['diagnosis']['rows'][0]['original_verdict']
    write_json(folder/'query_readback.json',dict(valid=True,airtable_accessed=False,queries=data));print(json.dumps(data,ensure_ascii=False),flush=True)
