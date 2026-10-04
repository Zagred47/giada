"""Read-only cross-check of recorded evidence and Task25 preregistration."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_memory.mirror import Mirror,ROOT


def main():
    m=Mirror();counts=m.query("SELECT table_key,COUNT(*) AS count FROM v_records WHERE stable_code LIKE '%-giada-roadmap-task24b-%' GROUP BY table_key",limit=100)['rows']
    assert next(r['count'] for r in counts if r['table_key']=='observations')==28713
    claims=m.query("SELECT stable_code,name FROM v_records WHERE table_key='claims' AND stable_code LIKE 'claims-giada-roadmap-task25-%'",limit=100)['rows'];assert len(claims)==6
    findings=m.query("SELECT stable_code,record_id FROM v_records WHERE table_key='findings' AND stable_code LIKE 'findings-giada-roadmap-task24b-%'",limit=100)['rows'];assert len(findings)==7
    table=m.contract.tables['findings']['name']
    links=m.query('SELECT "Codice stabile" AS code,json_array_length("Osservazioni") AS count FROM "'+table+'" WHERE "Codice stabile" LIKE \'findings-giada-roadmap-task24b-%\'',limit=100)['rows']
    assert all(r['count']>0 for r in links)
    verification=m.verify();assert verification['valid']
    report=dict(valid=True,task24b_records=counts,findings_observation_links=links,task25_claims=claims,sqlite_verification=verification,airtable_accessed=False)
    dest=ROOT.parent/'experiments/preflight/task25_heterogeneous/memory_audit.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(valid=True,observations=28713,findings=7,task25_claims=6)))


if __name__=='__main__':main()
